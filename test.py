from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset

import torch.utils.data as data_utils
from dataset import InstructionalAudioDataset, MyCollator, CompositeAudioDataset
from utils import get_model_config
import os
import shutil
import logging

if __name__ == "__main__":
    model_config = get_model_config()
    print(model_config)
    # model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_name}/last.ckpt")
    if model_config['epoch_to_test']>0:
        version = f"{model_config['model_name']}epoch-epoch={model_config['epoch_to_test']}"
        i=0
        while os.path.exists(f"checkpoints/{model_config['model_name']}/{version}-v{i+1}.ckpt"):
            i+=1
        print(f"Using epoch {model_config['epoch_to_test']} version {i}")
        if i==0: model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_config['model_name']}/{version}.ckpt")
        else:    model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_config['model_name']}/{version}-v{i}.ckpt")
    else:
        print("Using untrained model!")
        version='base'
        model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer
    trainer = Trainer(
        accelerator='gpu', devices=1, log_every_n_steps=100, callbacks=[TQDMProgressBar(refresh_rate=50)]
    )
    print("Model loaded")

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if model_config['test_on']=='AT': text_options = [("AT", True, True), ("T", True, False), ("A", False, True)]
    elif model_config['test_on']=='T': text_options = [('T', True, False)]
    elif model_config['test_on']=='A': text_options = [('A', False, True)]
    # text_options = [('A', False, True)]
    # text_options = [("AT", True, True), ("T", True, False), ("A", False, True)]

    for test_set in model_config['test_sets']:
        for suffix, use_text, use_audio in text_options:
            log_dir = f"exp/test_predictions/{model_config['model_name']}/{test_set}/{suffix}"
            try:
                os.makedirs(log_dir)
            except:
                print(f"{log_dir} exists")
            #update logger
            logger.handlers.clear()
            file_handler = logging.FileHandler(f"{log_dir}/{version}.txt", mode="w")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            print(f"Testing {test_set} using {suffix}")

            test_dataset = InstructionalAudioDataset(
                csv_file=f'./data/{test_set}.csv',
                mode='test', 
                max_len=model_config['max_number_seconds'],
                fields=model_config['test_sets'][test_set],
                use_text=use_text,
                prob_text=1,
                data_name=test_set
            )
            my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
            test_loader = data_utils.DataLoader(test_dataset, 
                    batch_size=model_config['batch_size'], 
                    shuffle=False, 
                    collate_fn=my_collator, 
                    num_workers=3)
            trainer.use_audio=use_audio
            trainer.test(model=model, dataloaders=test_loader)

            #kill logger
            logger.removeHandler(file_handler)
            file_handler.close()
    