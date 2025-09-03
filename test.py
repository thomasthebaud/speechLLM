from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
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
    version = f"{model_config['model_name']}epoch-epoch={model_config['epoch_to_test']}"
    model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_config['model_name']}/{version}.ckpt")
    tokenizer = model.llm_tokenizer
    trainer = Trainer(
        accelerator='gpu', devices=1, log_every_n_steps=100
    )
    print("Model loaded")

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for test_set in model_config['test_sets']:
        log_dir = f"exp/test_predictions/{model_config['model_name']}/{test_set}"
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

        logging.info(f"Testing {test_set}")

        test_dataset = InstructionalAudioDataset(
            csv_file=f'./data/{test_set}.csv',
            mode='test', 
            max_len=model_config['max_number_seconds'],
            fields=model_config['test_sets'][test_set]
        )
        my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=model_config['batch_size'], shuffle=False, collate_fn=my_collator, num_workers=3)

        trainer.test(model=model, dataloaders=test_loader)

        #kill logger
        logger.removeHandler(file_handler)
        file_handler.close()
    