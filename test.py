from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset

import torch.utils.data as data_utils
from dataset import InstructionalAudioDataset, MyCollator
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder')  
    parser.add_argument('--connector')  
    parser.add_argument('--llm')  
    parser.add_argument('--connector-k', default=2)
    parser.add_argument('--connector-dim', default=512)
    parser.add_argument('--batch-size', default=16)
    parser.add_argument("--no-lora", action='store_true')

    args = parser.parse_args()
    batch_size = int(args.batch_size)
    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{args.llm}"
    if args.no_lora: model_name = model_name+'_nolora'
    use_lora = not args.no_lora
    if "wavlm" in args.encoder: audio_encoder_name=args.encoder
    else: exit(f"Uknown encoder reference: {args.encoder}")

    connector_name=args.connector
    if args.llm=='TinyLlama-1.1B-Chat-v1.0':llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    batch_size = int(args.batch_size)

    model_config = {
                'audio_enc_dim': 768, 
                'llm_dim': 2048, 
                'audio_encoder_name': audio_encoder_name, 
                'connector_name': connector_name,
                'llm_name': llm_name,
                'finetune_encoder': False,
                'connector_k': int(args.connector_k),
                'connector_dim': int(args.connector_dim),
                'use_lora': use_lora,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': 1e-4 if 'linear' not in connector_name else 1e-5,
                'total_training_step': 10000000,
                'warmup_steps': 100,
                'train_batch_per_epoch': 80000//batch_size,
                'val_batch_per_epoch': 1000//batch_size,
                'grad_accumulate_steps': 8
        }

    # model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_name}/last.ckpt")
    model = SpeechLLMLightning.load_from_checkpoint("checkpoints/logs/wavlm-base-plus-cnn-TinyLlama-1.1B-Chat-v1.0-epoch=56.ckpt")
    tokenizer = model.llm_tokenizer

    test_dataset = InstructionalAudioDataset(
        csv_file='data/test.csv', # same train.csv and dev.csv
        mode='test'
        )
    
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collator, num_workers=3)
    
    trainer = Trainer(
        accelerator='gpu', devices=1
    )
    trainer.test(model=model, dataloaders=test_loader)
    