

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
from pytorch_lightning.strategies import DDPStrategy

import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder')  
    parser.add_argument('--connector')  
    parser.add_argument('--llm')  
    parser.add_argument('--connector-k', default=2, type=int)
    parser.add_argument('--connector-dim', default=512, type=int)
    parser.add_argument('--connector-layers', default=1, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--truncate-sec', default=60, type=int)
    parser.add_argument('--lr', default=1.0)
    parser.add_argument("--no-lora", action='store_true')
    parser.add_argument("--use-summaries", action='store_true')


    args = parser.parse_args()
    lr = float(args.lr)
    batch_size = int(args.batch_size)

    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{args.llm.split('-')[0]}-bs{batch_size}"
    if args.use_summaries: model_name = model_name+'_sum'
    if args.no_lora: model_name = model_name+'_nolora'
    if lr == 1.0: lr = 1e-4 if 'linear' not in args.connector else 1e-5
    model_name =  f"{model_name}_lr{lr}"
    log_path = 'logs/'+model_name
    use_lora = not args.no_lora
    wandb.init(project="speechllm", name=log_path)
    logger = WandbLogger(project="speechllm", name=log_path)

    if "wavlm" in args.encoder: 
        audio_encoder_name=args.encoder
        audio_enc_dim = 768
    elif 'MFCC' in args.encoder:
        audio_encoder_name=args.encoder
        audio_enc_dim = 80
    else: exit(f"Uknown encoder reference: {args.encoder}")

    
    if args.llm=='TinyLlama-1.1B-Chat-v1.0':llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    
    model_config = {
                'audio_enc_dim':audio_enc_dim, 
                'llm_dim': 2048, 
                'audio_encoder_name': audio_encoder_name, 
                'connector_name': args.connector,
                'llm_name': llm_name,
                'finetune_encoder': False,
                'connector_k': int(args.connector_k),
                'connector_dim': int(args.connector_dim),
                'connector_layers': int(args.connector_layers),
                'use_lora': use_lora,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': lr,
                'batch_size':batch_size,
                'total_training_epoch': 1000,
                'warmup_steps': 100,
                'grad_accumulate_steps': 128//batch_size,
                'max_number_seconds': args.truncate_sec
        }   
    
    model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer

    if args.use_summaries: train_file, dev_file = 'switchboard_train.csv', 'switchboard_val.csv'
    else: train_file, dev_file = 'train.csv', 'dev.csv'
    train_dataset = InstructionalAudioDataset(
        csv_file = f'./data/{train_file}',
        mode='train', 
        random_keys_prob=0.2,
        max_len=model_config['max_number_seconds']
        )

    val_dataset = InstructionalAudioDataset(
        csv_file=f'./data/{dev_file}', 
        mode='test',
        max_len=model_config['max_number_seconds']
        )

    print(f"Train set:{len(train_dataset)}, val set:{len(val_dataset)}, batch size:{batch_size}")

    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    sampler = data_utils.WeightedRandomSampler(train_dataset.datasets_weights, num_samples=len(train_dataset.datasets_weights), replacement=True)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=my_collator, num_workers=3)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collator, num_workers=3)

    checkpoint_callback = ModelCheckpoint(
                    dirpath=f"checkpoints/{model_name}", 
                    filename=model_name+'epoch-{epoch}', 
                    save_top_k=1, 
                    monitor="val/loss", 
                    save_last=True,
                    every_n_epochs=2)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")

    trainer = Trainer(
            max_epochs=model_config['total_training_epoch'], 
            devices=1, accelerator="gpu", 
            strategy=DDPStrategy(find_unused_parameters=False),
            # limit_train_batches=model_config['train_batch_per_epoch'], 
            log_every_n_steps=100, 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback],
            fast_dev_run=False, logger=logger, 
            accumulate_grad_batches=model_config['grad_accumulate_steps']
    )
    trainer.fit(model, train_loader, val_loader)

