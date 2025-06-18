

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator
from pytorch_lightning.strategies import DDPStrategy

import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
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
    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{args.llm}"
    if args.no_lora: model_name = model_name+'_nolora'
    log_path = 'logs/'+model_name
    use_lora = not args.no_lora
    wandb.init(project="speechllm", name=log_path)
    logger = WandbLogger(project="speechllm", name=log_path)

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
    
    model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer

    train_dataset = InstructionalAudioDataset(
        csv_file = './data/train.csv',
        mode='train', 
        random_keys_prob=0.2,
        )

    val_dataset = InstructionalAudioDataset(
        csv_file='./data/dev.csv', 
        mode='test'
        )

    print(f"Train set:{len(train_dataset)}, val set:{len(val_dataset)}, batch size:{batch_size}")

    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    sampler = data_utils.WeightedRandomSampler(train_dataset.datasets_weights, num_samples=len(train_dataset.datasets_weights), replacement=True)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=my_collator, num_workers=3)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collator, num_workers=3)

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename=log_path+'-{epoch}', save_top_k=1, monitor="val/loss", save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=10, verbose=False, mode="min")

    trainer = Trainer(
            max_epochs=model_config['total_training_step']//model_config['train_batch_per_epoch'], 
            devices=1, accelerator="gpu", 
            strategy=DDPStrategy(find_unused_parameters=False),
            limit_train_batches=model_config['train_batch_per_epoch'], 
            limit_val_batches=model_config['val_batch_per_epoch'], 
            log_every_n_steps=100, 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback],
            fast_dev_run=False, logger=logger, 
            accumulate_grad_batches=model_config['grad_accumulate_steps']
    )
    trainer.fit(model, train_loader, val_loader)

