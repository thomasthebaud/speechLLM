

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset, MyCollator, CompositeAudioDataset
from pytorch_lightning.strategies import DDPStrategy

import torch.utils.data as data_utils
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb
import argparse
import os

import torch


def make_weighted_sampler_from_dataset(dataset, dtype=torch.double):
    """
    Build a WeightedRandomSampler from a CompositeAudioDataset.
    - If dataset contains only one sub-dataset: return None (use shuffle=True outside).
    - If multiple sub-datasets: expand dataset.datasets_weights into per-sample weights.
    
    Args:
        dataset: CompositeAudioDataset or similar object containing .dataset.datasets and .datasets_weights
        dtype: torch dtype for the weights tensor (default: torch.double)
    
    Returns:
        WeightedRandomSampler if multiple sub-datasets, otherwise None.
    """
    # Get the list of sub-datasets (in case of ConcatDataset)
    subs = getattr(getattr(dataset, "dataset", None), "datasets", None)

    # If not multiple sub-datasets, return None (then set shuffle=True)
    if not (isinstance(subs, (list, tuple)) and len(subs) > 1):
        return None

    sizes = [len(d) for d in subs]
    dataset_weights = getattr(dataset, "datasets_weights", None)
    if dataset_weights is None or len(dataset_weights) != len(sizes):
        raise ValueError("datasets_weights is missing or does not match the number of sub-datasets.")

    # Expand per-dataset weight to per-sample weight
    weights_per_sample = torch.cat([
        torch.full((sz,), float(w) / sz, dtype=dtype)
        for w, sz in zip(dataset_weights, sizes)
    ])
    assert len(weights_per_sample) == len(dataset), "weights length must match the total number of samples."

    return data_utils.WeightedRandomSampler(weights_per_sample,
                                  num_samples=len(weights_per_sample),
                                  replacement=True)


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
    parser.add_argument("--ft-encoder", action='store_true')
    parser.add_argument("--use-summaries", action='store_true')


    args = parser.parse_args()
    lr = float(args.lr)
    batch_size = int(args.batch_size)

    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{args.llm.split('-')[0]}-bs{batch_size}"
    if args.use_summaries: model_name = model_name+'_sum'
    if args.no_lora: model_name = model_name+'_nolora'
    if args.ft_encoder: model_name = model_name+'_ft_encoder'
    if lr == 1.0: lr = 1e-4 if 'linear' not in args.connector else 1e-5
    model_name =  f"{model_name}_lr{lr}"
    log_path = 'logs/'+model_name
    use_lora = not args.no_lora

    wandb.init(project="speechllm", name=log_path, group="July experiments")
    logger = WandbLogger(project="speechllm", name=log_path, group="July experiments")

    if "wavlm" in args.encoder: 
        audio_encoder_name=args.encoder
        audio_enc_dim = 768
    elif 'MFCC' in args.encoder:
        audio_encoder_name=args.encoder
        audio_enc_dim = 80
    else: exit(f"Uknown encoder reference: {args.encoder}")
    
    if args.llm=='TinyLlama-1.1B-Chat-v1.0':llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    datasets = {
    "train":['librispeech_train-clean-100', 'iemocap_ses01-03', 'voxceleb1_dev', 'CV-EN_train', 'MSP_Podcast_Train', 'voxceleb2_enriched_dev'],
    "dev":['librispeech_dev-clean', 'iemocap_ses04', 'voxceleb1_test', 'CV-EN_dev', 'MSP_Podcast_Validation', 'voxceleb2_enriched_test'],
    "test":['librispeech_test-clean', 'iemocap_ses05', 'voxceleb1_test', 'CV-EN_test', 'MSP_Podcast_Test', 'voxceleb2_enriched_test'],
    }
    datasets = {
    "train":['librispeech_train-clean-100', 'librispeech_train-clean-360', 'iemocap_ses01-03', 'voxceleb2_enriched_dev'],
    "dev":['librispeech_dev-clean', 'librispeech_dev-other', 'iemocap_ses04', 'voxceleb2_enriched_test'],
    }
    if args.use_summaries: 
        datasets['train'] = ['switchboard_train', 'librispeech_train-clean-360']
        datasets['dev'] = ['switchboard_val', 'librispeech_dev-clean', 'librispeech_dev-other']
        datasets['test'] = ['switchboard_test', 'librispeech_test-clean', 'librispeech_test-other']
        # datasets['train'].append('switchboard_train')
        # datasets['dev'].append('switchboard_val')
        # datasets['test'].append('switchboard_test')

    model_config = {
                'audio_enc_dim':audio_enc_dim, 
                'llm_dim': 2048, 
                'audio_encoder_name': audio_encoder_name, 
                'connector_name': args.connector,
                'llm_name': llm_name,
                'finetune_encoder': args.ft_encoder,
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
                'max_number_seconds': args.truncate_sec,
                'train_batch_per_epoch': 4096,
                'train_sets':datasets['train'],
                'dev_sets':datasets['dev'],
                'max_size_per_dev_set':50
        }   
    
    model = SpeechLLMLightning(**model_config)
    tokenizer = model.llm_tokenizer

    train_dataset = CompositeAudioDataset(
        list_of_datasets=model_config['train_sets'],
        mode='train', 
        random_keys_prob=0.2,
        max_len=model_config['max_number_seconds']
        )

    val_dataset = CompositeAudioDataset(
        list_of_datasets = model_config['dev_sets'],
        mode='test',
        max_len=model_config['max_number_seconds'],
        max_size=model_config['max_size_per_dev_set']
        )

    print(f"Train set:{len(train_dataset)}, val set:{len(val_dataset)}, batch size:{batch_size}")
    num_workers=3 #put to 0 for debugging
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    # sampler = data_utils.WeightedRandomSampler(train_dataset.datasets_weights, num_samples=len(train_dataset.datasets_weights), replacement=True)
    # Check whether it is generated by Composite / ConcatDataset
    sampler = make_weighted_sampler_from_dataset(train_dataset)
    shuffle = sampler is None  # If sampler is used, shuffle must be False
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=my_collator, num_workers=num_workers)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collator, num_workers=num_workers)

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
            strategy=DDPStrategy(find_unused_parameters=args.ft_encoder),
            limit_train_batches=model_config['train_batch_per_epoch'], 
            log_every_n_steps=1, 
            enable_checkpointing=True, 
            callbacks=[checkpoint_callback],
            fast_dev_run=False, logger=logger, 
            accumulate_grad_batches=model_config['grad_accumulate_steps']
    )
    trainer.fit(model, train_loader, val_loader)

