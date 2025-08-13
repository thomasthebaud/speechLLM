from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from trainer import SpeechLLMLightning
from dataset import InstructionalAudioDataset

import torch.utils.data as data_utils
from dataset import InstructionalAudioDataset, MyCollator, CompositeAudioDataset
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder')  
    parser.add_argument('--connector')  
    parser.add_argument('--llm')  
    parser.add_argument('--connector-k', default=2)
    parser.add_argument('--connector-dim', default=512)
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--lr', default=1.0)
    parser.add_argument("--no-lora", action='store_true')
    parser.add_argument("--ft-encoder", action='store_true')
    parser.add_argument("--use-summaries", action='store_true')

    args = parser.parse_args()
    batch_size = int(args.batch_size)

    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{args.llm}"
    if args.use_summaries: model_name = model_name+'_sum'
    if args.no_lora: model_name = model_name+'_nolora'
    if args.ft_encoder: model_name = model_name+'_ft_encoder'
    if lr == 1.0: lr = 1e-4 if 'linear' not in args.connector else 1e-5
    model_name =  f"{model_name}_lr{lr}"
    log_path = 'logs/'+model_name
    use_lora = not args.no_lora

    if "wavlm" in args.encoder: 
        audio_encoder_name=args.encoder
        audio_enc_dim = 768
    elif 'MFCC' in args.encoder:
        audio_encoder_name=args.encoder
        audio_enc_dim = 80
    else: exit(f"Uknown encoder reference: {args.encoder}")
    
    if args.llm=='TinyLlama-1.1B-Chat-v1.0':llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    datasets = ['librispeech_test-clean', 'iemocap_ses05', 'voxceleb1_test', 'CV-EN_test', 'MSP_Podcast_Test', 'voxceleb2_enriched_test']
    if args.use_summaries: datasets = ['switchboard_test', 'librispeech_test-clean', 'librispeech_test-other']

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
                'test_sets':datasets,
                'max_size_per_dev_set':50
        }   
    print(model_config)
    # model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/{model_name}/last.ckpt")
    model = SpeechLLMLightning.load_from_checkpoint(f"checkpoints/logs/{model_name}/last.ckpt")
    tokenizer = model.llm_tokenizer

    test_dataset = CompositeAudioDataset(
        list_of_datasets=model_config['test_sets'],
        mode='test', 
        max_len=model_config['max_number_seconds']
        )
    
    my_collator = MyCollator(model_config['audio_encoder_name'], tokenizer)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collator, num_workers=3)
    
    trainer = Trainer(
        accelerator='gpu', devices=1
    )
    trainer.test(model=model, dataloaders=test_loader)
    