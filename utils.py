import argparse
import json

def get_config(args):
    if args.use_config is None: #use default config
        datasets = {
        "train":['librispeech_train-clean-100', 'iemocap_ses01-03', 'CV-EN_train', 'MSP_Podcast_Train', 'voxceleb2_enriched_dev'],
        "dev":['librispeech_dev-clean', 'iemocap_ses04', 'CV-EN_dev', 'MSP_Podcast_Validation', 'voxceleb2_enriched_test'],
        "test":['librispeech_test-clean', 'iemocap_ses05', 'CV-EN_test', 'MSP_Podcast_Test', 'voxceleb2_enriched_test'],
        }
        datasets = {split:{data:[] for data in datasets[split]} for split in datasets} #empty list means use all available fields for that dataset
    else:
        with open(f'config/{args.use_config}', 'r') as file:
            datasets = json.load(file)

    use_summaries = False
    for data in datasets['dev']:
        if "summary" in datasets['dev'][data]:
            use_summaries=True
            break
    return datasets, use_summaries


def get_model_config():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder')  
    parser.add_argument('--connector')  
    parser.add_argument('--llm')  
    parser.add_argument('--connector-k', default=2, type=int)
    parser.add_argument('--connector-dim', default=512, type=int)
    parser.add_argument('--connector-layers', default=1, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--truncate-sec', default=-1, type=int)
    parser.add_argument('--lr', default=1.0)
    parser.add_argument('--encoder-lr', default=-1)
    parser.add_argument("--no-lora", action='store_true')
    parser.add_argument("--ft-encoder", action='store_true')
    parser.add_argument("--use-text", action='store_true')
    parser.add_argument("--prob-text", default=0.5, type=float)
    parser.add_argument("--no-audio", action='store_true')
    parser.add_argument('--epoch-to-test', default=1, type=int)
    parser.add_argument("--meanpool", default=1, type=int)
    parser.add_argument("--total-training-epoch", default=1000, type=int)
    parser.add_argument("--use-config", default=None, type=str)
    parser.add_argument("--group", default='August experiments', type=str)
    parser.add_argument("--nickname", default='_', type=str)
    parser.add_argument("--test-on", default='A', type=str)

    args = parser.parse_args()

    # Training Parameters
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    use_lora = not args.no_lora

    # Model naming
    model_name = f"{args.encoder.split('/')[-1]}-{args.connector}-{args.llm.split('-')[0]}-bs{batch_size}"
    if args.no_lora: model_name = model_name+'_nolora'

    if args.use_text:
        model_name = model_name + f"_p{float(args.prob_text)}"
        if args.no_audio: model_name = 'T_'+model_name
        else: model_name = 'AT_'+model_name
    else:
        if args.no_audio: exit("not using text nor audio!")
        else: model_name = 'A_'+model_name

    if args.ft_encoder: model_name = model_name+'_ft_encoder'
    if args.encoder_lr==-1: args.encoder_lr = float(args.lr)/50
    if args.meanpool!=1: model_name = model_name+f'_mp{args.meanpool}'
    if lr == 1.0: lr = 1e-4 if 'linear' not in args.connector else 1e-5
    if args.connector=='cnn':
        stride = int(args.connector_k)*(int(args.connector_k)//2)*(int(args.connector_k)//2)
        model_name = model_name+f'_str{stride}'
    model_name =  f"{model_name}_lr{lr}"

    if args.nickname!='_': model_name = model_name + str(args.nickname)
    # Wandb params
    log_path = 'logs/'+model_name
    group = args.group

    # Encoder
    if "wavlm" in args.encoder: 
        audio_encoder_name=args.encoder
        audio_enc_dim = 768
    elif 'MFCC' in args.encoder:
        audio_encoder_name=args.encoder
        audio_enc_dim = 80
    else: exit(f"Uknown encoder reference: {args.encoder}")
    
    # LLM
    if args.llm=='TinyLlama-1.1B-Chat-v1.0':llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Datasets
    datasets, use_summaries = get_config(args)

    # Get all infos
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
                'meanpool':int(args.meanpool),
                'use_lora': use_lora,
                'use_text':args.use_text,
                'prob_text':float(args.prob_text),
                'use_audio':not args.no_audio,
                'lora_r': 8,
                'lora_alpha': 16,
                'max_lr': lr,
                'enc_lr': args.encoder_lr,
                'batch_size':batch_size,
                'total_training_epoch': int(args.total_training_epoch),
                'warmup_steps': 100,
                'grad_accumulate_steps': 64//batch_size,
                'max_number_seconds': args.truncate_sec,
                'train_batch_per_epoch': 4096,
                'train_sets':datasets['train'],
                'dev_sets':datasets['dev'],
                'test_sets':datasets['test'],
                'max_size_per_dev_set':50,
                'log_path':log_path,
                'group':group,
                'model_name':model_name,
                'epoch_to_test':int(args.epoch_to_test),
                'use_summaries':use_summaries,
                'test_on':str(args.test_on)
        }
    return model_config