#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --job-name=train_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/train/wavlm-base-plus_linear_TinyLlama_lr1e-3_transcript_%j.log
#SBATCH --output=logs/train/wavlm-base-plus_linear_TinyLlama_lr1e-3_transcript_%j.log
#SBATCH --exclude=e05

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

# echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'linear' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --no-lora \
    --batch-size 8 \
    --truncate-sec 60 \
    --connector-k 20 \
    --lr 0.001 \
    --use-config librispeech_transcript_only.json \
    --group 'Lightweight_connectors_reproduction' \
    --total-training-epoch 10
