#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name=train_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/train_wavlm-base-plus_mlp1_TinyLlama.log
#SBATCH --output=logs/train_wavlm-base-plus_mlp1_TinyLlama.log
#SBATCH --exclude=e03

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'linear' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --connector-k '20' \
    --batch-size 128
