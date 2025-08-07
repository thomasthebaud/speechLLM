#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --job-name=train_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/train_wavlm-base-plus_cnn_TinyLlama_lr1e-4_ft_e_%j.log
#SBATCH --output=logs/train_wavlm-base-plus_cnn_TinyLlama_lr1e-4_ft_e_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

# echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --lr 0.0001 \
    --ft-encoder
