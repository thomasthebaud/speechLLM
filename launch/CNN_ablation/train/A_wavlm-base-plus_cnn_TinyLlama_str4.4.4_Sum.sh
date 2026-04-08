#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=tr_S #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/CNN_ablation/train/A_wavlm-base-plus_cnn_TinyLlama_str4.4.4_Sum_%j.log
#SBATCH --output=logs/CNN_ablation/train/A_wavlm-base-plus_cnn_TinyLlama_str3.4.3_Sum_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str4.4.4' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 16 \
    --lr 0.001 \
    --meanpool 1 \
    --group 'CNN_ablation' \
    --use-config summarize_switchboard.json \
    --total-training-epoch 50 \
    --nickname 'not_pretrained'
