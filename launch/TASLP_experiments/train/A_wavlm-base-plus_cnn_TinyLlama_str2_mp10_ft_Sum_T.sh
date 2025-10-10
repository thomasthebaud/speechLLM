#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=ft_ST #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/TASLP/train/A_wavlm-base-plus_cnn_TinyLlama_str2_mp10_ft_Sum_T_%j.log
#SBATCH --output=logs/TASLP/train/A_wavlm-base-plus_cnn_TinyLlama_str2_mp10_ft_Sum_T_%j.log
#SBATCH --nodelist=e05

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str1.2.1' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --lr 0.0005 \
    --encoder-lr 0.0001 \
    --ft-encoder \
    --meanpool 10 \
    --group 'TALSP' \
    --use-config summarize_switchboard_librispeech960.json \
    --total-training-epoch 200 \
    --nickname "_T"

