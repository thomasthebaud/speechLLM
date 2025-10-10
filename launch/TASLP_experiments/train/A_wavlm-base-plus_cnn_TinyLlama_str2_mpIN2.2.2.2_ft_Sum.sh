#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=tri_S #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/TASLP/train/A_wavlm-base-plus_cnn_TinyLlama_str2_mpIN2.2.2.2_ft_Sum_%j.log
#SBATCH --output=logs/TASLP/train/A_wavlm-base-plus_cnn_TinyLlama_str2_mpIN2.2.2.2_ft_Sum_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str1.2.1_inmp2.2.2.2' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --lr 0.0005 \
    --encoder-lr 0.0001 \
    --ft-encoder \
    --meanpool 1 \
    --group 'TALSP' \
    --use-config summarize_switchboard.json \
    --total-training-epoch 200
