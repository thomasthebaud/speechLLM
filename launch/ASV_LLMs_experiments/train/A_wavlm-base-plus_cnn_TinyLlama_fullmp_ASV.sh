#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=tr_ASV #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/ASV_LLM/train/A_wavlm-base-plus_cnn_TinyLlama_fullmp_ASV_%j.log
#SBATCH --output=logs/ASV_LLM/train/A_wavlm-base-plus_cnn_TinyLlama_fullmp_ASV_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export TOKENIZERS_PARALLELISM=false
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train_ASV.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str1.2.1' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 4 \
    --lr 0.00001 \
    --meanpool -1 \
    --group 'ASV' \
    --use-config ASV_voxceleb1.json \
    --total-training-epoch 20 \
    --nickname "_ASV"

