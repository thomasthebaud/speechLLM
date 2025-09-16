#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=test_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/test/A_wavlm-base-plus_cnn_TinyLlama_str2_mp50_Sum_%j.log
#SBATCH --output=logs/test/A_wavlm-base-plus_cnn_TinyLlama_str2_mp50_Sum_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 test.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --lr 0.0001 \
    --connector-k 2 \
    --meanpool 50 \
    --group 'Summarization' \
    --use-config summarize_switchboard.json \
    --epoch-to-test 7