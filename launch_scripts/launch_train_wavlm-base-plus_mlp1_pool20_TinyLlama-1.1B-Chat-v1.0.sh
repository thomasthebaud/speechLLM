#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16000
#SBATCH --job-name=train_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/train_wavlm-base-plus_mlp1_pool20_TinyLlama_%j.log
#SBATCH --output=logs/train_wavlm-base-plus_mlp1_pool20_TinyLlama_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`
ME=`basename "$0"`
echo "My slurm job id is $ME"

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'mlp' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --connector-k '20' \
    --connector-layers '1' \
    --batch-size 64 \
    --lr 0.00001
