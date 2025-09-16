#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=test_sllm #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/test/T_base_tinyLLama_%j.log
#SBATCH --output=logs/test/T_base_tinyLLama_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 local/metrics/test_base_tinyLLAMA.py \