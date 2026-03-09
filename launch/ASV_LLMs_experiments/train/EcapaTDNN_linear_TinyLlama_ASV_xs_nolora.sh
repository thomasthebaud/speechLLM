#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=tr_ASV #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/ASV_LLM/train/EcapaTDNN_linear_TinyLlama_ASV_xs_nolora_%j.log
#SBATCH --output=logs/ASV_LLM/train/EcapaTDNN_linear_TinyLlama_ASV_xs_nolora_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export TOKENIZERS_PARALLELISM=false
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

#64 OOM

python3 train_ASV.py \
    --encoder 'precomputed/ecapa-tdnn' \
    --connector 'linear192' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 64 \
    --lr 0.001 \
    --group 'ASV' \
    --use-config ASV_voxceleb2_XS_ecapatdnn.json \
    --total-training-epoch 30 \
    --no-lora \
    --nickname "_ASV_xs_nolora"

