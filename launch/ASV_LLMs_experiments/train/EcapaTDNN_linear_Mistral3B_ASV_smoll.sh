#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=tr_ASV #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu   #queue
#SBATCH --exclude=c05
#SBATCH --error=logs/ASV_LLM/train/EcapaTDNN_linear_Mistral3B_ASV_1080_%j.log
#SBATCH --output=logs/ASV_LLM/train/EcapaTDNN_linear_Mistral3B_ASV_1080_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export TOKENIZERS_PARALLELISM=false
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

#64 OOM

python3 train_ASV.py \
    --encoder 'precomputed/ecapa-tdnn' \
    --connector 'linear192_2512' \
    --llm 'Ministral-3-3B-Base-2512' \
    --batch-size 16 \
    --lr 0.0001 \
    --group 'ASV' \
    --use-config ASV_voxceleb2_ecapatdnn.json \
    --total-training-epoch 100 \
    --nickname "_ASV_1080"

