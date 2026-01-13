#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=ft_3STGAEAc #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/TASLP_clean/train/A_wavlm-base-plus_ft_cnn_TinyLlama_str2_mpIN2x2_T.G.A.E.Ac_%j.log
#SBATCH --output=logs/TASLP_clean/train/A_wavlm-base-plus_ft_cnn_TinyLlama_str2_mpIN2x2_T.G.A.E.Ac_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 train.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str1.2.1_inmp2x2' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --ft-encoder \
    --encoder-lr 0.00001 \
    --lr 0.0001 \
    --meanpool 1 \
    --group 'TASLP_v2' \
    --use-config multitask_librispeech_voxceleb_iemocap_commonvoice.json \
    --total-training-epoch 100 \
    --nickname '_T.G.A.E.Ac_nosum' \
    --truncate-sec 59 

