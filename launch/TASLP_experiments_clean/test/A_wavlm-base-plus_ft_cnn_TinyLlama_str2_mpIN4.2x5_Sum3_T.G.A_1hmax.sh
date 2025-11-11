#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=te_S #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/TASLP_clean/test/A_wavlm-base-plus_ft_cnn_TinyLlama_str2_mpIN4.2x5_Sum3_T.G.A_1hmax_%j.log
#SBATCH --output=logs/TASLP_clean/test/A_wavlm-base-plus_ft_cnn_TinyLlama_str2_mpIN4.2x5_Sum3_T.G.A_1hmax_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 test.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str1.2.1_inmp4.2x5' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --ft-encoder \
    --encoder-lr 0.00001 \
    --lr 0.0001 \
    --truncate-sec 3600 \
    --meanpool 1 \
    --group 'TASLP_v2' \
    --use-config multitask_swb_AMI_ICSI_librispeech_voxceleb.json \
    --epoch-to-test 51 \
    --nickname '_3_T.G.A_1hmax'

