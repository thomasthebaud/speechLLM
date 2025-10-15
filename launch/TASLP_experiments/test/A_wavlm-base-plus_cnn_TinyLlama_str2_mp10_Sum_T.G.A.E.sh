#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24000
#SBATCH --job-name=te_STGAE #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --account=a100acct
#SBATCH --error=logs/TASLP/test/A_wavlm-base-plus_cnn_TinyLlama_str2_mp10_Sum_T.G.A.E_%j.log
#SBATCH --output=logs/TASLP/test/A_wavlm-base-plus_cnn_TinyLlama_str2_mp10_Sum_T.G.A.E_%j.log

export HF_HOME=./hf_cache/
export HF_DATASETS_CACHE=./hf_cache/

echo `date`

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python3 test.py \
    --encoder 'microsoft/wavlm-base-plus' \
    --connector 'cnn_str1.2.1' \
    --llm 'TinyLlama-1.1B-Chat-v1.0' \
    --batch-size 1 \
    --lr 0.0001 \
    --meanpool 10 \
    --group 'TALSP' \
    --use-config summarize+multitask_switchboard_librispeech_voxceleb_iemocap.json \
    --epoch-to-test 141 \
    --nickname "_T.G.A.E"

