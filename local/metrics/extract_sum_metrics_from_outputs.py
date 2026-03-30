import os
import json
import argparse
from jiwer import wer
from metrics import MAE
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='A_wavlm-base-plus-cnn-TinyLlama-bs1_mp5_str2_lr0.0001')  
    args = parser.parse_args()
    model_name = args.model_name
    metrics = {}
    rouge_scorer_ = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rouge2'], use_stemmer=True)
    for dataset in os.listdir(f"exp/test_predictions/{model_name}"):
        if dataset[-4:]=='.csv':continue
        metrics[dataset]={}
        methods = os.listdir(f"exp/test_predictions/{model_name}/{dataset}/")
        for method in methods:
            for output_file in os.listdir(f"exp/test_predictions/{model_name}/{dataset}/{method}"):
                try:
                    model_epoch=str(int(output_file[:-4].split('epoch=')[1]))
                except:
                    model_epoch=0
                metrics[dataset][model_epoch]={}
                with open(f"exp/test_predictions/{model_name}/{dataset}/{method}/{output_file}", 'r') as f:
                    outputs = f.readlines()
                outputs = [l.strip('\n').replace("', '", "\", \"").replace("': '", "\": \"").replace("': \"", "\": \"").replace("\": '", "\": \"").replace("'}", "\"}").replace("{'", "{\"").replace("\", '", "\", \"").replace("', \"", "\", \"") for l in outputs]

                hyp = [l.split('INFO - [TARGET]')[1] for l in outputs if 'INFO - [TARGET]' in l]
                pred = [l.split('INFO - [PREDICTION]')[1] for l in outputs if 'INFO - [PREDICTION]' in l]
                assert len(hyp)==len(pred)
                
                scorer_list = [
                    rouge_scorer_.score(target_sum.lower(),predicted_sum.lower())
                    for target_sum, predicted_sum in zip(hyp, pred)
                    ]
                rouge_1 = [r_scores['rouge1'].fmeasure for r_scores in scorer_list]
                rouge_L = [r_scores['rougeL'].fmeasure for r_scores in scorer_list]
                rouge_2 = [r_scores['rouge2'].fmeasure for r_scores in scorer_list]
                print(f"{model_name}\t{dataset}\t{model_epoch}\t Rouge_1_f1_Summary: {100*np.mean(rouge_1)}")
                print(f"{model_name}\t{dataset}\t{model_epoch}\t Rouge_2_f1_Summary: {100*np.mean(rouge_2)}")
                print(f"{model_name}\t{dataset}\t{model_epoch}\t Rouge_L_f1_Summary: {100*np.mean(rouge_L)}")

                