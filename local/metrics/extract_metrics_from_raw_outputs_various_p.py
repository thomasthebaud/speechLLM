import os
import json
from jiwer import wer
from metrics import MAE
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

if __name__=="__main__":
    for p in ['0.01', '0.1', '0.5', '1']:
        model_name = f'AT_wavlm-base-plus-cnn-TinyLlama-bs1_p{p}_mp10_str2_lr0.0001'
        metrics = {}
        rouge_scorer_ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print(f"########  prop text = {p}  #######")
        if not os.path.exists(f"exp/test_predictions/{model_name}"):model_name=model_name+f"_p{p}"
        for dataset in os.listdir(f"exp/test_predictions/{model_name}"):
            if dataset[-4:]=='.csv' or 'switch' not in dataset:continue
            metrics[dataset]={}
            for AT in os.listdir(f"exp/test_predictions/{model_name}/{dataset}/"):
                print(f"Audio or Text: {AT}")
                for output_file in os.listdir(f"exp/test_predictions/{model_name}/{dataset}/{AT}/"):
                    model_epoch=output_file[len(model_name):-4]
                    metrics[dataset][model_epoch]={}
                    with open(f"exp/test_predictions/{model_name}/{dataset}/{AT}/{output_file}", 'r') as f:
                        outputs = f.readlines()
                    outputs = [l.strip('\n').replace("', '", "\", \"").replace("': '", "\": \"").replace("': \"", "\": \"").replace("\": '", "\": \"").replace("'}", "\"}").replace("{'", "{\"").replace("\", '", "\", \"").replace("', \"", "\", \"") for l in outputs]

                    hyp_ = [l.split('INFO - [TARGET]')[1] for l in outputs if 'INFO - [TARGET]' in l]
                    pred = [l.split('INFO - [RAW OUTPUT]')[1] for l in outputs if 'INFO - [RAW OUTPUT]' in l]
                    assert len(hyp_)==len(pred)
                    hyp = []
                    for original_hyp in hyp_:
                        try:
                            hyp.append(json.loads(fr"{original_hyp}"))
                        except:
                            print(f"Failed to process:\nhyp:{original_hyp}")
                            continue
                    
                    keys = hyp[0].keys()
                    # print(keys)
                    outputs = {key:{'pred':[], 'hyp':[], 'miss':0} for key in keys}
                    for h, p in zip(hyp, pred):
                        if "Summary" in h:
                            outputs["Summary"]['hyp'].append(h["Summary"])
                            outputs["Summary"]['pred'].append(p)
                        # else:
                        #     print(h)
                    
                    scorer_list = [
                        rouge_scorer_.score(target_sum.lower(),predicted_sum.lower())
                        for target_sum, predicted_sum in zip(outputs['Summary']['hyp'], outputs['Summary']['pred'])
                        ]
                    rouge_1 = [r_scores['rouge1'].precision for r_scores in scorer_list]
                    rouge_L = [r_scores['rougeL'].precision for r_scores in scorer_list]
                    rouge_2 = [r_scores['rouge2'].precision for r_scores in scorer_list]
                    print(f"Epoch {model_epoch}- Rouge1 {100*np.mean(rouge_1):.2f}\tRouge2 {100*np.mean(rouge_2):.2f}\tRougeL {100*np.mean(rouge_L):.2f}")

            
            