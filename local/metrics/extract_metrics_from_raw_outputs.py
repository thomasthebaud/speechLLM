import os
import json
from jiwer import wer
from metrics import MAE
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

if __name__=="__main__":
    for mp in ['2', '5', '20', '50']:
        model_name = f'A_wavlm-base-plus-cnn-TinyLlama-bs1_mp{mp}_str2_lr0.0001'
        metrics = {}
        rouge_scorer_ = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        print(f"MP = {mp}")
        for dataset in os.listdir(f"exp/test_predictions/{model_name}"):
            if dataset[-4:]=='.csv' or 'switch' not in dataset:continue
            metrics[dataset]={}
            for AT in os.listdir(f"exp/test_predictions/{model_name}/{dataset}/"):
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
                        else:
                            print(h)
                    
                    scorer_list = [
                        rouge_scorer_.score(target_sum.lower(),predicted_sum.lower())
                        for target_sum, predicted_sum in zip(outputs['Summary']['hyp'], outputs['Summary']['pred'])
                        ]
                    rouge_1 = [r_scores['rouge1'].precision for r_scores in scorer_list]
                    rouge_L = [r_scores['rougeL'].precision for r_scores in scorer_list]
                    rouge_2 = [r_scores['rouge2'].precision for r_scores in scorer_list]
                    metrics[dataset][model_epoch]['Rouge_1_Summary'] = np.mean(rouge_1)
                    metrics[dataset][model_epoch]['Rouge_L_Summary'] = np.mean(rouge_L)
                    metrics[dataset][model_epoch]['Rouge_2_Summary'] = np.mean(rouge_2)

                # print(dataset)
                # print(metrics[dataset])
                # print()

        all_fields = []
        all_epochs = []
        for dataset in metrics:
            for epoch in metrics[dataset]:
                all_epochs.append(epoch)
                for metric in metrics[dataset][epoch]:
                    all_fields.append(metric)
        all_fields=sorted(list(set(all_fields)))
        all_epochs=sorted(list(set(all_epochs)))
        for model_epoch in all_epochs:
            outputs = {field:['']*len(metrics) for field in all_fields}
            outputs['dataset'] = list(metrics.keys())
            for i,dataset in enumerate(list(metrics.keys())):
                for metric in metrics[dataset][model_epoch]:
                    outputs[metric][i] = metrics[dataset][model_epoch][metric]
            outputs_df = pd.DataFrame(outputs)
            outputs_df = outputs_df[['dataset']+all_fields]
            print(model_epoch)
            print(outputs_df)
            outputs_df.to_csv(f"exp/test_predictions/{model_name}/metrics_{model_epoch}.csv", index=False)

            
            