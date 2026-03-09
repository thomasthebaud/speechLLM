import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import os


# print(scores)
# print(label)

def eer_from_scores(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    i = np.nanargmin(np.abs(fpr - fnr))
    eer = 0.5 * (fpr[i] + fnr[i])

    # Optional: try to interpolate around the crossing if it exists
    # (only if we have a neighbor on each side and the sign changes)
    if 0 < i < len(fpr) - 1:
        a, b = (fpr[i] - fnr[i]), (fpr[i+1] - fnr[i+1])
        if a == 0:
            eer = fpr[i]
        elif a * b < 0:  # crossing between i and i+1
            w = a / (a - b)  # in [0,1]
            eer = fpr[i] + w * (fpr[i+1] - fpr[i])
            thr_i = thr[i] + w * (thr[i+1] - thr[i])
            return eer, thr_i

    return eer, thr[i]


for trial_name in ['o', 'e', 'h']:
    filename = f"local/ASV_experiments/outputs_voxceleb1-{trial_name}.csv"
    if not os.path.exists(filename): 
        print(f"no outputs computed for trial {trial_name} yet.")
        continue
    df = pd.read_csv(filename, sep='\t') 
    # 600 samples: 22.66%


    scores = np.array([i[-4:-2].strip("'") for i in df['number']])
    keep = np.array([i for i,s in enumerate(scores) if len(s)>0])
    # print(set(scores))
    print(len(scores)-len(keep), len(scores))
    scores = scores[keep].astype(int)
    # print(set(scores))
    label = np.array([int(i=='target') for i in df['targettype']])
    label = label[keep]

    print(scores.shape, label.shape)

    eer, eer_thr = eer_from_scores(scores, label)
    eer = min(eer, 1-eer)
    print(f"EER Vox1-{trial_name}:", 100*eer, '%')