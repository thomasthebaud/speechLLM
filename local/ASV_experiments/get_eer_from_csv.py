import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve


# vox1_df = pd.read_csv('data/voxceleb1_test-o.csv') # EER = 0.8985%
# vox1_df = pd.read_csv('data/voxceleb1_test-e.csv') # EER = 0.4563%
# vox1_df = pd.read_csv('data/voxceleb1_test-h.csv') # EER = 0.9622%
# vox1_df = pd.read_csv('data/voxceleb2_test.csv')   # EER = 2.8269%
vox1_df = pd.read_csv('data/voxceleb2_dev_valid.csv') # EER = 2.8486%

if 'embedding_path_model' in vox1_df:
    model   = np.concatenate([np.load(i) for i in tqdm(vox1_df['embedding_path_model'], desc='loading models')])
    segment = np.concatenate([np.load(i) for i in tqdm(vox1_df['embedding_path_segment'], desc='loading segments')])
    label = np.array([int(i=='target') for i in vox1_df['targettype']])
else:
    # vox1_df = vox1_df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:1000]
    vox1_df = vox1_df.iloc[:1000]
    embeddings = list(vox1_df['embedding_path'])
    speakers = list(vox1_df['speaker'])
    model, segment, label = [], [], []
    for i, (e_i, s_i) in tqdm(enumerate(zip(embeddings[:-1], speakers[:-1])), total=len(embeddings), desc='building embedding couples for testing'):
        for e_j, s_j in zip(embeddings[i+1:], speakers[i+1:]):
            model.append(np.load(e_i))
            segment.append(np.load(e_j))
            label.append(int(s_j==s_i))
    model, segment, label = np.array(model)[:,0], np.array(segment)[:,0], np.array(label)

print(model.shape, segment.shape, label.shape)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def cosine_scores(model: np.ndarray, segment: np.ndarray) -> np.ndarray:
    m = l2_normalize(model)
    s = l2_normalize(segment)
    return np.sum(m * s, axis=1) 

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

scores = cosine_scores(model, segment)
print(scores.shape, label.shape)
eer, eer_thr = eer_from_scores(scores, label)
print("EER:", 100*eer, '%')