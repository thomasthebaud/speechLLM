import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_stats(df):
    N_spks = len(set(df['speaker']))
    N_vid = len(set(df['video_id']))
    N_seg_p_spk = len(df)/len(set(df['speaker']))
    Dur = np.sum(df['audio_len'])
    print(f"N speakers: {N_spks}, N videos: {N_vid},\tSegments per video: {int(N_seg_p_spk)},\tDuration (h): {Dur/3600}, total length: {len(df)}")

df = pd.read_csv('data/voxceleb2_dev.csv')
get_stats(df)

# dfs = []
# for spk in tqdm(list(set(df['speaker'])), desc='taking 10 segments per speaker'):
#     dfs.append(df[df['speaker']==spk].sample(n=10, random_state=42))
# df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
# get_stats(df)
# df.to_csv(f'data/voxceleb2_dev_S.csv', index=False)


print('Keeping 10% of speakers')
spks = list(set(df[df['gender']=='M']['speaker']))[:300] + list(set(df[df['gender']=='F']['speaker']))[:300]
df = df[df['speaker'].isin(spks)].reset_index(drop=True)
get_stats(df)
# df.to_csv(f'data/voxceleb2_dev_XS.csv', index=False)

dfs = []
for spk in tqdm(list(set(df['speaker'])), desc='taking 2 segments per speaker'):
    dfs.append(df[df['speaker']==spk].sample(n=2, random_state=42))
df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
get_stats(df)
df.to_csv(f'data/voxceleb2_dev_XXS.csv', index=False)