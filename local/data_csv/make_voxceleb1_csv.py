import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen

from save_csv import save_csv

root = '/export/corpora5/VoxCeleb1_v2/'
metadata = pd.read_csv(root+'vox1_meta.csv', sep='\t')
print(f"loaded metadata for {len(metadata)} files")

for set in ['dev', 'test']:
    labels = {'audio_path':[], 'audio_len':[], 'gender':[], 'accent':[]}
    for spk in tqdm(os.listdir(root+'wav'), desc='voxceleb1_'+set):
        meta = metadata[metadata['VoxCeleb1 ID']==spk].iloc[0]
        if meta['Set']==set:
            for vid in os.listdir(root+f'wav/{spk}/'):
                for file in os.listdir(root+f'wav/{spk}/{vid}/'): 
                    audio_path=f"{root}/wav/{spk}/{vid}/{file}"
                    labels['audio_path'].append(audio_path)
                    labels['gender'].append(meta['Gender'].upper())
                    labels['accent'].append(meta['Nationality'])
                    audio = mutagen.File(audio_path)
                    labels['audio_len'].append(audio.info.length)


            else: continue

    df = pd.DataFrame(labels)
    print(df.shape)
    save_csv(df, 'voxceleb1', set)