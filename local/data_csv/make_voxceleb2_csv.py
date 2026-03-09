import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
from sklearn.model_selection import train_test_split

emb_root = '/export/fs05/tthebau1/EDART/VoxCeleb2/ecapatdnn_speechbrain/all/'

print("Stage 1: Voxceleb2 dev and test sets")
root = '/export/corpora5/VoxCeleb2/'
metadata = pd.read_csv(root+'vox2_meta.csv', sep=' \t')
print(f"loaded metadata for {len(metadata)} files")
print(metadata.columns)
all_dfs = []
for set_ in ['test', 'dev']:
    labels = {'id':[], 'audio_path':[], 'audio_len':[], 'gender':[], 'speaker':[], 'video_id':[], 'embedding_path':[]}
    for spk in tqdm(os.listdir(root+f'{set_}/aac'), desc='voxceleb2_'+set_):
        meta = metadata[metadata['VoxCeleb2 ID']==spk].iloc[0]
        if meta['Set']==set_:
            for vid in os.listdir(root+f'{set_}/aac/{spk}/'):
                for file in os.listdir(root+f'{set_}/aac/{spk}/{vid}/'): 
                    audio_path=f"{root}/{set_}/aac/{spk}/{vid}/{file}"
                    labels['audio_path'].append(audio_path)
                    labels['gender'].append(meta['Gender'].upper())
                    audio = mutagen.File(audio_path)
                    labels['audio_len'].append(audio.info.length)
                    labels['speaker'].append(spk)
                    id = f"{spk}-{vid}-{file[:-4]}"
                    labels['id'].append(id)
                    labels['video_id'].append(vid)
                    labels['embedding_path'].append(emb_root+id+'.npy')

            else: continue


    df = pd.DataFrame(labels)
    print(f"Voxceleb2-{set_}:", df.shape)
    df.to_csv(f'data/voxceleb2_{set_}.csv', index=False)
    all_dfs.append(df.copy())

    if set_=='dev':
        df_train, df_dev = train_test_split(df, test_size=0.1)
        df_train.to_csv(f'data/voxceleb2_{set_}_train.csv', index=False)
        df_dev.to_csv(f'data/voxceleb2_{set_}_valid.csv', index=False)

    
        
    
