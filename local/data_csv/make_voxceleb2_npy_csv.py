import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
from sklearn.model_selection import train_test_split

from save_csv import save_csv

root = '/export/fs05/tthebau1/EDART/VoxCeleb2/ecapatdnn_speechbrain'
for set_ in ['test', 'dev']:
    print(set_)
    labels = {'embedding_path':[], 'speaker':[], 'video_id':[]}
    file_list = os.listdir(f'{root}/{set_}/')
    for file in tqdm(file_list, desc=set_):
        if '.npy' in file:
            video_id = file[8:-10]
            speaker_id = file[:7]
            labels['embedding_path'].append(f'{root}/{set_}/{file}')
            labels['speaker'].append(speaker_id)
            labels['video_id'].append(video_id)

    df = pd.DataFrame(labels)
    print(df.shape)
    save_csv(df, 'voxceleb2_npy', set_)

    if set_=='dev':
        df_train, df_dev = train_test_split(df, test_size=0.1)
        save_csv(df_train, 'voxceleb2_npy', set_+'_train')
        save_csv(df_dev, 'voxceleb2_npy', set_+'_valid')

            
        
    
