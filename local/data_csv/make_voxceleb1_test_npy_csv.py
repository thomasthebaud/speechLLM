import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
from sklearn.model_selection import train_test_split

from save_csv import save_csv

root = '/export/fs05/tthebau1/EDART/VoxCeleb1/ecapatdnn_speechbrain/test'
file_list = os.listdir(root)

names = pd.read_csv("/export/corpora5/VoxCeleb1_v2/vox1_meta.csv", sep='\t')
id_to_names = {row['VoxCeleb1 ID']:row['VGGFace1 ID'] for _,row in names.iterrows()}

def model_id_to_named(model_id, id_to_names):
    spk, vid, num = model_id[:7], model_id[8:-6], model_id[-5:]
    return f"{id_to_names[spk]}-{vid}-00{num}.npy"


for trial_name in ['o', 'e', 'h']:
    labels = {'embedding_path_model':[], 'embedding_path_segment':[], 'targettype':[], 'model_id':[], 'segment_id':[]}
    trials = pd.read_csv(f'data/trials_{trial_name}.csv')
    missing = 0

    for idx, row in tqdm(trials.iterrows(), total=len(trials), desc=f'verifying all files present, split {trial_name}'):
        model_id, segment_id = row['modelid'], row['segmentid']
        if model_id.replace('-', '_')+'.npy' in file_list or segment_id.replace('-', '_')+'.npy' in file_list: 
            print(f'EXCEPTION:', model_id.replace('-', '_')+'.npy', segment_id.replace('-', '_')+'.npy')

        named_model_id = model_id_to_named(model_id, id_to_names)
        named_segment_id = model_id_to_named(segment_id, id_to_names)
        if named_model_id in file_list and named_segment_id in file_list:
            assert named_model_id in file_list, print(f'missing {named_model_id}')
            assert named_segment_id in file_list, print(f'missing {named_segment_id}')

            labels['embedding_path_model'].append(f"{root}/{named_model_id}")
            labels['embedding_path_segment'].append(f"{root}/{named_segment_id}")
            labels['targettype'].append(row['targettype'])
            labels['model_id'].append(model_id)
            labels['segment_id'].append(segment_id)
        else: missing+=1
        

    df = pd.DataFrame(labels)
    print(df.shape)
    print(f"Trial {trial_name}, missing: {missing} embeddings")
    save_csv(df, 'voxceleb1_npy', f'test-{trial_name}')
            
        
    
