import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
from sklearn.model_selection import train_test_split

do_stage_1 = True
emb_root = '/export/fs05/tthebau1/EDART/VoxCeleb1/ecapatdnn_speechbrain/all/'

if do_stage_1:
    print("Stage 1: Voxceleb1 dev and test sets")
    root = '/export/corpora5/VoxCeleb1_v2/'
    metadata = pd.read_csv(root+'vox1_meta.csv', sep='\t')
    print(f"loaded metadata for {len(metadata)} files")

    all_dfs = []
    for set_ in ['test', 'dev']:
        labels = {'id':[], 'audio_path':[], 'audio_len':[], 'gender':[], 'accent':[], 'speaker':[], 'video_id':[], 'embedding_path':[]}
        for spk in tqdm(os.listdir(root+'wav'), desc='voxceleb1_'+set_):
            meta = metadata[metadata['VoxCeleb1 ID']==spk].iloc[0]
            if meta['Set']==set_:
                for vid in os.listdir(root+f'wav/{spk}/'):
                    for file in os.listdir(root+f'wav/{spk}/{vid}/'): 
                        audio_path=f"{root}/wav/{spk}/{vid}/{file}"
                        labels['audio_path'].append(audio_path)
                        labels['gender'].append(meta['Gender'].upper())
                        labels['accent'].append(meta['Nationality'])
                        audio = mutagen.File(audio_path)
                        labels['audio_len'].append(audio.info.length)
                        labels['speaker'].append(spk)
                        id = f"{spk}-{vid}-{file[:-4]}"
                        labels['id'].append(id)
                        labels['video_id'].append(vid)
                        labels['embedding_path'].append(emb_root+id+'.npy')

                else: continue


        df = pd.DataFrame(labels)
        print(f"Voxceleb1-{set_}:", df.shape)
        df.to_csv(f'data/voxceleb1_{set_}.csv', index=False)
        all_dfs.append(df.copy())

        if set_=='dev':
            df_train, df_dev = train_test_split(df, test_size=0.1)
            df_train.to_csv(f'data/voxceleb1_{set_}_train.csv', index=False)
            df_dev.to_csv(f'data/voxceleb1_{set_}_valid.csv', index=False)
else:
    all_dfs = [pd.read_csv(f'data/voxceleb1_{set_}.csv') for set_ in ['test', 'dev']]

dfs = pd.concat(all_dfs)
print(f"Full Voxceleb1: {dfs.shape}")

print("Stage 2: Voxceleb1 trials files")

for trial_name in ['o', 'e', 'h']:
    trials = pd.read_csv(f'data/trials_{trial_name}.csv')
    trials['embedding_path_model'] = [emb_root+id+'.npy' for id in trials['modelid']]
    trials['embedding_path_segment'] = [emb_root+id+'.npy' for id in trials['segmentid']]
    trials.to_csv(f'data/voxceleb1_test-{trial_name}.csv', index=False)
    print(f"VoxCeleb1-test-{trial_name} saved: {trials.shape}")
    
        
    
