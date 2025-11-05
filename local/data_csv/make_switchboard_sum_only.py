import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
import numpy as np
from sklearn.model_selection import train_test_split

from save_csv import save_csv

model_list = ['GPT3.5', 'GPT4.o', 'GPT5-mini']
root = '/export/fs05/tthebau1/EDART/SwitchBoard/'
transcript_files = os.listdir(root+'transcripts_clean/')
transcript_files = [i for i in transcript_files if len(i)>10]

labels = {'audio_path':[], 'audio_len':[], 'summary':[]}
# get audios
audio_files = ['sw0'+i[8:-4]+'.wav' for i in transcript_files]

not_found = []
for file in tqdm(audio_files, desc='audio files processing'):
    audio_path=f"{root}wav/{file}"
    labels['audio_path'].append(audio_path)
    try: 
        audio = mutagen.File(audio_path)
        labels['audio_len'].append(audio.info.length)
    except:
        not_found.append(audio_path)
        labels['audio_len'].append(0)
        print(f"file {audio_path} not found")
        
print(f"Longest segment is: {np.max(labels['audio_len'])} seconds")

all_dfs = []
for model in model_list:
    labels['summary'] = []
    summary_files = os.listdir(root+f'summaries_{model}/')
    summary_files = [i for i in summary_files if len(i)>10]
    assert len(transcript_files)==len(summary_files)
    # get summaries
    for file in summary_files:
        with open(root+f'summaries_{model}/'+file, 'r') as f:
            lines = f.readlines()
        summary = ' '.join([l.strip('\n') for l in lines])
        labels['summary'].append(summary)

    df = pd.DataFrame(labels)
    print(model, df.tail())
    df['model']=model
    df = df[~df['audio_path'].isin(not_found)]
    print(f"Processed {len(df)} conversations, by model {model}")
    all_dfs.append(df)

df = pd.concat(all_dfs)
print(f"Total = {len(df)} conversations, by {len(model_list)} models!")
refs = list(set(df['audio_path']))
train_ref, test_ref = train_test_split(refs, test_size=0.2)
train_ref, val_ref = train_test_split(train_ref, test_size=0.2)
train = df[df['audio_path'].isin(train_ref)]
val   = df[df['audio_path'].isin(val_ref)]
test  = df[df['audio_path'].isin(test_ref)]

print(test.tail())

print(f"Total:      \t Train: {len(train)}, Dev: {len(val)}, Test: {len(test)}")
for model in model_list:
    train_ = train[train['model']==model]
    # train_.pop('model')
    val_ = val[val['model']==model]
    # val_.pop('model')
    test_ = test[test['model']==model]
    # test_.pop('model')
    print(f"model {model} \t Train: {len(train_)}, Dev: {len(val_)}, Test: {len(test_)}")
    save_csv(train_, f'switchboard_{model}', 'train')
    save_csv(val_, f'switchboard_{model}', 'val')
    save_csv(test_, f'switchboard_{model}', 'test')

            
        
    
