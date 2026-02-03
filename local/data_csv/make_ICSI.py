import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from save_csv import save_csv
import xml.etree.ElementTree as ET

data_root = '/export/corpora5/ICSI/Signals'
metadata_root = '/export/fs05/tthebau1/EDART/ami-and-icsi-corpora/icsi-corpus/output/abstractive/'

labels = {'audio_path':[], 'audio_len':[], 'abstract_summary':[], 'decisions_summary':[], 'problems_summary':[], 'progress_summary':[], 'dialogue_id':[]}

summaries = []
dialogues = []
types = ['problems', 'progress', 'decisions', 'abstract']
set_types = []
for file in os.listdir(metadata_root):
    with open(f"{metadata_root}{file}", 'r') as json_file:
        data_dict = json.load(json_file)
    
    dialogue_id = file[:-5]
    labels['dialogue_id'].append(dialogue_id)
    labels['audio_path'].append(f"{data_root}/{dialogue_id}/{dialogue_id}.interaction.wav")
    set_types = set_types+ list(set([i['type'] for i in data_dict]))
    for t in types:
        labels[f'{t}_summary'].append(' '.join([i['text'] for i in data_dict if i['type']==t]))

print(f"types of summaries found: {set(set_types)}")

for audio_path in tqdm(labels['audio_path'], desc='audio files processing'):
    audio = mutagen.File(audio_path)
    labels['audio_len'].append(audio.info.length)

df = pd.DataFrame(labels)
df['summary'] = df['abstract_summary']
df.pop('abstract_summary')
print(f"Dataset loaded: {df.shape}, max_len = {np.max(df['audio_len'])}")
# print(df.head())

## from 'https://github.com/guokan-shang/ami-and-icsi-corpora/tree/master?tab=readme-ov-file' ## 
icsi_test = ['Bed004', 'Bed009', 'Bed016', 'Bmr005', 'Bmr019', 'Bro018']
#####

train_val = df[~df['dialogue_id'].isin(icsi_test)]
test = df[df['dialogue_id'].isin(icsi_test)]
print(f"Train/val: {len(train_val)}, test: {len(test)}, total: {len(df)}")
train_val.pop('dialogue_id')
test.pop('dialogue_id')

train, val = train_test_split(train_val, test_size=0.1)


print(f"Train: {len(train)}, Dev: {len(val)}, Test: {len(test)}")
save_csv(train, 'ICSI', 'train')
save_csv(val, 'ICSI', 'val')
save_csv(test, 'ICSI', 'test')

            
        
    
