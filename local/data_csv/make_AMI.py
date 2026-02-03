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

data_root = '/export/corpora5/amicorpus/'
metadata_root = '/export/fs05/tthebau1/EDART/ami-and-icsi-corpora/ami-corpus/output/abstractive/'

labels = {'audio_path':[], 'audio_len':[], 'abstract_summary':[], 'decisions_summary':[], 'problems_summary':[], 'actions_summary':[], 'dialogue_id':[]}

summaries = []
dialogues = []
types = ['problems', 'actions', 'decisions', 'abstract']
for file in os.listdir(metadata_root):
    with open(f"{metadata_root}{file}", 'r') as json_file:
        data_dict = json.load(json_file)
    
    dialogue_id = file[:-5]
    labels['dialogue_id'].append(dialogue_id)
    labels['audio_path'].append(f"{data_root}/{dialogue_id}/audio/{dialogue_id}.Mix-Headset.wav")

    for t in types:
        labels[f'{t}_summary'].append(' '.join([i['text'] for i in data_dict if i['type']==t]))
    
for audio_path in tqdm(labels['audio_path'], desc='audio files processing'):
    audio = mutagen.File(audio_path)
    labels['audio_len'].append(audio.info.length)

df = pd.DataFrame(labels)
df['summary'] = df['abstract_summary']
df.pop('abstract_summary')
print(f"Dataset loaded: {df.shape}, max_len = {np.max(df['audio_len'])}")


## from 'https://github.com/guokan-shang/ami-and-icsi-corpora/tree/master?tab=readme-ov-file' ##
def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]
 
ami_train = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
ami_train = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in ami_train])
ami_train.remove('IS1002a')
ami_train.remove('IS1005d')

ami_validation = ['ES2003', 'ES2011', 'IS1008', 'TS3004', 'TS3006']
ami_validation = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in ami_validation])

ami_test = ['ES2004', 'ES2014', 'IS1009', 'TS3003', 'TS3007']
ami_test = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in ami_test])
#####


train = df[df['dialogue_id'].isin(ami_train)]
val = df[df['dialogue_id'].isin(ami_validation)]
test = df[df['dialogue_id'].isin(ami_test)]
train.pop('dialogue_id')
val.pop('dialogue_id')
test.pop('dialogue_id')

print(f"Train: {len(train)}, Dev: {len(val)}, Test: {len(test)}")
save_csv(train, 'AMI', 'train')
save_csv(val, 'AMI', 'val')
save_csv(test, 'AMI', 'test')

            
        
    
