import os
import json
from tqdm import tqdm
import pandas as pd

import mutagen

from save_csv import save_csv

root = '/export/corpora5/CommonVoice/en_1488h_2019-12-10/'
for split in ['train', 'test', 'dev']:
    metadata = pd.read_csv(root+split+'.tsv', sep='\t')
    print(f"loaded {split} metadata for {len(metadata)} files")
    metadata = metadata[metadata['accent'].notnull()]
    metadata = metadata[metadata['accent']!='other']
    metadata = metadata[metadata['age'].notnull()]
    metadata = metadata[metadata['age']!='other']
    metadata = metadata[metadata['gender'].notnull()]
    metadata = metadata[metadata['gender']!='other']

    print(split, len(set(metadata['accent'])), set(metadata['accent']), len(metadata), set(metadata['age']))
    age_dic = {'seventies':70, 'fourties':40, 'thirties':30, 'eighties':80, 'teens':10, 'twenties':20, 'sixties':60, 'fifties':50, 'nineties':90}
    labels = {'audio_path':[], 'audio_len':[], 'gender':[], 'accent':[], 'age':[], 'transcript':[]}
    print(split, len(metadata))
    for idx,row in tqdm(metadata.iterrows(), total=len(metadata)):
        file, age, gender, accent = row['path'], row['age'], row['gender'], row['accent']
        audio_path=f"{root}/clips/{file}"
        labels['audio_path'].append(audio_path)
        labels['gender'].append(gender.upper()[0])
        labels['accent'].append(accent)
        labels['age'].append(age_dic[age])
        audio = mutagen.File(audio_path)
        labels['audio_len'].append(audio.info.length)
        labels['transcript'].append(row['sentence'])

    
    df = pd.DataFrame(labels)
    print(df.shape)
    save_csv(df, 'CV-EN', split)
