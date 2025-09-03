import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen
import numpy as np
from sklearn.model_selection import train_test_split

from save_csv import save_csv


root = '/export/fs05/tthebau1/EDART/SwitchBoard/'
transcript_files = os.listdir(root+'transcripts_clean/')
transcript_files = [i for i in transcript_files if len(i)>10]
summary_files = os.listdir(root+'summaries_GPT3.5/')
summary_files = [i for i in summary_files if len(i)>10]
labels = {'audio_path':[], 'audio_len':[], 'summary':[]}

assert len(transcript_files)==len(summary_files)
# get summaries
for file in summary_files:
    with open(root+'summaries_GPT3.5/'+file, 'r') as f:
        lines = f.readlines()
    summary = ' '.join([l.strip('\n') for l in lines])
    labels['summary'].append(summary)

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
  
df = pd.DataFrame(labels)
df = df[~df['audio_path'].isin(not_found)]
print(f"Processed {len(df)} conversations!")

train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(f"Train: {len(train)}, Dev: {len(val)}, Test: {len(test)}")
save_csv(train, 'switchboard', 'train')
save_csv(val, 'switchboard', 'val')
save_csv(test, 'switchboard', 'test')

            
        
    
