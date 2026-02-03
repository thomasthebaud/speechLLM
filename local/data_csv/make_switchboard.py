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
labels = {'audio_path':[], 'audio_len':[], 'transcript':[], 'noises':[], 'summary':[]}

assert len(transcript_files)==len(summary_files)
# Get transcripts / noises
for file in transcript_files:
    with open(root+'transcripts_clean/'+file, 'r') as f:
        lines = f.readlines()
    lines = [l[12:].strip('\n') for l in lines] #remove the '[Speaker A]:' at the begining and the '\n' at the end.
    transcript = ''
    noises = []
    for l in lines:
        if '<' in l:
            while '<' in l:
                spl = l.split('<')
                if len(spl)==2:
                    a,b = spl
                    b,c = b.split('>')
                    noises.append(b)
                    l = a+c
                else:
                    a,b,d = spl[0], spl[1], spl[2:]
                    b,c = b.split('>')
                    noises.append(b)
                    l = a+c+'<'.join(d)
        transcript = transcript+l
    
    labels['transcript'].append(transcript)
    labels['noises'].append(', '.join(list(set(noises))))
    
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

            
        
    
