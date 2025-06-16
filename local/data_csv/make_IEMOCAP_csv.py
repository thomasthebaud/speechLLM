import os
import json
from tqdm import tqdm
import pandas as pd
from save_csv import save_csv

source = '/export/corpora6/IEMOCAP'

#Get labels
emo_dic = {'hap':'Happy', 'sad':'Sad', 'neu':'Neutral', 'ang':'Angry'}
for ses in range(1,6):
    labels = {'audio_path':[],'transcript':[], 'audio_len':[], 'emotion':[], 'gender':[]}
    for file in tqdm(os.listdir(f"{source}/Session{ses}/dialog/EmoEvaluation/"), desc=f"ses0{ses}"):
        if file[-3:]=='txt':
            with open(f"{source}/Session{ses}/dialog/EmoEvaluation/{file}", 'r') as f:
                lines = f.readlines()
            with open(f"{source}/Session{ses}/dialog/transcriptions/{file}", 'r') as f:
                transcripts = f.readlines()
            for idx, line in enumerate(lines):
                if line[0]=='[':
                    times, name, emo, _ = line.strip('\n').split('\t')
                    start, stop = times.strip('[]').split(' - ')
                    if emo in emo_dic.keys():
                        labels['audio_path'].append(f"{source}/Session{ses}/dialog/wav/{file[:-3]}wav")
                        labels['audio_len'].append(float(stop)-float(start))
                        
                        labels['emotion'].append(emo_dic[emo])
                        labels['gender'].append('F' if 'F' in file else 'M')

                        found=False
                        for l in transcripts:
                            if start[:-2] in l and stop[:-2] in l: #times are rounded sometimes
                                labels['transcript'].append(l.split(']: ')[1][:-1])
                                found=True
                                break
                        if not found:
                            print(f'transcript not found: {file} - {times}')
                            print(transcripts)
                            exit()


    labels = pd.DataFrame(labels)
    save_csv(labels, 'iemocap', f"ses0{ses}")