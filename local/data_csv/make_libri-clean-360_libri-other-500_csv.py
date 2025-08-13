import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen

from save_csv import save_csv

root = '/export/corpora5/LibriSpeech/'

#Get speakers gender / name
with open(root+'SPEAKERS.TXT', 'r') as f:
    spks = f.readlines()
spks = [line.strip('\n').split('|') for line in spks if line[0]!=';']
spks = {int(s[0].strip(' ')): [i.strip(' ') for i in s[1:]] for s in spks}

for subset in ['train-clean-360', 'train-other-500', 'test-other', 'dev-other']:
    source = root+subset

    #Get labels
    labels = {'audio_path':[], 'audio_len':[], 'transcript':[], 'gender':[]}
    for spk in tqdm(os.listdir(source), desc=subset):
        g = spks[int(spk)][0]
        for ses in os.listdir(f"{source}/{spk}"):
            with open(f"{source}/{spk}/{ses}/{spk}-{ses}.trans.txt", 'r') as f:
                lines = f.readlines()
            for line in lines:
                file = line.split(' ')[0]
                audio_path=f"{source}/{spk}/{ses}/{file}.flac"
                labels['audio_path'].append(audio_path)
                labels['transcript'].append(line[len(file)+1:-1])
                labels['gender'].append(g)
                audio = mutagen.File(audio_path)
                labels['audio_len'].append(audio.info.length)


    labels = pd.DataFrame(labels)
    print(labels.shape)

    save_csv(labels, 'librispeech', subset)