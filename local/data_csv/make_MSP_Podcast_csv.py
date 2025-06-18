import os
import json
from tqdm import tqdm
import pandas as pd

import mutagen

from save_csv import save_csv

root = '/export/corpora5/MSP_Podcast_EmotionDataset/'
partitions = pd.read_csv(root+'Partitions.txt', sep='; ', names=['set', 'audio'])
print(set(partitions['set']))
partitions.sort_values('audio', inplace=True)

with open(f'{root}labels.txt', 'r') as f:
    labels = f.readlines()
    labels = [i.split(';')[:2] for i in labels if '.wav; ' in i]

code_to_emo = {'A': 'Angry',
                'S':'Sad',
                'H':'Happy',
                'U':'Suprise',
                'F':'Fear',
                'D':'Disgust',
                'C':'Contempt',
                'N':'Neutral',
                'O':'Other',
                'X':'X'
                }

metadata = {'audio_path':[i[0] for i in labels], 
        'emotion':[code_to_emo[i[1].strip(' ')] for i in labels],
        }

assert metadata['audio_path']==list(partitions['audio'])
metadata['set'] = list(partitions['set'])

metadata = pd.DataFrame(metadata)


with open(f'{root}Speaker_ids.txt', 'r') as f:
    labels = f.readlines()
    spk_to_gender = [i.split('; ') for i in labels if 'Speaker_' in i]
    audio_to_spk = [i.split('; ') for i in labels if '.wav; ' in i]

spk_idx_to_gender = {line[0].split('_')[1]:line[1][0] for line in spk_to_gender}
spk_idx_to_gender['Unknown'] = 'None'
audio_to_spk = {line[0]:line[1].strip('\n') for line in audio_to_spk}
# print(len(audio_to_spk), len(metadata))
# print(spk_idx_to_gender)
# print(audio_to_spk)

metadata['gender'] = [spk_idx_to_gender[audio_to_spk[i]] if i in audio_to_spk.keys() else 'None' for i in metadata['audio_path']]
metadata = metadata[metadata['gender']!='J']
metadata = metadata[metadata['gender']!='None']
print(set(metadata['gender']))

audio_lens = []
for audio_path in tqdm(metadata['audio_path']):
    audio = mutagen.File(root+'Audio/'+audio_path)
    audio_lens.append(audio.info.length)

metadata['audio_len'] = audio_lens
print(metadata.head(), metadata.shape)

for split in set(metadata['set']):
    subset = metadata[metadata['set']==split]
    print(split, subset.shape)
    save_csv(subset, 'MSP_Podcast', split)
