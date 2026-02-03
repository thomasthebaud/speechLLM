import os
import json
from tqdm import tqdm
import pandas as pd
from save_csv import save_csv
import mutagen


source = '/export/corpora6/CREMA-D'
metadata = pd.read_csv(f"{source}/VideoDemographics.csv")

#Get labels
emo_dic = {'HAP':'Happy', 'SAD':'Sad', 'NEU':'Neutral', 'ANG':'Angry', 'DIS':'Disgust', 'FEA':'Fear'}
labels = {'audio_path':[], 'audio_len':[], 'emotion':[], 'gender':[], 'age':[], 'split':[], 'transcript':[]}
split_2_utt = {}
for split in ['train', 'dev', 'test']:
    utt_list=pd.read_csv(f"{source}/utt_list/cremad_utt2spk2emo.{split}")
    split_2_utt[split] = list(utt_list['utt'])

code_to_transcript = {
    "IEO":"It's eleven o'clock.",
    "TIE":"That is exactly what happened.",
    "IOM":"I'm on my way to the meeting.",
    "IWW":"I wonder what this is about.",
    "TAI":"The airplane is almost full.",
    "MTI":"Maybe tomorrow it will be cold.",
    "IWL":"I would like a new alarm clock.",
    "ITH":"I think I have a doctor's appointment.",
    "DFA":"Don't forget a jacket.",
    "ITS":"I think I've seen this before.",
    "TSI":"The surface is slick.",
    "WSI":"We'll stop in a couple of minutes.",
}
for file in tqdm(os.listdir(f"{source}/AudioWAV/")):
    if file[-3:]=='wav':
        spk, sentence, emo, volume = file[:-4].split('_')
        labels['audio_path'].append(f"{source}/AudioWAV/{file}")
        audio = mutagen.File(f"{source}/AudioWAV/{file}")
        labels['audio_len'].append(audio.info.length)
        
        labels['emotion'].append(emo_dic[emo])
        meta_spk = metadata[metadata['ActorID']==int(spk)]
        if len(meta_spk)!=1:
            print(spk, meta_spk)
            exit()
        gender = meta_spk['Sex'].item()[0]
        labels['gender'].append(gender)
        age = meta_spk['Age'].item()
        labels['age'].append(age)
        labels['transcript'].append(code_to_transcript[sentence])
        tot_split=0
        for split in split_2_utt:
            if file[:-4] in split_2_utt[split]:
                labels['split'].append(split)
                tot_split+=1

        if tot_split==0: labels['split'].append('None')

        if tot_split>1:
            print(tot_split, file[:-4])
            for split in split_2_utt:
                if file[:-4] in split_2_utt[split]: print(split_2_utt[split])
            exit()

labels = pd.DataFrame(labels)
print(labels.shape)
for split in ['train', 'dev', 'test']:
    split_labels = labels[labels['split']==split]
    split_labels.pop('split')
    print(split, split_labels.shape)
    save_csv(split_labels, 'crema-d', split)

print(f"{len(labels[labels['split']=='None'])} samples ignored")