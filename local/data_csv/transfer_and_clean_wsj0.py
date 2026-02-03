import os
from tqdm import tqdm
import pandas as pd
import mutagen

from save_csv import save_csv

source= '/export/corpora5/LDC/LDC93S6A/'
target = '/export/fs05/tthebau1/WSJ0/'
stage=4
# Stage 1: copy wv2, wv1 and metadata files
disks = os.listdir(source)
if stage==1:
    print(f"{len(disks)} disks to be processed.")
    for dir in disks:
        if dir[:2]=='11':
            splits = os.listdir(f"{source}{dir}/wsj0/")
            for split, subset in [('si_tr_s', 'train'), ('si_dt_05','dev'), ('si_et_05', 'eval')]:
                if split in splits:
                    for num_dir in tqdm(os.listdir(f"{source}{dir}/wsj0/{split}"), desc=dir):
                        if not os.path.exists(f"{target}{subset}/{num_dir}"): os.makedirs(f"{target}{subset}/{num_dir}")
                        os.system(f"cp {source}{dir}/wsj0/{split}/{num_dir}/* {target}{subset}/{num_dir}/")
            
    print('stage 1 done')
    exit()

if stage==2:
    # stage 2: get the metadata
    metadata = {'split':[], 'transcript':[], 'ref':[], 'dir':[]}
    for dir in tqdm(disks):
        if dir[:2]=='11':
            splits = os.listdir(f"{source}{dir}/wsj0/")
            if 'transcrp' in splits:
                splits = os.listdir(f"{source}{dir}/wsj0/transcrp/dots")
                for split, subset in [('si_tr_s', 'train'), ('si_dt_05','dev'), ('si_et_05', 'eval')]:
                    if split in splits:
                        for num_dir in os.listdir(f"{source}{dir}/wsj0/transcrp/dots/{split}"):
                            for dot_file in os.listdir(f"{source}{dir}/wsj0/transcrp/dots/{split}/{num_dir}"):
                                with open(f"{source}{dir}/wsj0/transcrp/dots/{split}/{num_dir}/{dot_file}", 'r') as f:
                                    lines = f.readlines()
                                for line in lines:
                                    metadata['split'].append(subset)
                                    metadata['transcript'].append(line[:-12])
                                    metadata['ref'].append(line[-10:-2])
                                    metadata['dir'].append(num_dir)
            subset='eval'
            for num_dir in os.listdir(f"{target}{subset}"):
                for dot_file in os.listdir(f"{target}{subset}/{num_dir}"):
                    if dot_file[-3:]=='dot':
                        with open(f"{target}{subset}/{num_dir}/{dot_file}", 'r') as f:
                            lines = f.readlines()
                        for line in lines:
                            metadata['split'].append(subset)
                            metadata['transcript'].append(line[:-12])
                            metadata['ref'].append(line[-10:-2])
                            metadata['dir'].append(num_dir)

    df = pd.DataFrame(metadata)
    df.to_csv(f'{target}/metadata.csv')
    print(set(df['split']))

    #convert to wav
    wav_list = []
    for split in ['train', 'dev', 'eval']:
        for dir in tqdm(os.listdir(f'{target}{split}'), desc=split):        
            list_files = os.listdir(f'{target}{split}/{dir}')
            for file in list_files:
                if file[-3:]=='wv1':
                    if file[:-3]+'wv2' in list_files:
                        if not os.path.exists(f"{file[:-3]}wav"): os.system(f"ffmpeg -loglevel quiet -y -i {target}{split}/{dir}/{file} {target}{split}/{dir}/{file[:-3]}wav")
                        wav_list.append(file[:-4])
                    else:
                        print(f"{file[:-3]+'wv2'} is missing!")

    print(len(wav_list), df.shape)
    df = df[df['ref'].isin(wav_list)]
    print(set(df['split']))
    df = df.drop_duplicates()
    print(len(wav_list), df.shape)
    print(set(df['split']))
    df.to_csv(f'{target}/metadata.csv', index=False)
    print("stage 2 done")

if stage==3:
    #Stage 3: make csv for WSJ0
    df = pd.read_csv(f'{target}/metadata.csv')
    audio_path = []
    audio_len = []
    idx_to_keep = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        split, dir, file = row['split'], row['dir'], row['ref']
        file = f"{target}{split}/{dir}/{file}.wav"
        try:
            audio_path.append(file)
            audio = mutagen.File(file)
            audio_len.append(audio.info.length)
            idx_to_keep.append(idx)
        except:
            print(f"{file} missing!")
    df = df.iloc[idx_to_keep]
    df['audio_path'] = audio_path
    df['audio_len'] = audio_len

    df.pop('ref')
    df.pop('dir')
    for split in set(df['split']):
        subset = df[df['split']==split]
        print(split, subset.shape)
        save_csv(subset, 'WSJ0', split)

if stage==4:
    def filter(i):
        i = i.replace('\\', '')
        i = i.replace('/', '')
        i = i.replace('ELLIPSIS', '')
        i = i.replace('PERIOD', '')
        i = i.replace('COMMA', '')
        i = i.replace('DOUBLE-QUOTE', '')
        i = i.replace('HYPHEN', '')
        i = i.replace('QUOTE', '')
        i = i.replace('PERCENT', '')
        i = i.replace('POINT', '')
        i = i.replace('RIGHT-PAREN', '')
        i = i.replace('LEFT-PAREN', '')
        i = i.replace(' .', '.')
        i = i.replace(' - ', '-')
        i = i.replace('<', '').replace('>', '')
        i = i.replace('[loud_breath]', '')
        i = i.replace('[door_slam]', '')
        i = i.replace('[cross_talk]', '')
        i = i.replace('-DASH', '')
        i = i.replace('[loud_beath]', '')
        i = i.replace('[tongue_click]', '')
        i = i.replace('[laughter]', '')
        i = i.replace('[misc_noise]', '')
        i = i.replace('[beep]', '')
        i = i.replace('[chair_squeak]', '')
        i = i.replace('[disk_noise]', '')
        i = i.replace('[paper_rustle]', '')
        i = i.replace('[bad_recording]', '')
        i = i.replace('[typing]', '')
        i = i.replace('[phone_ring]', '')
        i = i.replace('[lip_smack]', '')
        i = i.replace('[phone_ring]', '')
        i = i.replace('[thump]', '')
        i = i.replace('[movement]', '')
        i = i.replace('[inhalation]', '')
        i = i.replace('[sigh]', '')
        i = i.replace('[tap]', '')
        i = i.replace('[microphone_mvt]', '')
        i = i.replace('[exhalation]', '')
        i = i.replace('[lip-smack]', '')
        i = i.replace('[uh]', '')
        i = i.replace('[throat_clear]', '')
        i = i.replace('[unitelligible]', '')
        i = i.replace('[unintelligible]', '')
        i = i.replace('[door_open]', '')
        i = i.replace('[mm]', '')
        i = i.replace('[ah]', '')
        return i.strip(' ')

    for split in ['eval', 'dev', 'train']:
        file = f'data/WSJ0_{split}.csv'
        df = pd.read_csv(file)
        df['transcript'] = [filter(i) for i in df['transcript']]
        df.to_csv(file, index=False)