import os
import json
from tqdm import tqdm
import pandas as pd
import mutagen

from save_csv import save_csv

metadata_root = '/home/tthebau1/EDART/voxceleb_metadata/voxceleb_enrichment_age_gender/dataset/'
metadata = pd.read_csv(metadata_root+'final_dataframe_extended.csv')
print(f"loaded metadata for {len(metadata)} files")
metadata = metadata[metadata['speaker_age_title_only'].notna()]
print(f"{len(metadata)} had age label")
n = 0

for set in ['dev', 'test']:
    print(set)
    source = f'/export/corpora5/VoxCeleb2/{set}/aac'
    labels = {'audio_path':[], 'audio_len':[], 'transcript':[], 'gender':[], 'age':[]}
    for root, dirs, files in os.walk(source):
        for file in files:
            video_id = root.split('/')[-1]
            if video_id in list(metadata['video_id']):
                row = metadata[metadata['video_id']==video_id]
                gender = row['gender_wiki'].iloc[0]
                age = int(row['speaker_age_title_only'].iloc[0])

                audio_path=f"{root}/{file}"
                labels['audio_path'].append(audio_path)
                labels['gender'].append(gender)
                labels['age'].append(age)
                audio = mutagen.File(audio_path)
                labels['audio_len'].append(audio.info.length)


            else: continue

    df = pd.DataFrame(labels)
    print(df.shape)
    save_csv(df, 'voxceleb2_enriched', set)

            
        
    
