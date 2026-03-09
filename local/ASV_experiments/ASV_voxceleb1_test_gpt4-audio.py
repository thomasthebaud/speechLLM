import os
# import openai
from openai import OpenAI
import pickle
import pandas as pd
from tqdm import tqdm
import base64
import re

def audio_to_input_audio_block(audio_path: str) -> dict:
    # read + b64
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # infer format from extension
    ext = os.path.splitext(audio_path)[1].lstrip(".").lower()
    return {
        "type": "input_audio",
        "input_audio": {
            "data": audio_b64,
            "format": ext
        }
    }

def ask_with_two_audios(model_name, audio_path_1, audio_path_2, temp=0.2):
    instruction = f"""These are two distinct audios.
    First think about the elements that characterize each speaker, such as their gender, accent, tone, prosody, speech rate, 
    Give the characteristics for each audio
    then from those characteristics, infer the likelihood both speakers are the same. 
    Answer by Yes or No, and give a confidence score between 0 and 100:  
    0 correspond to the certainty they are from different speakers, 
    100 corresponds to the certainty that they are from the same speaker, 
    and 50 means you are uncertain.
    This is the first audio: """

    middle_instruct= "and this is the second audio:"

    
    client = OpenAI(
    api_key=api_key,
    organization=org
    )

    user_content = [
        {"type": "text", "text": instruction},
        audio_to_input_audio_block(audio_path_1),
        {"type": "text", "text": middle_instruct},
        audio_to_input_audio_block(audio_path_2),
    ]

    response = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can understand and respond to speech."},
            {"role": "user", "content": user_content}
        ]
    )

    choice = response.choices[0]

    reply_message = choice.message.content
    return reply_message.replace('.', '.\n')


if __name__=="__main__":
    models = {
        'GPT4.o-audio':'gpt-4o-audio-preview-2025-06-03',
    }

    root = '/export/fs05/tthebau1/EDART/ASV_LLM_experiments/'
    #what gpt to use
    model_name='GPT4.o-audio'
    print(f"Starting ASV evaluation by {model_name}!")

    tgt = f"{root}{model_name}/voxceleb1_test/"
    os.makedirs(tgt, exist_ok=True)
    total = 5000

    for trial_name in ['o', 'e', 'h']:
        trials = pd.read_csv(f"data/trials_{trial_name}.csv")
        trials['modelaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in trials['modelid']]
        trials['segmentaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in trials['segmentid']]
        outputs = []
        output_file = f"local/ASV_experiments/outputs_voxceleb1-{trial_name}.csv"

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                n_lines = len(f.readlines())-2

            print(f"Previously computed {n_lines} trials")
            trials = trials.iloc[n_lines:]
            trials = trials.reset_index(drop=True)

        else:
            with open(output_file, "w") as f:
                f.write(f"modelid\tsegmentid\ttargettype\tnumber\tmessage\n")
            n_lines=0
            print("Starting a new log file")
            
        
        for idx, row in tqdm(trials.iterrows(), total=total, desc=f'computing pairs for trials {trial_name}'):
            reply_message = ask_with_two_audios(model_name=models[model_name], audio_path_1=row['modelaudio'], audio_path_2=row['segmentaudio'])
            reply_message = reply_message.replace('\n', ' ').replace('\t', ' ')
            model, segment, target = row['modelid'], row['segmentid'], row['targettype']
            num = re.findall(r'\d+' , reply_message)
            with open(output_file, "a") as f:
                f.write(f"{model}\t{segment}\t{target}\t{num}\t{reply_message}\n")
            if idx==total: break

        print(f"Finished trials vox1-{trial_name}!")