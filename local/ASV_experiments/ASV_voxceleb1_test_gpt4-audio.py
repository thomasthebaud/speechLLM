import os
# import openai
from openai import OpenAI
import pickle
import pandas as pd
from tqdm import tqdm
import base64

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
    instruction = f"These are two distinct audios. Answer by yes or no: Are those from the same speaker?"
    org = "your key"
    api_key = "your key"
    client = OpenAI(
    api_key=api_key,
    organization=org
    )

    user_content = [
        {"type": "text", "text": instruction},
        audio_to_input_audio_block(audio_path_1),
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
        'GPT4-realtime':'gpt-realtime-2025-08-28'
    }

    root = '/export/fs05/tthebau1/EDART/ASV_LLM_experiments/'
    #what gpt to use
    model_name='GPT4-realtime'
    print(f"Starting ASV evaluation by {model_name}!")

    tgt = f"{root}{model_name}/voxceleb1_test/"
    os.makedirs(tgt, exist_ok=True)

    trials = pd.read_csv("data/trials_o.csv")
    trials['modelaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in trials['modelid']]
    trials['segmentaudio'] = [f'/export/corpora5/VoxCeleb1_v2/wav/{i[:7]}/{i[8:-6]}/{i[-5:]}.wav' for i in trials['segmentid']]
    outputs = []

    for idx, row in tqdm(trials.iterrows()):
        reply_message = ask_with_two_audios(model_name=models[model_name], audio_path_1=row['modelaudio'], audio_path_2=row['segmentaudio'])
        print(reply_message)
        break

    print("Finished!")