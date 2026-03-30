import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()
file_handler = logging.FileHandler(f"/home/tthebau1/EDART/SpeechLLM/exp/test_predictions/base_TinyLLaMA/switchboard_test/T/base.txt", mode="w")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logging.info('Logger initiated')


df = pd.read_csv("/home/tthebau1/EDART/SpeechLLM/data/switchboard_test.csv")
transcripts, summaries = list(df['transcript']), list(df['summary'])
logging.info('Data Loaded')

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
logging.info('Model Loaded')

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

for transcript, summary in tqdm(zip(transcripts, summaries), total=len(df)):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can summarize conversations transcripts",
        },
        {"role": "user", "content": "Here is a conversation transcript, can you summarize it?\n"+transcript},
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, max_new_tokens=256)
    output = outputs[0]["generated_text"].replace('\n', ' ')
    if '<|assistant|>' in output: output=output.split('<|assistant|>')[-1]
    logging.info(f'[PREDICTION] {output}')
    logging.info(f'[TARGET] {summary}')