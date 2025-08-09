import torch
from transformers import AutoProcessor, AutoFeatureExtractor

import torch
from torch.utils.data import Dataset, Sampler
import torchaudio
from torchtune.datasets import ConcatDataset
import pandas as pd
import random
import numpy as np

class MyCollator:
    def __init__(self, audio_encoder_name, tokenizer):

        self.audio_encoder_name = audio_encoder_name
        self.tokenizer = tokenizer
        if self.audio_encoder_name in ["facebook/hubert-xlarge-ll60k", "microsoft/wavlm-large", 'microsoft/wavlm-base-plus']:
            self.hubert_processor = AutoFeatureExtractor.from_pretrained(audio_encoder_name) # change according to the encoder
        else:
            self.hubert_processor = None
            

    def __call__(self, batch):
        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = [], [], [], []
        for el in batch:
            m,pe,po,o = self.process(el)
            mel.append(m)
            pre_tokenized_ids.append(pe)
            post_tokenized_ids.append(po)
            output_tokenized_ids.append(o)
        return (
            self.pad(mel), 
            self.pad(pre_tokenized_ids).long(), 
            self.pad(post_tokenized_ids).long(), 
            self.pad(output_tokenized_ids).long())

    def process(self, element):
        waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt = element

        if waveform is not None:
            # if "openai/whisper" in self.audio_encoder_name:
            #     mel = self.wav_2_mel(waveform).unsqueeze(0)
            # else:
            mel = self.hubert_processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else:
            mel = None

        pre_tokenized_ids = self.tokenizer(pre_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        post_tokenized_ids = self.tokenizer(post_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        output_tokenized_ids = self.tokenizer(self.tokenizer.bos_token + output_prompt + self.tokenizer.eos_token, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        
        return mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids

    def pad(self, list_tensors):
        max_len = np.max([i.shape[1] for i in list_tensors])
        # print(list_tensors[0].shape)
        output = torch.zeros((len(list_tensors), max_len))

        for i,t in enumerate(list_tensors):
            # print(t.shape, t.squeeze().shape, len(t.squeeze()), output.shape)
            output[i, :len(t.squeeze())] = t.squeeze() #mono channel only
        return output

class AudioDataset(Dataset):
    def __init__(self, csv_file, mode='train', random_keys_prob=0.001, max_len = 60, max_size=-1):
        self.data_frame = pd.read_csv(csv_file)
        if max_size>0 and len(self.data_frame) > max_size : self.data_frame = self.data_frame.sample(n=max_size)
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.mode = mode
        self.random_keys_prob = random_keys_prob
        self.labels = ['transcript', 'gender', 'emotion', 'age', 'accent', 'noises', 'summary'] #'isspeech', 
        self.max_len = max_len*16_000
        # datasets = list(self.data_frame['dataset'])
        # dataset_to_index = {d:i for i,d in enumerate(set(datasets))}
        # dataset_indices = np.array([dataset_to_index[d] for d in datasets])
        # index_to_weight = [len(dataset_indices[dataset_indices==i]) for i in set(dataset_indices)]
        # self.datasets_weights = np.array([len(self.data_frame)/index_to_weight[i] for i in dataset_indices])
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Load audio
        audio_row = self.data_frame.iloc[idx]
        audio_path = audio_row['audio_path']
        # if self.mode=='test':print(idx, audio_row, audio_path, sep='\t')
        if pd.isna(audio_path):
            waveform = None
        elif '.mp3' in audio_path:
            waveform, sample_rate = torchaudio.load(audio_path, format='mp3')
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0]==2:waveform=torch.mean(waveform, axis=0).unsqueeze(0)
        if waveform.shape[1]>self.max_len: waveform=waveform[:, :self.max_len]
        # Prepare labels dictionary based on mode and probability
        labels_str = {}
        if self.mode == 'train' and random.random() < self.random_keys_prob:
            random_labels = random.sample(self.labels, k=random.randint(1, len(self.labels)))
            for label in random_labels:
                if label in audio_row and pd.notnull(audio_row[label]):
                    formatted_label = label.capitalize()
                    if audio_row[label] == True or audio_row[label] == False:
                        labels_str[formatted_label] = audio_row[label]
                    else:
                        labels_str[formatted_label] = str(audio_row[label]).lower()
        else:
            # Most of the time, include all available labels
            for label in self.labels:
                if label in audio_row and pd.notnull(audio_row[label]):
                    formatted_label = label.capitalize()
                    if audio_row[label] == True or audio_row[label] == False:
                        labels_str[formatted_label] = audio_row[label]
                    else:
                        labels_str[formatted_label] = str(audio_row[label]).lower()

        
        if 'context' in audio_row.index:
            conv_history = audio_row['context']
        else:
            conv_history = ""
        
        return waveform, labels_str, conv_history
    
class InstructionalAudioDataset(AudioDataset):
    def __init__(self, csv_file, mode='train', random_keys_prob=0.1, max_len = 60, max_size=-1):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
            random_keys_prob (float, optional): The probability of using random keys, defaults to 0.1.

        Returns:
            None
        """
        super().__init__(csv_file, mode, random_keys_prob, max_len = max_len, max_size=max_size)
        self.instruction_phrases = [
            "Provide the details about the audio",
            "I need the following information from the audio",
            "Tell me about the audio regarding",
            "Extract the following details from the audio",
            "Give me the following information about the audio",
            "Provide details from the audio file",
            "I need information extracted from this speech",
            "Detail the contents of the following audio",
            "Share insights about this speech recording",
            "Describe the specifics captured in this audio file",
            "Summarize the audio's key information",
            "Convey the details embedded in this speech",
            "Outline the main points from this audio file",
            "Unpack the content of the following speech",
            "Present the facts from this audio recording",
            "Elucidate the elements within this speech",
            "Decipher the audio file's information",
            "Break down the details in this speech",
            "Analyze the following audio for details",
            "Report on the specifics of this speech file",
            "Transcribe the key points from this audio",
            "Explain the content of the speech recording",
            "Interpret the information within this audio file",
            "Catalog the details from this speech",
            "Narrate the findings in the audio",
            "Recount the specifics of this speech file",
            "Review the contents of the audio",
            "Assess the information provided by this speech",
            "Evaluate the details in the audio file",
            "Investigate the speech for key information",
            "Scrutinize the audio and provide insights",
            "Inspect the details within this speech",
            "Examine the audio file for specific information",
            "Survey the speech and detail your findings",
            "Study the audio and summarize the content",
            "Audit the speech for important details",
            "Appraise the audio file's key points",
            "Annotate the specifics found in the speech",
            "Dissect the audio to find important information",
            "Extract insights from the speech file",
            "Unveil the details in the audio recording",
            "Shed light on the speech's content",
            "Clarify the specifics within the audio file",
            "Illuminate the information in the speech",
            "Highlight the key points of the audio",
            "Reveal the contents captured in the speech file",
            "Uncover the details within the audio",
            "Delve into the speech for essential information",
            "Probe the audio file for details",
            "Explore the speech recording's specifics",
            "Research the contents of the audio",
            "Inquire into the details of the speech",
            "Sift through the audio for key information",
            "Dive into the speech to extract details",
            "Investigate the nuances of the audio file",
            "Give me the following information about the audio",
            "Fetch information",
            "Give me details about the audio",
            "what does this audio say",
            'what is in the file',
            'give me these details',
        ]
    
    def __getitem__(self, idx):
        waveform, labels_str, conv_history = super().__getitem__(idx)
        instruction_phrase = random.choice(self.instruction_phrases)

        pre_speech_prompt = f"Instruction:\n{instruction_phrase} - ["
        pre_speech_prompt += ', '.join(['IsSpeech' if k == 'isSpeech' else k for k in labels_str.keys()]) + "]\n\nInput:\n<speech>"
        pre_speech_prompt = pre_speech_prompt.replace("Isspeech", "SpeechActivity")
        post_speech_prompt = f"</speech>\n\n" + \
             "Output:\n"
        output_prompt = "{"
        for key, value in labels_str.items():
            if key=="Isspeech": key = 'SpeechActivity'
            output_prompt += f'  "{key}": "{value}", '
        output_prompt = output_prompt.rstrip(',\n') + "}"

        complete_prompt = pre_speech_prompt + post_speech_prompt + output_prompt
        return waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt

class CompositeAudioDataset(Dataset):
    def __init__(self, list_of_datasets, mode='train', random_keys_prob=0.1, max_len = 60, max_size=-1):
        datasets = [
            InstructionalAudioDataset(
                        csv_file = f'./data/{dataset}.csv',
                        mode=mode, 
                        random_keys_prob=random_keys_prob,
                        max_len=max_len,
                        max_size=max_size
                        )
            for dataset in list_of_datasets
            ]
        
        # if only one dataset, use it directly
        if len(datasets) == 1:
            self.dataset = datasets[0]
            self.len = len(self.dataset)
            self.datasets_weights = np.array([1.0])  # single dataset, weight is 1
        else:
            # more than one dataset, use ConcatDataset
            self.dataset = ConcatDataset(datasets)
            self.len = len(self.dataset)
            self.datasets_weights = np.array([self.len/len(d) for d in datasets])


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.dataset[idx]


# Example usage
if __name__ == "__main__":
    dataset = InstructionalAudioDataset(csv_file='dev.csv', mode='test', random_keys_prob=0.0001)
    waveform, pre_speech_prompt, post_speech_prompt, output_prompt, complete_prompt = dataset[121]

    print(complete_prompt)
    print(waveform)
