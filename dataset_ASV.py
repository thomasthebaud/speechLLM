import torch
from transformers import AutoProcessor, AutoFeatureExtractor

import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
import torchaudio
from torchtune.datasets import ConcatDataset
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from itertools import combinations

class TestCollator:
    def __init__(self, audio_encoder_name, tokenizer):

        self.audio_encoder_name = audio_encoder_name
        self.tokenizer = tokenizer

        if self.audio_encoder_name in ["facebook/hubert-xlarge-ll60k", "microsoft/wavlm-large", 'microsoft/wavlm-base-plus']:
            self.hubert_processor = AutoFeatureExtractor.from_pretrained(audio_encoder_name) # change according to the encoder
        else:
            self.hubert_processor = None

        pre_speech_prompt = 'Answer by yes or no, are those two audio embeddings from the same speaker:\n<speech>'
        post_speech_prompt = f"</speech>.\nOutput:\n"
        output_prompt_pos = "yes"
        output_prompt_neg = "no"

        pre_tokenized_ids = self.tokenizer(pre_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        post_tokenized_ids = self.tokenizer(post_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        output_tokenized_ids_pos = self.tokenizer(self.tokenizer.bos_token + output_prompt_pos + self.tokenizer.eos_token, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        output_tokenized_ids_neg = self.tokenizer(self.tokenizer.bos_token + output_prompt_neg + self.tokenizer.eos_token, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        self.prompts = pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids_pos, output_tokenized_ids_neg

    def __call__(self, batch):
        enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = [], [], [], [], []
        pre, post, out_pos, out_neg = self.prompts
        for el in batch:
            mel, pos_mel, same = self.process(el)

            enroll_mel.append(mel)
            test_mel.append(pos_mel)
            pre_tokenized_ids.append(pre)
            post_tokenized_ids.append(post)
            output_tokenized_ids.append(out_pos if same else out_neg)

        return (
            self.pad(enroll_mel), 
            self.pad(test_mel),
            self.pad(pre_tokenized_ids).long(), 
            self.pad(post_tokenized_ids).long(), 
            self.pad(output_tokenized_ids).long()
            )

    def process(self, element):
        waveform, pos_wave, same = element
        if 'precomputed' in self.audio_encoder_name : return waveform, pos_wave, same
        if waveform is not None: mel = self.hubert_processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else: mel = None
        if pos_wave is not None: pos_mel = self.hubert_processor(pos_wave.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else: pos_mel = None
        return mel, pos_mel, same

    def pad(self, list_tensors):
        max_len = np.max([i.shape[1] for i in list_tensors])
        # print(list_tensors[0].shape)
        output = torch.zeros((len(list_tensors), max_len))

        for i,t in enumerate(list_tensors):
            # print(t.shape, t.squeeze().shape, len(t.squeeze()), output.shape)
            output[i, :len(t.squeeze())] = t.squeeze() #mono channel only
        return output
                                  
class MyCollator:
    def __init__(self, audio_encoder_name, tokenizer):

        self.audio_encoder_name = audio_encoder_name
        self.tokenizer = tokenizer

        if self.audio_encoder_name in ["facebook/hubert-xlarge-ll60k", "microsoft/wavlm-large", 'microsoft/wavlm-base-plus']:
            self.hubert_processor = AutoFeatureExtractor.from_pretrained(audio_encoder_name) # change according to the encoder
        else:
            self.hubert_processor = None

        pre_speech_prompt = 'Answer by yes or no, are those two audio embeddings from the same speaker:\n<speech>'
        post_speech_prompt = f"</speech>.\nOutput:\n"
        output_prompt_pos = "yes"
        output_prompt_neg = "no"

        pre_tokenized_ids = self.tokenizer(pre_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        post_tokenized_ids = self.tokenizer(post_speech_prompt, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        output_tokenized_ids_pos = self.tokenizer(self.tokenizer.bos_token + output_prompt_pos + self.tokenizer.eos_token, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        output_tokenized_ids_neg = self.tokenizer(self.tokenizer.bos_token + output_prompt_neg + self.tokenizer.eos_token, padding="do_not_pad", return_tensors='pt', truncation=False, add_special_tokens=False)["input_ids"]
        self.prompts = pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids_pos, output_tokenized_ids_neg

    def __call__(self, batch):
        enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = [], [], [], [], []
        pre, post, out_pos, out_neg = self.prompts
        for el in batch:
            mel, pos_mel, neg_mel = self.process(el)

            enroll_mel.append(mel)
            test_mel.append(pos_mel)
            pre_tokenized_ids.append(pre)
            post_tokenized_ids.append(post)
            output_tokenized_ids.append(out_pos)

            enroll_mel.append(mel)
            test_mel.append(neg_mel)
            pre_tokenized_ids.append(pre)
            post_tokenized_ids.append(post)
            output_tokenized_ids.append(out_neg)
            
        return (
            self.pad(enroll_mel), 
            self.pad(test_mel),
            self.pad(pre_tokenized_ids).long(), 
            self.pad(post_tokenized_ids).long(), 
            self.pad(output_tokenized_ids).long()
            )

    def process(self, element):
        waveform, pos_wave, neg_wave = element

        if 'precomputed' in self.audio_encoder_name : return waveform, pos_wave, neg_wave

        if waveform is not None: mel = self.hubert_processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else: mel = None
        if pos_wave is not None: pos_mel = self.hubert_processor(pos_wave.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else: pos_mel = None
        if neg_wave is not None: neg_mel = self.hubert_processor(neg_wave.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
        else: neg_mel = None

        return mel, pos_mel, neg_mel

    def pad(self, list_tensors):
        max_len = np.max([i.shape[1] for i in list_tensors])
        # print(list_tensors[0].shape)
        output = torch.zeros((len(list_tensors), max_len))

        for i,t in enumerate(list_tensors):
            # print(t.shape, t.squeeze().shape, len(t.squeeze()), output.shape)
            output[i, :len(t.squeeze())] = t.squeeze() #mono channel only
        return output

class AudioDataset(Dataset):
    def __init__(self, csv_file, mode='train', max_len = 60):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.mode = mode
        self.max_len = max_len*16_000

    def get_pos_neg(self, spk, idx):
        pos = self.data_frame[(self.data_frame['speaker']==spk) & (self.data_frame.index != idx)].sample()['audio_path'].item()
        neg = self.data_frame[self.data_frame['speaker']!=spk].sample()['audio_path'].item()
        return pos, neg

    def __len__(self):
        return len(self.data_frame)

    def load_audio(self, audio_path):
        if '.mp3' in audio_path:
            waveform, sample_rate = torchaudio.load(audio_path, format='mp3')
        else:
            waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0]==2:waveform=torch.mean(waveform, axis=0).unsqueeze(0)
        if waveform.shape[1]>self.max_len and self.max_len>0: 
            start = int(np.random.rand(1)*(waveform.shape[1]-self.max_len))
            waveform=waveform[:, start:start+self.max_len]
        
        return waveform
    
    def __getitem__(self, idx):
        # Load audio
        audio_row = self.data_frame.iloc[idx]
        waveform = self.load_audio(audio_row['audio_path'])
        pos, neg = self.get_pos_neg(audio_row['speaker'], idx)
        pos_wave = self.load_audio(pos)
        neg_wave = self.load_audio(neg)
        
        return waveform, pos_wave, neg_wave
    
class InstructionalAudioDataset(AudioDataset):
    def __init__(self, csv_file, mode='train', max_len = 60):
        """
        Initialize the class with the specified CSV file, mode, and random keys probability.

        Args:
            csv_file (str): The path to the CSV file.
            mode (str, optional): The mode of the operation, defaults to 'train'.
        Returns:
            None
        """
        super().__init__(csv_file, mode, max_len = max_len)
    
    def __getitem__(self, idx):
        waveform, pos_wave, neg_wave = super().__getitem__(idx)
        return waveform, pos_wave, neg_wave


class NumpyDataset(Dataset):
    def __init__(self, csv_file, mode='train'):
        self.data_frame = pd.read_csv(csv_file)

        if mode!='train':
            keep_spk = []
            spks = list(set(self.data_frame['speaker']))
            for spk in tqdm(spks, total=len(spks), desc='checking speakers with unique ids', mininterval=20):
                if len(self.data_frame[self.data_frame['speaker']==spk])>1: keep_spk.append(spk)
            prev_len = len(self.data_frame)
            self.data_frame = self.data_frame[self.data_frame['speaker'].isin(keep_spk)]
            print(f"Keeping {len(self.data_frame)}/{prev_len} samples")
        
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.mode = mode


    def get_pos_neg(self, spk, idx):
        pos = self.data_frame[(self.data_frame['speaker']==spk) & (self.data_frame.index != idx)].sample()['embedding_path'].item()
        neg = self.data_frame[self.data_frame['speaker']!=spk].sample()['embedding_path'].item()
        return pos, neg

    def __len__(self):
        return len(self.data_frame)

    def load_embedding(self, emb_path):
        emb = torch.from_numpy(np.load(emb_path))
        if len(emb.shape)<2: emb = emb.unsqueeze(0)
        return emb
    
    def __getitem__(self, idx):
        # Load audio
        audio_row = self.data_frame.iloc[idx]
        array = self.load_embedding(audio_row['embedding_path'])
        pos, neg = self.get_pos_neg(audio_row['speaker'], idx)
        pos_array = self.load_embedding(pos)
        neg_array = self.load_embedding(neg)
        
        return array, pos_array, neg_array
    
class InstructionalNumpyDataset(NumpyDataset):
    def __init__(self, csv_file, mode='train'):
        super().__init__(csv_file, mode)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)


class TestNumpyDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.data_frame = df.sample(frac=1, random_state=42).reset_index(drop=True)

    def __len__(self):
        return len(self.data_frame)

    def load_embedding(self, emb_path): 
        emb = torch.from_numpy(np.load(emb_path))
        if len(emb.shape)<2: emb = emb.unsqueeze(0)
        return emb
    
    def __getitem__(self, idx):
        # Load audio
        audio_row = self.data_frame.iloc[idx]
        array = self.load_embedding(audio_row['embedding_path_segment'])
        pos_array = self.load_embedding(audio_row['embedding_path_model'])
        return array, pos_array, audio_row['targettype']=='target'

class TestInstructionalNumpyDataset(TestNumpyDataset):
    def __init__(self, csv_file):
        super().__init__(csv_file)
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)

