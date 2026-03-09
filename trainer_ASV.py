import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import wandb
import pytorch_lightning as pl
import numpy as np
from jiwer import wer
import torchmetrics
import random
import re
import os
import json

import datetime
from model.encoder import get_audio_encoder, TransformerAudioEncoder
from model.connector import get_connector, CNNConnector
from model.llm import get_llm
from metrics import MAE
from rouge_score import rouge_scorer
# from evaluate import load
import logging

class MeanPooler(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.k = k

    def forward(self, x):
        if self.k==-1: return torch.mean(x, dim=1).unsqueeze(1)
        elif self.k==1: return x
        else: return self.pool(x.transpose(1, 2)).transpose(1, 2)

class SpeechLLMLightning(pl.LightningModule):
    def __init__(self, 
                 audio_encoder_name="speech-tokenizer",
                 connector_args={
                    "name": "cnn",
                    "k": [1,2,1],
                    "n_layers":3,
                    "input_dim": 768,
                    "inside_dim": 512,
                    "output_dim": 2048,
                    "stride":2,
                    "kernel_size":5,
                    "in_meanpool":[]
                    },
                 llm_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 finetune_encoder=False,
                 ft_layers=(0,100),
                 use_audio=True,
                 meanpool=1,
                 use_lora=True,
                 lora_r=32,
                 lora_alpha=2,
                 max_lr=3e-4,
                 enc_lr=2e-6,
                 total_training_step=500000,
                 warmup_steps=1000,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.audio_enc_dim = connector_args["input_dim"]
        self.llm_dim = connector_args['output_dim']
        self.llm_name = llm_name
        self.finetune_encoder = finetune_encoder and use_audio
        self.use_lora = use_lora
        self.max_processed_length = 10
        if "in_meanpool" in connector_args:
            self.modified_encoder = len(connector_args["in_meanpool"])>0
            if self.modified_encoder: print(f"Encoder has been modified, will need longer segments than 1s or hybrid behavior for long/short segments.")
            else: print(f"Encoder has not been modified.")
            self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder, ft_layers, in_meanpool=connector_args["in_meanpool"], hybrid=False)
        else:  
            self.modified_encoder = False
            self.audio_encoder = get_audio_encoder(audio_encoder_name, finetune_encoder, ft_layers)
        print("Encoder Loaded")
        self.connector = get_connector(connector_args)
        self.pooling = MeanPooler(k=meanpool)
        print("Connector Loaded")
        self.llm_tokenizer, self.llm_model = get_llm(llm_name, use_lora, lora_r, lora_alpha)
        print("LLM Loaded")
        self.max_lr = max_lr
        self.enc_lr = enc_lr
        self.total_training_step = total_training_step
        self.warmup_steps = warmup_steps
        self.use_embedding_loss = False
        self.num_validation_samples = 5000
        self.use_audio = use_audio

    def configure_optimizers(self):
        opt = [
            {"params": self.audio_encoder.parameters(), "lr": self.enc_lr if (self.finetune_encoder and self.use_audio) else 0},
            {"params": self.connector.parameters(), "lr": self.max_lr if self.use_audio else 0},
            {"params": self.llm_model.parameters(), "lr": self.max_lr if self.use_lora else 0},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer

    def encode_mel(self, mel, verbose=False):
        if self.finetune_encoder:
            if self.modified_encoder: speech_embeds = self.audio_encoder(mel)
            else: speech_embeds = self.audio_encoder(mel)
        else:
            with torch.no_grad():
                speech_embeds = self.audio_encoder(mel)
        return speech_embeds

    def encode(self, 
        enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, 
        return_embedding_loss=False, chunk_size=60*16_000, test_mode=False, 
        verbose=False):
        batch_size = enroll_mel.shape[0]
        
        # print(f"enroll mel shape = {enroll_mel.shape}")
        enroll_speech_embeds = self.encode_mel(enroll_mel, verbose=verbose)
        # print(f"enroll before CNN: {enroll_speech_embeds.shape}")
        enroll_speech_embeds = self.connector(enroll_speech_embeds)
        # print(f"enroll before pooling: {enroll_speech_embeds.shape}")
        enroll_speech_embeds = self.pooling(enroll_speech_embeds)
        # print(f"enroll after pooling: {enroll_speech_embeds.shape}")

        # print(f"test mel shape = {test_mel.shape}")
        test_speech_embeds = self.encode_mel(test_mel, verbose=verbose)
        test_speech_embeds = self.connector(test_speech_embeds)
        # print(f"test before pooling: {test_speech_embeds.shape}")
        test_speech_embeds = self.pooling(test_speech_embeds)
        # print(f"test after pooling: {test_speech_embeds.shape}")

        if len(enroll_speech_embeds.shape)==2: enroll_speech_embeds = enroll_speech_embeds.unsqueeze(1)
        if len(test_speech_embeds.shape)==2: test_speech_embeds = test_speech_embeds.unsqueeze(1)

        speech_embeds = torch.cat((enroll_speech_embeds, test_speech_embeds), dim=1)
        # print(f"output speech embeddings: {speech_embeds.shape}")

        if 'mistralai' in self.llm_name:
            if self.use_lora: embedder = self.llm_model.model.model.get_input_embeddings()
            else: embedder = self.llm_model.model.get_input_embeddings()
        else:
            if self.use_lora: embedder = self.llm_model.model.model.embed_tokens
            else: embedder = self.llm_model.model.embed_tokens

        pre_prompt_embeds = embedder(pre_tokenized_ids)
        post_prompt_embeds = embedder(post_tokenized_ids)
        output_prompt_embeds = embedder(output_tokenized_ids)

        cat_embs = [pre_prompt_embeds]
        input_token_length = pre_tokenized_ids.shape[1]

        cat_embs.append(speech_embeds)
        input_token_length+=speech_embeds.shape[1]

        cat_embs.append(post_prompt_embeds)
        input_token_length+=post_prompt_embeds.shape[1]

        if not test_mode: cat_embs.append(output_prompt_embeds)
        dtype = self.llm_model.dtype
        combined_embeds = torch.cat(cat_embs, dim=1).to(dtype)
        atts = torch.ones(combined_embeds.size()[:-1], dtype=torch.long).to(combined_embeds.device)

        label_ids = torch.cat([
            torch.ones([batch_size, input_token_length], device=combined_embeds.device)*-100,
            output_tokenized_ids
        ], 1).to(combined_embeds.device).to(torch.int64)
        return combined_embeds, atts, label_ids

    def forward(self, embeds, atts, label_ids):
        # print(embeds.dtype, atts.dtype, label_ids.dtype)
        out = self.llm_model(
            inputs_embeds=embeds,
            attention_mask=atts,
            labels=label_ids,
        )
        return out

    def generate(self, embeds, max_new_tokens=32):
        # print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] start generate, {embeds.shape}")
        outputs = self.llm_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_scores=True
        )
        scores = outputs.scores
        # print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] end of generate.")
        return outputs.sequences, [F.softmax(sc.detach().cpu(), dim=-1).numpy() for sc in scores]
    
    def training_step(self, batch, batch_idx):
        enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=False)
        outputs = self.forward(embeds, atts, label_ids)
        loss =  outputs["loss"]
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Start validation step.")
        enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=False)
        outputs = self.forward(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log("val/loss", loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        # logits = outputs.logits
        # predicted_ids = torch.argmax(logits, dim=-1).cpu()
        embeds, _, _ = self.encode(enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=True)
        predicted_ids, scores = self.generate(embeds=embeds)

        self.get_keys_and_log(scores, predicted_ids, output_tokenized_ids, v='val')

        if batch_idx in self.selected_samples_for_logging:
            sample_idx = self.selected_samples_for_logging.index(batch_idx)
            # Use wandb.log to log prediction and truth texts
            generated_output_text = self.llm_tokenizer.decode(predicted_ids[0], skip_special_tokens=False)
            target_text = self.llm_tokenizer.decode(output_tokenized_ids[0], skip_special_tokens=False)
            out = f"{generated_output_text.replace('<s>', '').replace('</s>', '')}"
            lab = f"{str(target_text).replace('<s>', '').replace('</s>', '')}"
            wandb.log({
                f"val_sample_{sample_idx}_out_label": wandb.Html(f"<pre>{lab} -> {out}</pre>"),
            }, commit=False)
        # print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] end validation step.")

        return {"val_loss": loss}

    def on_test_epoch_start(self):
        # store on CPU to avoid GPU memory growth
        self._test_scores = []
        self._test_labels = []
        self.root_embeddings = 'exp/llm_scores/ASV_'+self.llm_name.split('/')[-1]
        if not os.path.exists(self.root_embeddings): os.makedirs(self.root_embeddings, exist_ok=True)
    
    def test_step(self, batch, batch_idx):
        enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(enroll_mel, test_mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids, test_mode=True)
        predicted_ids, scores = self.generate(embeds=embeds)
        
        # logits = outputs.logits
        # predicted_ids = torch.argmax(logits, dim=-1)

        self.get_keys_and_log(scores, predicted_ids, output_tokenized_ids, v='test')

        return {"test_loss": 0}
    
    def get_keys_and_log(self, scores, predicted_ids, output_tokenized_ids, v='val'):
        # print("scoring function shapes:", len(scores), scores[0].shape, predicted_ids.shape, output_tokenized_ids.shape)
        # Scores are (L, B, D) outputs = tensor([[   1, 4874,  2],[   1,  694,    2]
        batch_size = predicted_ids.shape[0]
        accuracy, prob_yes, prob_no, yes_or_no = [], [], [], []
        
        for b in range(batch_size):
            generated_output_text = self.llm_tokenizer.decode(predicted_ids[b], skip_special_tokens=False).lower()
            target_text = self.llm_tokenizer.decode(output_tokenized_ids[b], skip_special_tokens=False).lower()
            accuracy.append(int(('yes' in generated_output_text and 'yes' in target_text) or ('no' in generated_output_text and 'no' in target_text)))
            yes_or_no.append(int('yes' in generated_output_text or 'no' in generated_output_text))
            if 'yes' in target_text: prob_yes.append(np.max([s[b, 4874] for s in scores]))

            if 'no' in target_text: prob_no.append(np.max([s[b, 694] for s in scores]))

            self._test_scores.append(np.max([s[b, 694] for s in scores])/np.max([s[b, 4874] for s in scores]))
            self._test_labels.append(int('yes' in target_text))

            # if v=='test':self.save_scores([s[b] for s in scores], int('yes' in target_text))

        self.log(f"{v}/accuracy", float(100*np.mean(accuracy)), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{v}/yes_or_no", float(100*np.mean(yes_or_no)), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{v}/prob_yes_on_yes", float(100*np.mean(prob_yes)), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{v}/prob_no_on_no", float(100*np.mean(prob_no)), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def save_scores(self, score_list, target_bool):
        score_array = np.array(score_list)
        # print(score_array.shape)
        np.save(f"{self.root_embeddings}/{len(self._test_labels)}_{target_bool}.npy", score_array)

    def on_test_epoch_end(self):
        eer = self.get_EER()
        self.log("test/eer", eer, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        eer = self.get_EER()
        self.log("val/EER", eer, prog_bar=True, sync_dist=True)

    def get_EER(self):
        scores = np.array(self._test_scores, dtype=np.float64).ravel()
        labels = np.array(self._test_labels, dtype=np.int64).ravel()

        assert scores.shape == labels.shape, "scores and labels must have same shape"

        # Sort scores in descending order
        order = np.argsort(scores)[::-1]
        labels_sorted = labels[order]

        # Number of positives and negatives
        P = max(np.sum(labels_sorted), 1)
        N = max(len(labels_sorted) - np.sum(labels_sorted), 1)

        # Cumulative true positives / false positives
        tp = np.cumsum(labels_sorted)
        fp = np.cumsum(1 - labels_sorted)

        # Rates
        tpr = tp / P
        fpr = fp / N
        fnr = 1.0 - tpr

        # Find threshold where |FPR - FNR| is minimal
        idx = np.argmin(np.abs(fpr - fnr))
        eer = 0.5 * (fpr[idx] + fnr[idx])
        
        return 100*float(min(eer, 1-eer))


    def on_validation_epoch_start(self, n=16):
        """Select n=16 random validation samples to log for each epoch."""
        self.selected_samples_for_logging = random.sample(range(self.num_validation_samples), 16)
        self._test_scores = []
        self._test_labels = []

    
    def extract_dictionary(self, input_string):
        pattern = r'<s>\s*(\{.*?\})\s*</s>'
        match = re.search(pattern, input_string, re.DOTALL)
        if match:
            dict_string = match.group(1)
            dict_string = re.sub(r',\s*}', '}', dict_string)
            try:
                return json.loads(dict_string)
            except json.JSONDecodeError as e:
                return {}
        else:
            return {}
    
    def extract_prediction_values(self, input_string):
        json_str_match = re.search(r'<s>\s*\{.*?\}\s*</s>', input_string)
        try:
            json_str = json_str_match.group(0)
        except:
            json_str = '{}'
        return self.extract_dictionary(json_str)
