import torch
from torch import nn
from transformers import AutoModel
from torchaudio.transforms import MFCC
#from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder):
    if name in ["facebook/hubert-xlarge-ll60k", "microsoft/wavlm-large", 'microsoft/wavlm-base-plus']:
        return TransformerAudioEncoder(model_name=name, finetune=finetune_encoder)
    elif name=='MFCC':
        return MFCC(
            sample_rate=16_000,
            n_mfcc=80,
            )
    else:
        print(f"encoder {name} not in approved list")
        raise NotImplementedError
    
class TransformerAudioEncoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
        # for param in self.encoder.encoder.layers[-15:].parameters():
        #     param.requires_grad = finetune

    def forward(self, x):
        return self.encoder(x).last_hidden_state

