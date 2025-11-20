import torch
from torch import nn
from transformers import AutoModel, WavLMModel
from torchaudio.transforms import MFCC
from numpy import min as npmin
#from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder,ft_layers, in_meanpool=[], hybrid=False):
    if name in ["facebook/hubert-xlarge-ll60k", "microsoft/wavlm-large", 'microsoft/wavlm-base-plus']:
        # return TransformerAudioEncoder(model_name=name, finetune=finetune_encoder)
        if len(in_meanpool)>0 and name=='microsoft/wavlm-base-plus': return ModifiedWavLMAudioEncoder(ft_layers, finetune=finetune_encoder, in_meanpool=in_meanpool, hybrid=hybrid)
        else: return TransformerAudioEncoder(ft_layers, model_name=name, finetune=finetune_encoder)
    elif name=='MFCC':
        return MFCC(
            sample_rate=16_000,
            n_mfcc=80,
            )
    else:
        print(f"encoder {name} not in approved list")
        raise NotImplementedError
    
class TransformerAudioEncoder(nn.Module):
    def __init__(self, ft_layers, model_name='microsoft/wavlm-base-plus', finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if ft_layers[0]==0 and ft_layers[1]==100:
            print("Will finetune all encoder parameters")
            for param in self.encoder.parameters():
                param.requires_grad = finetune
        else:
            num_layers = len(self.encoder.encoder.layers)
            print(f"Will finetune layers {ft_layers[0]} to {min(ft_layers[1], num_layers)}")
            for param in self.encoder.encoder.layers[ft_layers[0]:min(ft_layers[1], num_layers)].parameters():
                param.requires_grad = finetune

    def forward(self, x):
        return self.encoder(x).last_hidden_state

class ModifiedWavLMAudioEncoder(nn.Module):
    def __init__(self,ft_layers,  finetune=False, in_meanpool=[[2,1]], hybrid=False):
        super().__init__()
        self.hybrid = hybrid
        self.middle_layer = in_meanpool[1][0]-1
        self.encoder = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
        print("Using Modified WavLM encoder")

        # changing the way layers are processed
        in_meanpool = {a:b for a,b in in_meanpool}
        old_layers = self.encoder.encoder.layers
        new_layers = []
        for i, layer in enumerate(old_layers):
            if not self.hybrid:
                if i in in_meanpool: new_layers.append(WavLMEncoderLayer_proxy(layer, meanpool=in_meanpool[i]))
                else:                new_layers.append(WavLMEncoderLayer_proxy(layer, meanpool=1))
            else:
                if i in in_meanpool: new_layers.append(WavLMEncoderLayer_proxy_hybrid(layer, meanpool=in_meanpool[i]))
                else:                new_layers.append(WavLMEncoderLayer_proxy_hybrid(layer, meanpool=[1,1]))
        self.encoder.encoder.layers = nn.ModuleList(new_layers)

        num_layers = len(self.encoder.encoder.layers)
        if ft_layers[0]==0 and ft_layers[1]==100:
            print("Will find optimal layers to finetune based on the meanpooling layers:")
            ft_layers = (npmin([i for i in in_meanpool]), num_layers)
            print(f"from layer {ft_layers[0]} to the end.")
        else:
            print(f"Will finetune layers {ft_layers[0]} to {min(ft_layers[1], num_layers)}")

        for param in self.encoder.parameters():
            param.requires_grad = False
        # for param in self.encoder.encoder.layers[ft_layers[0]:min(ft_layers[1], num_layers)].parameters():
        #     param.requires_grad = True
        for layer in in_meanpool:
            if layer < len(self.encoder.encoder.layers)-1:
                for param in self.encoder.encoder.layers[layer+1].parameters():
                    param.requires_grad = True
        # for param in self.encoder.encoder.layers[-15:].parameters():
        #     param.requires_grad = finetune
        
    def forward(self, x):
        return self.encoder(x).last_hidden_state


import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, AvgPool1d, AvgPool2d
from transformers.modeling_layers import GradientCheckpointingLayer

class WavLMEncoderLayer_proxy(GradientCheckpointingLayer):
    def __init__(self, layer, meanpool=1):
        super().__init__()
        self.layer = layer
        if meanpool!=1:
            self.pooling_1d = AvgPool1d(meanpool, stride=meanpool)
            self.pooling_2d = AvgPool2d(meanpool, stride=meanpool)
            self.use_pool=True
        else:
            self.use_pool=False

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        states, attention = self.layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    output_attentions=output_attentions,
                    index=index,
                )
        # print(states.shape, index)
        if self.use_pool:
            # print('before:', states.shape, attention.shape)
            states = self.pooling_1d(states.transpose(1,2)).transpose(1,2)
            attention = self.pooling_2d(attention)
            # print('after:', states.shape, attention.shape)
        return (states, attention)

class WavLMEncoderLayer_proxy_hybrid(GradientCheckpointingLayer):
    def __init__(self, layer, meanpool=(1,1)):
        super().__init__()
        self.layer = layer
        if meanpool[0]!=1 and meanpool[1]!=1:
            self.pooling_1d_a = AvgPool1d(meanpool[0], stride=meanpool[0])
            self.pooling_2d_a = AvgPool2d(meanpool[0], stride=meanpool[0])
            self.pooling_1d_b = AvgPool1d(meanpool[1], stride=meanpool[1])
            self.pooling_2d_b = AvgPool2d(meanpool[1], stride=meanpool[1])
            self.use_pool=True
        else:
            self.use_pool=False

        # for 1 min, shapes of the inputs for "in_meanpool":[[7,4],[8,4],[9,4],[10,4]]
        # torch.Size([1, 2999, 768]) 0
        # torch.Size([1, 2999, 768]) 1
        # torch.Size([1, 2999, 768]) 2
        # torch.Size([1, 2999, 768]) 3
        # torch.Size([1, 2999, 768]) 4 
        # torch.Size([1, 2999, 768]) 5
        # torch.Size([1, 2999, 768]) 6
        # torch.Size([1, 2999, 768]) 7
        # torch.Size([1, 749, 768]) 8
        # torch.Size([1, 187, 768]) 9
        # torch.Size([1, 46, 768]) 10
        # torch.Size([1, 11, 768]) 11

        # for 1 min, shapes of the inputs for "in_meanpool":[[7,2],[8,2],[9,2],[10,2]]
        # torch.Size([1, 2999, 768]) 0
        # torch.Size([1, 2999, 768]) 1
        # torch.Size([1, 2999, 768]) 2
        # torch.Size([1, 2999, 768]) 3
        # torch.Size([1, 2999, 768]) 4 
        # torch.Size([1, 2999, 768]) 5
        # torch.Size([1, 2999, 768]) 6
        # torch.Size([1, 2999, 768]) 7
        # torch.Size([1, 1499, 768]) 8
        # torch.Size([1, 749, 768]) 9
        # torch.Size([1, 374, 768]) 10
        # torch.Size([1, 187, 768]) 11

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        states, attention = self.layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    output_attentions=output_attentions,
                    index=index,
                )
        path_b=False
        if index<5 and states.shape[1]<2998:path_b==True
        # print(states.shape, index)
        if self.use_pool:
            # print('before:', states.shape, attention.shape)
            if (
                (index==7 and states.shape[1]!=2999) or 
                (index==8 and states.shape[1]!=749) or 
                (index==9 and states.shape[1]!=187) or 
                (index==10 and states.shape[1]!=46)
            ):path_b=True

            if path_b:
                states = self.pooling_1d_b(states.transpose(1,2)).transpose(1,2)
                attention = self.pooling_2d_b(attention)
            else:
                states = self.pooling_1d_a(states.transpose(1,2)).transpose(1,2)
                attention = self.pooling_2d_a(attention)
            # print('after:', states.shape, attention.shape)
        return (states, attention)
