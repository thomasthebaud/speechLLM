import torch
from torch import nn


def get_connector(name, audio_enc_dim, llm_dim, k, dim, layers=1):
    if len(k)==1: k=k[0]
    if name == 'linear-pool':
        return LinearPoolConnector(audio_enc_dim, llm_dim)
    elif name == 'linear':
        return LinearConnector(audio_enc_dim, llm_dim, k)
    elif name == 'mlp':
        return MLPConnector(audio_enc_dim, llm_dim, k, dim, layers)
    elif name == 'cnn':
        return CNNConnector(audio_enc_dim, llm_dim, k)
    else:
        raise NotImplementedError

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer(x)
        return x

class MLPConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k, dim, n_layers=1):
        super().__init__()
        if n_layers==1: self.layer = nn.Linear(in_dim, out_dim)
        elif n_layers==2: 
            self.layer = nn.Sequential(
          nn.Linear(in_dim, dim),
          nn.ReLU(),
          nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class LinearPoolConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearPoolConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU())
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, x):
        # x: [B, T, d]
        x = self.linear1(x)  # x: [B, T, D]
        x = self.linear2(x)
        return x

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        if type(k)==type(list()):
            assert len(k)==3
            self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
                      stride=k[0], padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
                      stride=k[1], padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,
                      stride=k[2], padding=2),
        )
        elif type(k)==type(int()):
            self.layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
                        stride=1, padding=2),
                nn.ReLU(),
                nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
                        stride=k, padding=2),
                nn.ReLU(),
                nn.Conv1d(out_channels, out_channels, kernel_size=5,
                        stride=1, padding=2),
            )
        else:
            print(f"Error: parameter k is not list nor int: k={k}, type={type(k)}")
            exit()
    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)

