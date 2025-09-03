import torch
from torch import nn


def get_connector(name, audio_enc_dim, llm_dim, k, dim, layers=1, meanpool=1):
    if name == 'linear-pool':
        return LinearPoolConnector(audio_enc_dim, llm_dim, meanpool=meanpool)
    elif name == 'linear':
        return LinearConnector(audio_enc_dim, llm_dim, k)
    elif name == 'mlp':
        return MLPConnector(audio_enc_dim, llm_dim, k, dim, layers)
    elif name == 'cnn':
        return CNNConnector(audio_enc_dim, llm_dim, k, meanpool=meanpool)
    else:
        raise NotImplementedError

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.pool(x)  
        x = x.transpose(1, 2)
        x = self.layer(x)
        return x

class MLPConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k, dim, n_layers=1, meanpool=1):
        super().__init__()
        if n_layers==1: self.layer = nn.Linear(in_dim, out_dim)
        elif n_layers==2: 
            self.layer = nn.Sequential(
          nn.Linear(in_dim, dim),
          nn.ReLU(),
          nn.Linear(dim, out_dim),
        )
        self.pool = nn.AvgPool1d(kernel_size=meanpool, stride=meanpool)

    def forward(self, x):
        self.pool(x)
        x = self.layer(x)
        return x

class LinearPoolConnector(nn.Module):
    def __init__(self, input_dim, output_dim, meanpool):
        super(LinearPoolConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU())
        self.pool = nn.AvgPool1d(kernel_size=meanpool, stride=meanpool)
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, x):
        # x: [B, T, d]
        x = self.linear1(x)  # x: [B, T, D]
        x = x.transpose(1, 2)  # x: [B, D, T]
        x = self.pool(x)  # x: [B, D, T']
        x = x.transpose(1, 2)  # x: [B, T', D]
        x = self.linear2(x)
        return x

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k, meanpool):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
                      stride=k//2, padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
                      stride=k, padding=2),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5,
                      stride=k//2, padding=2),
        )
        self.pool = nn.AvgPool1d(kernel_size=meanpool, stride=meanpool)

    def forward(self, x):
        # print(x.shape)
        x=self.pool(x.transpose(1,2))
        # print(x.shape)
        return self.layer(x).transpose(1,2)



if __name__ == "__main__":
    model = CNNConnector(128, 256)
    x = torch.randn(4, 50, 128)
    z = model(x)
    print(z.shape)