import torch
from torch import nn


def get_connector(args):
    if args['name'] == 'linear':
        return LinearConnector(args['input_dim'], args['output_dim'])
    elif args['name'] == 'mlp':
        return MLPConnector(args['input_dim'], args['output_dim'], args['inside_dim'], args['n_layers'])
    elif args['name'] == 'cnn':
        return CNNConnector(args['input_dim'], args['output_dim'], args['k'], args['n_layers'], args['kernel_size'])
    else:
        raise NotImplementedError

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer(x)
        return x

class MLPConnector(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_layers=1):
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

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k, n_layers, kernel_size):
        super().__init__()
        assert len(k)==n_layers
        self.layer = nn.Sequential(
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels//2, kernel_size=kernel_size,
                    stride=k[0], padding=2),
        nn.ReLU(),
        nn.Conv1d(out_channels//2, out_channels, kernel_size=kernel_size,
                    stride=k[1], padding=2),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                    stride=k[2], padding=2),
        )
    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)

