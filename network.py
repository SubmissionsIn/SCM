import torch.nn as nn
from torch.nn.functional import normalize
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)
class Network(nn.Module):
    def __init__(self, input_size, feature_dim, high_feature_dim,device):
        super(Network, self).__init__()
        self.encoders = Encoder(input_size, feature_dim).to(device)
        self.decoders = Decoder(input_size, feature_dim).to(device)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(high_feature_dim, 64),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.encoders(x)
        z = normalize(self.feature_contrastive_module(h), dim=1)
        q = self.label_contrastive_module(z)
        xr = self.decoders(h)
        return xr, h, z, q
