import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.mean = nn.Linear(512, latent_dim)
        self.cov = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        return self.mean(x), self.cov(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.l1 = nn.Linear(latent_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.out = nn.Linear(512, 784)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.out(x)
        x = rearrange(x, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        return F.sigmoid(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x, sample=False):
        if sample:
            return self.decoder(x)
        eps = torch.randn((x.shape[0], self.latent_dim), requires_grad=False, device=x.device)
        mean, cov = self.encoder(x)
        x = mean + (eps * torch.exp(0.5*cov))
        return self.decoder(x), mean, cov

def main():
    x = torch.randn(64, 1, 28, 28).cuda()
    vae = VAE().cuda()
    print(vae(x).shape)

if __name__ == '__main__':
    main()