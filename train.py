import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from model import VAE
import numpy as np
import math

class ELBO_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCE = nn.BCELoss(reduction="sum")

    def forward(self, z, x, mean, cov):
        reconstruction_loss = self.BCE(z, x)
        kl_divergence = -0.5 * torch.sum(1 + cov - mean.pow(2) - cov.exp())
        return reconstruction_loss + kl_divergence

def main():
    # train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    # test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    # test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = VAE().cuda()
    criterion = ELBO_Loss()
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    for i in range(30):
        total_loss = 0
        for bidx, (x,_) in enumerate(train_loader):
            x = x.cuda()
            output, mean, cov = model(x)
            optimizer.zero_grad()
            loss = criterion(output, x, mean, cov)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'Epoch {i+1} | Loss {total_loss / (60000/32)}')
    #torch.save(model, 'mnist_vae_lowdim')

    #Sample
    with torch.no_grad():
        model.eval().cpu()
        b = 5
        z = torch.randn(b, 2)
        for i in range(b):
            print(z[i].unsqueeze(0).shape)
            output = model(z[i].unsqueeze(0), sample=True)
            output = output.squeeze(0)
            output = rearrange(output, 'c h w -> h w c')
            output.numpy()
            plt.imshow(output)
            plt.show()

    #Visualize Latent Space
    n = 21
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)
    grid = np.array(np.meshgrid(grid_x, grid_y)).reshape(2, -1).T
    x = torch.tensor(grid).float()
    with torch.no_grad():
        model.eval().cpu()
        outputs = model(x, sample=True)
    fig, axes = plt.subplots(n, n, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        img = rearrange(outputs[i], 'c h w -> h w c').numpy()
        ax.imshow(img)
        ax.axis('off')
    plt.show()





if __name__ == '__main__':
    main()