#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import torch
import struct
import torch.nn as nn
import torch.nn.functional as f
import pandas as pd
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.nn.functional import relu
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
torch.manual_seed(1)
np.random.seed(1)
device = torch.device("cuda:0")


# In[ ]:


class Cats(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(os.path.join(root_dir,csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[idx,0])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            #print(f"Error reading {img_path}: {e}")
            image = Image.new("RGB", (256,256))

        if self.transform:
            image = self.transform(image)   

        return image


# In[ ]:


def dataset():
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(p = 0.5), 
        transforms.ToTensor(),   
        transforms.Normalize(
            [0.5,0.5,0.5],
            [0.5,0.5,0.5]
        )
    ])
    dataset = Cats(csv_file='output.csv',
                   root_dir='/root/PetImages',
                   transform = transform)
    loader = DataLoader(dataset,batch_size = 32, shuffle = True, drop_last = True)
    return loader


# In[ ]:


class res(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,1,1)
        self.norm1 = nn.GroupNorm(min(8,out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,1,1)
        self.norm2 = nn.GroupNorm(min(8,out_ch), out_ch)
        self.silu = nn.SiLU()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.shortcut = nn.Identity()

    def forward(self,x):
        identity = self.shortcut(x)

        t = self.conv1(x)
        t = self.norm1(t)
        t = self.silu(t)

        t = self.conv2(t)
        t = self.norm2(t)

        t = t + identity

        return self.silu(t)


# In[ ]:


class down(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv = nn.Conv2d(c,c,3,2,1)

    def forward(self,x):
        return self.conv(x)

class up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,3,1,1)

    def forward(self,x):
        x = torch.nn.functional.interpolate(
            x,scale_factor = 2,
            mode = 'nearest'
        )
        return self.conv(x)


# In[ ]:


class VAE(nn.Module):
    def __init__(self, ndf,nz):
        super().__init__()
        self.ndf = ndf
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,3,padding = 1),
            res(ndf,ndf),
            down(ndf),
            res(ndf,ndf*2),
            down(ndf*2),
            res(ndf*2,ndf*4),
            down(ndf*4),
            res(ndf*4,ndf*4)
        )
        self.decoder = nn.Sequential(
            res(ndf*4,ndf*4),
            up(ndf*4,ndf*4),
            res(ndf*4,ndf*2),
            up(ndf*2,ndf*2),
            res(ndf*2,ndf),
            up(ndf,ndf),
            res(ndf,ndf),
            nn.GroupNorm(8,64),
            nn.SiLU(),
            nn.Conv2d(64,3,3,padding = 1),
            nn.Tanh()
        )
        self.mu = nn.Conv2d(ndf*4,nz,1)
        self.log_var = nn.Conv2d(ndf*4,nz,1)
        self.decoder_input = nn.Conv2d(nz,ndf*4,1)

    def encode(self,x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def reparameterize(self,mu,log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self,z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


# In[ ]:


loader = dataset()
epoches = 100
vae = VAE(ndf = 64,nz = 64).cuda()
optimizer = torch.optim.Adam(vae.parameters(),lr = 2e-4,betas = (0.5,0.999))
criterion = nn.MSELoss(reduction = 'sum')


# In[ ]:


def kl_loss(mu,log_var):
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl.mean()

for epoch in range(epoches):
    vae.train()
    total_loss = 0
    for step,x in enumerate(loader):
        x = x.cuda()
        recon,mu,log_var = vae(x)
        recon_loss = criterion(recon,x)
        kl = kl_loss(mu, log_var)
        kl_weight = min(epoch/50,1) * 1e-4
        loss = recon_loss + kl_weight * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if step % 100 == 0:
        #     print(f"Epoch {epoch},Step {step},Recon:{recon_loss.item():.4f},KL:{kl.item():.1f} ")

    print(f'Epoch {epoch} Loss:{total_loss/len(loader):.2f}')

    vae.eval()

    with torch.no_grad():

        recon, _, _ = vae(x[:16])

        recon = (recon + 1) / 2
        recon = torch.clamp(recon,0,1)

        real = (x[:16] + 1) / 2

        grid = make_grid(
            torch.cat([real,recon]),
            nrow=8
        )

        plt.figure(figsize=(10,5))
        plt.imshow(
            grid.cpu().permute(1,2,0)
        )
        plt.axis("off")
        plt.show()

