#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import random
import pandas as pd
folder_path = '/root/PetImages/Cat'
pet_list = []
for item in os.listdir(folder_path):
    pet_list.append([f'cats/{item}'])
random.shuffle(pet_list)
df = pd.DataFrame(pet_list)
df.to_csv('output.csv',index = False)


# In[1]:


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


# In[2]:


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


# In[3]:


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


class TimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.SiLU(),
            nn.Linear(dim*4,dim)
                     )

    def forward(self,t):
        half = self.dim // 2
        emb = torch.exp(
            -math.log(10000)
            *torch.arange(half,device=t.device)/half
        )
        emb = t[:,None] * emb[None]
        emb = torch.cat(
            [emb.sin(),emb.cos()],
            dim = 1
        )
        return self.mlp(emb)


# In[5]:


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,1,1)
        self.norm1 = nn.GroupNorm(8,in_ch)
        self.time = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,1,1)
        self.norm2 = nn.GroupNorm(8,out_ch)
        self.silu = nn.SiLU()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.shortcut = nn.Identity()

    def forward(self,x,t):
        h = self.conv1(self.silu(self.norm1(x)))
        h = h + self.time(t)[:,:,None,None]
        h = self.conv2(self.silu(self.norm2(h)))
        return h + self.shortcut(x)


# In[6]:


class AttentionBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8,channels)
        self.q = nn.Conv2d(channels,channels, 1)
        self.k = nn.Conv2d(channels,channels, 1)
        self.v = nn.Conv2d(channels,channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self,x):
        B,C,H,W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B,C,H*W).permute(0,2,1)
        k = self.k(h).reshape(B,C,H*W)
        v = self.v(h).reshape(B,C,H*W)
        attn = torch.softmax(torch.bmm(q,k)/ math.sqrt(C),dim=-1)
        out = torch.bmm(v,attn.permute(0,2,1))
        out = out.reshape(B,C,H,W)
        return x + self.proj(out)


# In[7]:


class Down(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv = nn.Conv2d(c,c,3,2,1)

    def forward(self,x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv = nn.Conv2d(c,c,3,1,1)

    def forward(self,x):
        x = torch.nn.functional.interpolate(
            x,scale_factor = 2,
            mode = 'nearest'
        )
        return self.conv(x)


# In[8]:


class encoder(nn.Module):
    def __init__(self,in_chan,out_chan,time_dim):
        super().__init__()
        if in_chan == out_chan == 64:
            self.step1 = ResBlock(64,64,time_dim)
            self.step2 = ResBlock(64,64,time_dim)
        else:
            self.step1 = ResBlock(in_chan,out_chan,time_dim)
            self.step2 = ResBlock(out_chan,out_chan,time_dim)

    def forward(self,x,t):
        x = self.step1(x,t)
        x = self.step2(x,t)
        return x

class mid(nn.Module):
    def __init__(self,in_chan,out_chan,time_dim):
        super().__init__()
        self.step1 = ResBlock(in_chan,out_chan,time_dim)
        self.step2 = AttentionBlock(out_chan)
        self.step3 = ResBlock(out_chan,out_chan,time_dim)

    def forward(self,x,t):
        x = self.step1(x,t)
        x = self.step2(x)
        x = self.step3(x,t)
        return x

class decoder(nn.Module):
    def __init__(self,out_chan,in_chan,time_dim):
        super().__init__()
        self.step1 = ResBlock(out_chan,in_chan,time_dim)
        self.step2 = AttentionBlock(in_chan)
        self.step3 = ResBlock(in_chan,in_chan,time_dim)

    def forward(self,x,t):
        x = self.step1(x,t)
        x = self.step2(x)
        x = self.step3(x,t)
        return x


# In[9]:


class UNet(nn.Module):
    def __init__(self,in_channels = 3, time_dim = 256):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(3,64,3,padding = 1)
        self.encoder1 = encoder(64,64,time_dim)
        self.down1 = Down(64)
        self.encoder2 = encoder(64,128,time_dim)
        self.down2 = Down(128)
        self.encoder3 = encoder(128,256,time_dim)
        self.down3 = Down(256)
        self.encoder4 = encoder(256,512,time_dim)
        self.down4 = Down(512)
        self.mid = mid(512,512,time_dim)
        self.up1 = Up(512)
        self.decoder1 = decoder(1024,512,time_dim)
        self.up2 = Up(512)
        self.decoder2 = decoder(768,256,time_dim)
        self.up3 = Up(256)
        self.decoder3 = decoder(384,128,time_dim)
        self.up4 = Up(128)
        self.decoder4 = decoder(192,64,time_dim)
        self.out = nn.Conv2d(64,3,3,padding = 1)

    def forward(self,x,t):
        t = self.time_mlp(t)
        x = self.in_conv(x)
        x = self.encoder1(x,t)
        skip1 = x
        x = self.down1(x)

        x = self.encoder2(x,t)
        skip2 = x
        x = self.down2(x)

        x = self.encoder3(x,t)
        skip3 = x
        x = self.down3(x)

        x = self.encoder4(x,t)
        skip4 = x
        x = self.down4(x)

        x = self.mid(x,t)

        x = self.up1(x)
        x = torch.cat([x,skip4],1)
        x = self.decoder1(x,t)

        x = self.up2(x)
        x = torch.cat([x,skip3],1)
        x = self.decoder2(x,t)

        x = self.up3(x)
        x = torch.cat([x,skip2],1)
        x = self.decoder3(x,t)

        x = self.up4(x)
        x = torch.cat([x,skip1],1)
        x = self.decoder4(x,t)

        return self.out(x)


# In[10]:


def forward_diffusion_sample(x_0,t,noise = None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_hat = alpha_hat[t][:,None,None,None].sqrt()
    sqrt_one_minus = (1 - alpha_hat[t])[:,None,None,None].sqrt()
    return sqrt_alpha_hat * x_0 + sqrt_one_minus * noise, noise


# In[11]:


def sample_ddpm(model,image_size = 64, n_samples = 32):
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size,device = device)

    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.full((n_samples,), t, dtype = torch.long,device = device)
            predicted_noise = model(x, t_tensor)
            beta_t = beta[t]
            alpha_t = alpha[t]
            alpha_hat_t = alpha_hat[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_hat_t).sqrt() * predicted_noise
            ) + beta_t.sqrt() * noise
    return x


# In[12]:


import math
T = 500
beta = torch.linspace(1e-4,0.02,T,device = device)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha,dim = 0)

dataloader = dataset()
model = UNet().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = 2e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
criterion = nn.MSELoss()
for epoch in range(300):
    for step, x in enumerate(dataloader):
        x = x.cuda()
        t = torch.randint(0,T,(x.shape[0],),dtype = torch.long, device = device)
        x_noisy, noise = forward_diffusion_sample(x,t)
        noise_pred = model(x_noisy, t)
        loss = criterion(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f'Epoch {epoch}, Step{step}, Loss:{loss.item():.4f}')

    torch.cuda.empty_cache()
    samples = sample_ddpm(model)
    samples = (samples + 1) / 2
    samples = torch.clamp(samples,0,1)
    grid = make_grid(samples,nrow = 4)
    scheduler.step()
    npimg = grid.cpu().numpy().transpose(1,2,0)
    plt.figure(figsize = (6,6))
    plt.imshow(npimg)
    plt.axis('off')
    plt.title(f'Samples at Epoch {epoch}')
    plt.show()


# In[14]:


torch.save(model.state_dict(),'diffusion2.pth')


# In[18]:


model.eval()
with torch.no_grad():
    for i,x in enumerate(dataloader):
        x = x.to(device)
        t = torch.randint(0,62,(x.shape[0],),device = device)
        x_noisy, noise = forward_diffusion_sample(x,t)
        noise_pred = model(x_noisy, t)
        sqrt_alpha_hat = alpha_hat[t][:,None,None,None].sqrt()
        sqrt_one_minus = (1-alpha_hat[t])[:,None,None,None].sqrt()

        x_recon = (
            x_noisy
            - sqrt_one_minus * noise_pred
        ) / sqrt_alpha_hat
        x_show = (x + 1)/2
        noisy_show = (x_noisy + 1)/2
        recon_show = (x_recon + 1)/2

        x_show = torch.clamp(x_show,0,1)
        noisy_show = torch.clamp(noisy_show,0,1)
        recon_show = torch.clamp(recon_show,0,1)

        for j in range(min(4, x.size(0))):

            plt.figure(figsize=(9,3))

            plt.subplot(1,3,1)
            plt.imshow(x_show[j].cpu().permute(1,2,0))
            plt.title("Original")
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.imshow(noisy_show[j].cpu().permute(1,2,0))
            plt.title("Noisy")
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.imshow(recon_show[j].cpu().permute(1,2,0))
            plt.title("Reconstructed")
            plt.axis("off")

            plt.show()


# In[ ]:




