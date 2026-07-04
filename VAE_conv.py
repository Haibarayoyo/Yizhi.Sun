#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import random
import pandas as pd
folder_path_1 = '/root/PetImages/cats'
petlist = []
for item in os.listdir(folder_path_1):
    petlist.append([f'cats/{item}'])

random.shuffle(petlist)
df = pd.DataFrame(petlist, columns=['Images'])
df.to_csv("output_cat.csv", index=False)


# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm 
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.nn.functional import relu
from PIL import Image
from torchvision import models


# In[2]:


class Cats(Dataset):
    def __init__(self,csv_file, root_dir, transform = None, index_range = None):
        self.annotations = pd.read_csv(os.path.join(root_dir,csv_file))
        self.root_dir = root_dir
        self.transform = transform
        if index_range is not None:
            self.annotations = self.annotations.iloc[index_range]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,idx):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[idx,0])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            #print(f"Error reading {img_path}: {e}")
            image = Image.new("RGB", (64,64))

        if self.transform:
            image = self.transform(image)   

        return image


# In[3]:


def create_dataset():
    total_len = len(pd.read_csv('/root/output_cat.csv'))
    transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5,0.5,0.5],
    #                      [0.5,0.5,0.5])
])
    dataset = Cats(csv_file='output_cat.csv',root_dir='/root/PetImages',transform = transform,index_range = range(0,12500))
    return dataset


# In[4]:


dataset = create_dataset()

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 64,
                                          shuffle=True,num_workers=2)


# In[6]:


class VAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1), 
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.ReLU(),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512*2*2, z_dim)
        self.fc_logvar = nn.Linear(512*2*2, z_dim)

        self.fc_decode = nn.Linear(z_dim, 512*2*2)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), 
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),   
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 512, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# In[7]:


def loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl


# In[8]:


model = VAE().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[9]:


epoch_losses = []
epochs = 50

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    beta = min(1.0, epoch / 10)

    for images in data_loader:
        images = images.cuda()

        recon, mu, logvar = model(images)
        loss = loss_function(recon, images, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(data_loader):.2f}, beta={beta:.3f}")


# In[11]:


with torch.no_grad():

    z = torch.randn(64,128).cuda()

    out = model.decode(z)

    out = (out + 1) / 2

    save_image(out,'/root/vae_image.png')

    images = next(iter(data_loader)).cuda()
    out, _, _ = model(images)
    x_concat = torch.cat([images, out], dim=3)
    save_image(x_concat, '/root/vae_result.png')


# In[12]:


reconsPath = '/root/vae_result.png'
Image = mpimg.imread(reconsPath)
plt.imshow(Image)  
plt.axis('off')  
plt.show()

