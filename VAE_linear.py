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
from tqdm import tqdm # 进度条
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
            print(f"Error reading {img_path}: {e}")
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


# In[9]:


class VAE(nn.Module):
    def __init__(self,image_size = 64*64*3,h_dim = 2048, z_dim = 128):
        super().__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self,x):
        x = self.relu(self.fc1(x))
        return self.fc2(x), self.fc3(x)

    def reparameterize(self,mu,var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self,z):
        h = self.relu(self.fc4(z))
        return self.sigmoid(self.fc5(h))

    def forward(self,x):
        mu,var = self.encode(x)
        z = self.reparameterize(mu,var)
        return self.decode(z),mu,var


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


# In[9]:


with torch.no_grad():

    z = torch.randn(64,128).cuda()

    samples = model.fc_decode(z)
    samples = samples.view(-1,256,4,4)
    samples = model.decoder(samples)
    save_image(samples, '/root/vae_result.png')


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

