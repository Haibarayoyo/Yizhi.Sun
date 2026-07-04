#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import random
folder_path = '/root/PetImages/cats'
petlist = []
for item in os.listdir(folder_path):
    petlist.append([f'cats/{item}'])
random.shuffle(petlist)
df = pd.DataFrame(petlist,columns = ['Images'])
df.to_csv('cat_output.csv',index = False)


# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import pandas as pd
import os


# In[10]:


torch.manual_seed(1)
class Cat(Dataset):
    def __init__(self,csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx,0])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB',(64,64))

        y_label = torch.tensor(int(self.annotations.iloc[idx,1]),dtype = torch.long)
        if self.transform:
            image = self.transform(image)

        return image, y_label


# In[13]:


def create_dataset():
    length = len(pd.read_csv('/root/cat_output.csv'))
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    dataset = Cat(csv_file = 'cat_output.csv',root_dir = '/root',transform = transform)
    dataloader = DataLoader(dataset,batch_size = 512,shuffle = True)
    return dataloader


# In[14]:


dataloader = create_dataset()
print(dataloader)


# In[15]:


z_dim = 128
lr = 0.0001
epoches = 50


# In[16]:


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ct1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.ct2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.ct3 = nn.ConvTranspose2d(256,128, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.ct4 = nn.ConvTranspose2d(128,64, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.ct5 = nn.ConvTranspose2d(64,3, kernel_size=4, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self,x):
        x = self.relu(self.bn1(self.ct1(x)))
        x = self.relu(self.bn2(self.ct2(x)))
        x = self.relu(self.bn3(self.ct3(x)))
        x = self.relu(self.bn4(self.ct4(x)))
        x = self.Tanh(self.ct5(x))
        return x


# In[17]:


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cn1 = nn.Conv2d(3,64, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.cn2 = nn.Conv2d(64,128, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(128)
        self.cn3 = nn.Conv2d(128,256, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.cn4 = nn.Conv2d(256,512, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.bn4 = nn.BatchNorm2d(512)
        self.cn5 = nn.Conv2d(512,1, kernel_size = 4, stride = 2, padding = 1, bias = False)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.relu(self.bn1(self.cn1))
        x = self.relu(self.bn2(self.cn2))
        x = self.relu(self.bn3(self.cn3))
        x = self.relu(self.bn4(self.cn4))
        x = self.sigmoid(self.cn5(x))
        return x


# In[18]:


D = Discriminator().cuda()
G = Generator().cuda()
criterion = nn.BCELoss()
G_opt = optim.Adam(G.parameters(),lr = lr,betas=(0.5, 0.999))
D_opt = optim.Adam(D.parameters(),lr = lr,betas=(0.5, 0.999))


# In[19]:


def show_images(fake_images):
    fake_imgs = fake_images.view(-1,1,64,64)
    grid = fake_imgs[:16]
    grid = (grid + 1) / 2
    fig, axes = plt.subplots(4,4,figsize = (4,4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(grid[i].detach().detach().cpu().squeeze(),cmap = 'gray')
        ax.axis('off')
    plt.show()


# In[20]:


for epoch in range(50):
    for i, (real_imgs,_) in enumerate(dataloader):
        real_imgs = real_imgs.view(real_imgs.size(0),-1).cuda()
        batch_size_now = real_imgs.size(0)
        real_labels = torch.full((batch_size_now,1),0.9).cuda()
        fake_labels = torch.full((batch_size_now,1),0.1).cuda()

        z = torch.randn(batch_size_now, z_dim).cuda()
        fake_imgs = G(z).detach()
        loss_real = criterion(D(real_imgs),real_labels)
        loss_fake = criterion(D(fake_imgs),fake_labels)
        D_loss = loss_real + loss_fake
        D_opt.zero_grad()
        D_loss.backward()
        D_opt.step()

        z = torch.randn(batch_size_now, z_dim).cuda()
        fake_imgs = G(z)
        G_loss = criterion(D(fake_imgs),real_labels)
        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()

        if i % 50 == 0:
            print(f'Epoch[{epoch}/{epoches}]'
                  f'Step[{i}/{len(dataloader)}]'
                  f'loss D:{D_loss.item():.4f},loss G:{G_loss.item():.4f}')

    with torch.no_grad():
        z = torch.randn(16, z_dim).cuda()
        fake_imgs = G(z)
        show_images(fake_imgs)


# In[ ]:




