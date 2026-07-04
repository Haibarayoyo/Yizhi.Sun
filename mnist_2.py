#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import relu
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary
torch.manual_seed(1)


# In[2]:


def create_dataset():
    train_dataset = torchvision.datasets.MNIST(root='/root/myp/model/data',download=True,train=True,
                                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = torchvision.datasets.MNIST(root='/root/myp/model/data',download=True,train = False,transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))]))
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)


# In[3]:


train_loader, test_loader = create_dataset()


# In[11]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32*32,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,128)
        self.linear4 = nn.Linear(128,64)
        self.linear5 = nn.Linear(64,10)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        x = x.view(-1,32*32)
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = relu(self.linear4(x))
        x = relu(self.linear5(x))
        return x


# In[12]:


def train(train_loader):
    model = MnistModel()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))
    criterion = nn.CrossEntropyLoss()
    epoches = 10
    for epoch in range(epoches):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x,y in train_loader:
            x,y = x.cuda(),y.cuda()
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_correct += (torch.argmax(y_pred,-1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        print(f'epoch:{epoch + 1},loss:{total_loss / total_samples:.5f},acc:{total_correct / total_samples:.2f},'
              f'time:{time.time() - start:.5f}')
        torch.save(model.state_dict(),'/root/model_2.pth')


# In[13]:


def evaluate(test_loader):
    model = MnistModel()
    model = model.cuda()
    model.load_state_dict(torch.load('/root/model_2.pth'))
    total_correct, total_samples = 0, 0
    for x,y in test_loader:
        model.eval()
        x,y = x.cuda(),y.cuda()
        y_pred = model(x)
        total_correct += (torch.argmax(y_pred,-1) == y).sum()
        total_samples += len(y)
    print(f'acc:{total_correct / total_samples:.2f}')


# In[14]:


if __name__ == '__main__':
    train_loader, test_loader = create_dataset()
    train(train_loader)
    evaluate(test_loader)


# In[ ]:




