#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.nn.functional import relu, log_softmax, nll_loss
torch.manual_seed(1)


# In[2]:


def create_dataset():
    train_dataset = torchvision.datasets.MNIST(root='/root/train_data',train=True,
                                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    test_dataset = torchvision.datasets.MNIST(root='/root/test_data',train = False, transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))]))
    train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)
    return train_loader, test_loader


# In[3]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(320, 50)
        self.linear2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.pool1(relu(self.conv1(x)))
        x = self.pool2(relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# In[4]:


def train(train_loader):
    model = MnistModel()
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    epoches = 2
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
        torch.save(model.state_dict(),'/root/model.pth')


# In[5]:


def evaluate(test_loader):
    model = MnistModel()
    model = model.cuda()
    model.load_state_dict(torch.load('/root/model.pth'))
    total_correct, total_samples = 0, 0
    for x,y in test_loader:
        model.eval()
        x,y = x.cuda(),y.cuda()
        y_pred = model(x)
        total_correct += (torch.argmax(y_pred,-1) == y).sum()
        total_samples += len(y)
    print(f'acc:{total_correct / total_samples:.2f}')


# In[6]:


if __name__ == '__main__':
    train_loader, test_loader = create_dataset()
    train(train_loader)
    evaluate(test_loader)


# In[7]:


train_dataset = torchvision.datasets.MNIST(root='/root/train_data',download=True,train=True,
                                               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = torchvision.datasets.MNIST(root='/root/test_data',download=True,train = False, transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))]))


# In[21]:


len(train_dataset[0][0][0][5]


# In[20]:


len*test_dataset


# In[ ]:




