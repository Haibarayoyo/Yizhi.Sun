#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import pandas as pd
folder_path_1 = '/root/PetImages/cats'
folder_path_2 = '/root/PetImages/dogs'
petlist = []
for item in os.listdir(folder_path_1):
    petlist.append([f'cats/{item}',0])
print(len(petlist))
for item in os.listdir(folder_path_2):
     petlist.append([f'dogs/{item}',1])
print(len(petlist))
random.shuffle(petlist)
df = pd.DataFrame(petlist, columns=['Images','label'])
df.to_csv("output.csv", index=False)


# In[ ]:


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

torch.manual_seed(1)

class CatsAndDogsDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None,index_range = None):
        self.annotations = pd.read_csv(os.path.join(root_dir,csv_file))
        self.root_dir = root_dir
        self.transform = transform

        if index_range is not None:
            self.annotations = self.annotations.iloc[index_range]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[idx,0])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            #print(f"Error reading {img_path}: {e}")
            image = Image.new("RGB", (224,224))

        y_label = torch.tensor(
            int(self.annotations.iloc[idx,1]),
            dtype=torch.long
        )

        if self.transform:
            image = self.transform(image)   

        return image, y_label


# In[ ]:


def create_dataset():
    total_len = len(pd.read_csv('/root/PetImages/output.csv'))
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),   
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
    train_dataset = CatsAndDogsDataset(csv_file='output.csv',root_dir='/root/PetImages',transform = transform,index_range = range(0,7502))
    test_dataset = CatsAndDogsDataset(csv_file='output.csv',root_dir='/root/PetImages',transform = transform ,index_range =range(7502,total_len))
    train_loader = DataLoader(train_dataset,batch_size=30,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)
    return train_loader, test_loader


# In[ ]:


class Catagory(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(256*26*26, 512)
        self.linear2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pool1(relu(self.bn1(self.conv1(x))))
        x = self.pool2(relu(self.bn2(self.conv2(x))))
        x = self.pool3(relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# In[ ]:


def train(train_loader):
    model = Catagory()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    epoches = 20
    for epoch in range(epoches):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x,y in train_loader:
            model.train()
            x,y = x.cuda(),y.cuda()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_correct += (torch.argmax(y_pred, -1) == y).sum().item()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch + 1},loss:{total_loss / total_samples:.5f},acc:{total_correct / total_samples:.2f},'
              f'time:{time.time() - start:.5f}')
        torch.save(model.state_dict(), 'model_2.pth')


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


def evaluate(test_loader):
    model = Catagory()
    model = model.cuda()
    model.load_state_dict(torch.load('model_2.pth'))
    correct,total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            model.eval()
            x,y = x.cuda(),y.cuda()
            y_pred = model(x)
            correct += (torch.argmax(y_pred, -1) == y).sum()
            total += len(y)
    print(f'acc:{correct/total:.2f}')

