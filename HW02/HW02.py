#!/usr/bin/env python
# coding: utf-8

# # **Homework 2 Phoneme Classification**
# 
# * Slides: https://docs.google.com/presentation/d/1v6HkBWiJb8WNDcJ9_-2kwVstxUWml87b9CnA16Gdoio/edit?usp=sharing
# * Kaggle: https://www.kaggle.com/c/ml2022spring-hw2
# * Video: TBA
# 

# In[1]:


# ## Download Data
# Download data from google drive, then unzip it.
# 
# You should have
# - `libriphone/train_split.txt`
# - `libriphone/train_labels`
# - `libriphone/test_split.txt`
# - `libriphone/feat/train/*.pt`: training feature<br>
# - `libriphone/feat/test/*.pt`:  testing feature<br>
# 
# after running the following block.
# 
# > **Notes: if the links are dead, you can download the data directly from [Kaggle](https://www.kaggle.com/c/ml2022spring-hw2/data) and upload it to the workspace, or you can use [the Kaggle API](https://www.kaggle.com/general/74235) to directly download the data into colab.**
# 

# ### Download train/test metadata
# Data directory "../../data/hw2/libriphone"

# ### Preparing Data

# **Helper functions to pre-process the training data from raw MFCC features of each utterance.**
# 
# A phoneme may span several frames and is dependent to past and future frames. \
# Hence we concatenate neighboring phonemes for training to achieve higher accuracy. The **concat_feat** function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.
# 
# Feel free to modify the data preprocess functions, but **do not drop any frame** (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)

# In[3]:


import os
import random
import pandas as pd
import torch
from tqdm import tqdm


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:   # 将一个块整体下移
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:     # 将一个块整体上移
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)    # 将右边的块上移
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)   # 将左边的块下移

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41  # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
            y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]  # 截取这么长
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X


# ## Define Dataset

# In[4]:


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# ## Define Model

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.LSTM(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=3, hidden_dim=256):
        super(Classifier, self).__init__()

        # self.fc = nn.Sequential(
        #     BasicBlock(input_dim, hidden_dim),
        #     *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
        #     nn.Linear(hidden_dim, output_dim)
        # )
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LSTM(hidden_dim,hidden_dim)

        )

    def forward(self, x):
        x = self.fc(x)
        return x


# ## Hyper-parameters

# In[6]:


# data parameters
concat_nframes = 11  # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.8  # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 0  # random seed
batch_size = 512  # batch size
num_epoch = 30  # the number of training epoch
learning_rate = 0.0001  # learning rate
model_path = './model.ckpt'  # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 5  # the number of hidden layers
hidden_dim = 256  # the hidden dim

# ## Prepare dataset and model

# In[9]:


import gc

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat/', phone_path='./libriphone/',
                                   concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat/', phone_path='./libriphone/',
                               concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# In[9]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# In[10]:


import numpy as np


# fix seed
def same_seeds(seed):   # 确保网络每次输入相同时，输出是相同的
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# In[11]:


# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ## Training

# In[12]:


best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()  # 将所有正确分类的样本计数
        train_loss += loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                # get the index of the class with the highest# probability
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

# In[13]:


del train_loader, val_loader
gc.collect()

# ## Testing
# Create a testing dataset, and load model from the saved checkpoint.

# In[14]:


# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone',
                         concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# In[15]:


# load model
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

# Make prediction.

# In[16]:


test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# Write prediction to a CSV file.
# 
# After finish running this block, download the file `prediction.csv` from the file section on the left-hand side and
# submit it to Kaggle.



with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))

