# %% LOAD LIBRARIES

import os
import sys
sys.path.append('./..')

import argparse
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from tqdm import tqdm

from scipy import io

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import models
import utils

# %% LOAD DATASET

# Training set
train_data = io.loadmat('./../data/MNIST/mnist_training_data.mat')
train_labels = io.loadmat('./../data/MNIST/mnist_training_label.mat')

num_train_samples = len(train_data['training_data'])
train_samples = train_data['training_data'].reshape(num_train_samples, 28, 28).astype(np.float32)
train_labels = train_labels['training_label'][:,0].astype(np.int64)

# Testing set
test_data = io.loadmat('./../data/MNIST/mnist_test_data.mat')
test_labels = io.loadmat('./../data/MNIST/mnist_test_label.mat')

num_test_samples = len(test_data['test_data'])
test_samples = test_data['test_data'].reshape(num_test_samples, 28, 28).astype(np.float32)
test_labels = test_labels['test_label'][:,0].astype(np.int64)

# test_samples = utils.numpy_to_torch(test_samples)
# test_labels = utils.numpy_to_torch(test_labels)

# %% ADD NOISE
noisy_train_samples = np.zeros((50000,28,28))
noisy_test_samples = np.zeros((10000,28,28))

for idx in tqdm(range(num_train_samples)):  
    noisy_train_samples[idx] = utils.add_noise(train_samples[idx], snr=15)

for idx in tqdm(range(num_test_samples)):
    noisy_test_samples[idx] = utils.add_noise(test_samples[idx], snr=15)

# %% PLOT TRAINING AND GROUND TRUTH IMAGES

f, axes=plt.subplots(2,2)

#showing images with gaussian noise
axes[0,0].imshow(noisy_train_samples[0],cmap="gray")
axes[0,0].set_title("Original Image")
axes[1,0].imshow(train_samples[0],cmap='gray')
axes[1,0].set_title("Noised Image")

#showing images with speckle noise
axes[0,1].imshow(noisy_train_samples[25000],cmap='gray')
axes[0,1].set_title("Original Image")
axes[1,1].imshow(train_samples[25000],cmap="gray")
axes[1,1].set_title("Noised Image")

# %% CREATE DATASET

class mnist_denoising(Dataset):
  
  def __init__(self, noisy, clean, labels, transform):
    self.noisy = noisy
    self.clean = clean
    self.labels = labels
    self.transform = transform
  
  def __len__(self):
    return len(self.noisy)
  
  def __getitem__(self,idx):
    noisy_image = self.noisy[idx]
    clean_image = self.clean[idx]
    label = self.labels[idx]
    
    if self.transform != None:
        noisy_image = self.transform(noisy_image)
        clean_image = self.transform(clean_image)
    
    return (noisy_image,clean_image,label)

tsfms=transforms.Compose([
    transforms.ToTensor()
])

train_set = mnist_denoising(noisy_train_samples, train_samples, train_labels, tsfms)
test_set = mnist_denoising(noisy_test_samples, test_samples, test_labels, tsfms)

train_loader=DataLoader(train_set, batch_size=32, shuffle=True)
test_loader=DataLoader(test_set, batch_size=1, shuffle=True)

# %% SETUP

if torch.cuda.is_available()==True:
  device="cuda:0"
else:
  device ="cpu"
  
model = models.denoising_ae().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,weight_decay=1e-5)

# %% TRAINING

num_epochs = 120
l = len(train_loader)
loss_list = list()
epoch_loss = 0
running_loss = 0
for epoch in range(num_epochs):
  print("Entering Epoch: ",epoch)
  
  for noisy, clean, label in tqdm((train_loader)):
    noisy = noisy.view(noisy.size(0),-1).type(torch.FloatTensor)
    clean = clean.view(clean.size(0),-1).type(torch.FloatTensor)
    dirty, clean = noisy.to(device),clean.to(device)
    
    #-----------------Forward Pass----------------------
    output = model(noisy)
    loss = criterion(output,clean)
    
    #-----------------Backward Pass---------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    epoch_loss += loss.item()

  #-----------------Log-------------------------------
  loss_list.append(running_loss/l)
  running_loss = 0
  print("======> epoch: {}/{}, Loss:{}".format(epoch, num_epochs, loss.item()))

# %% TESTING

f,axes= plt.subplots(6,3,figsize=(20,20))
axes[0,0].set_title("Original Image")
axes[0,1].set_title("Noisy Image")
axes[0,2].set_title("Cleaned Image")

test_imgs = np.random.randint(0,10000,size=6)
for idx in range((6)):
  noisy = test_set[test_imgs[idx]][0]
  clean = test_set[test_imgs[idx]][1]
  label = test_set[test_imgs[idx]][2]
  noisy = noisy.view(noisy.size(0),-1).type(torch.FloatTensor)
  noisy = noisy.to(device)
  output = model(noisy)
  
  output = output.view(1,28,28)
  output = output.permute(1,2,0).squeeze(2)
  output = output.detach().cpu().numpy()
  
  noisy = noisy.view(1,28,28)
  noisy = noisy.permute(1,2,0).squeeze(2)
  noisy = noisy.detach().cpu().numpy()
  
  clean = clean.permute(1,2,0).squeeze(2)
  clean = clean.detach().cpu().numpy()
  
  axes[idx,0].imshow(clean,cmap="gray")
  axes[idx,1].imshow(noisy,cmap="gray")
  axes[idx,2].imshow(output,cmap="gray")

# %%
