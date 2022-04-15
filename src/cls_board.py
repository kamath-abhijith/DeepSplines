'''

FEEDFORWARD NEURAL NETWORK ON BOARD DATASET

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import sys
sys.path.append('./..')

import argparse
import numpy as np

import torch
from torch import nn as nn
from tqdm import tqdm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import models
import utils

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% PARSE ARGUMENTS

debug = True

if debug:
    datafile = 'Board0'
    training_size = 1000
    training_method = 'Adam'
    num_epochs = 100
    force_train = False

else:
    parser = argparse.ArgumentParser(
        description = "FEEDFORWARD NEURAL NETWORK WITH ONE HIDDEN LAYER ON BOARD DATASET"
    )

    parser.add_argument('--dataset', help="dataset for training and testing", default='Board0')
    parser.add_argument('--training_size', help="size of training set", type=int, default=1000)
    parser.add_argument('--method', help="method of optimisation", default='Adam')
    parser.add_argument('--num_epochs', help="number of epochs", type=int, default=100)
    parser.add_argument('--force_train', help="force training", type=bool, default=False)

    args = parser.parse_args()

    datafile = args.dataset
    training_size = args.training_size
    training_method = args.method
    num_epochs = args.num_epochs
    force_train = args.force_train

# %% LOAD DATASET

# datafile = 'Board0'
dataset = utils.Board(datafile)

train_samples, test_samples, train_labels, test_labels = \
    utils.train_test_splitter(dataset.samples, dataset.labels)

num_train_samples, input_dim = train_samples.shape
output_dim = len(np.unique(train_labels))

train_samples = utils.numpy_to_torch(train_samples)
test_samples = utils.numpy_to_torch(test_samples)
train_labels = utils.numpy_to_torch(train_labels).type(torch.LongTensor)
test_labels = utils.numpy_to_torch(test_labels).type(torch.LongTensor)

# %% MODEL PARAMETERS

hidden_dim = 10000
num_classes = output_dim
# num_epochs = 1000
# training_method = 'SGD'

model = models.feedforward_network(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()

if training_method == 'SGD':
    learning_rate = 0.01
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
elif training_method == 'Adam':
    learning_rate = 0.01
    mom_weights = (0.9, 0.999)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate,
        betas=mom_weights)

# %% TRAINING

# force_train = False
# training_size = 2000

os.makedirs('./../models/ex2', exist_ok=True)
path = './../models/ex2/'

if os.path.isfile(path + 'model_NN_dataset_' + datafile + '_size_' + \
    str(training_size) + '_method_' + str(training_method) + '.pth') and force_train==False:

    print('PICKING PRE-TRAINED MODEL')
    model = torch.load(path + 'model_NN_dataset_' + datafile + '_size_' + \
        str(training_size) + '_method_' + str(training_method) + '.pth')
    model.eval()

else:
    np.random.seed(34)
    random_idx = np.random.randint(num_train_samples, size=training_size)

    print('TRAINING IN PROGRESS...')

    training_error = np.zeros(num_epochs)
    for epoch in tqdm(range(num_epochs)):

        # Forward pass
        outputs = model(train_samples[random_idx])
        loss = criterion(outputs, train_labels[random_idx])
        training_error[epoch] = loss.item()
        
        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if (epoch+1) % num_epochs/10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model, path + 'model_NN_dataset_' + datafile + '_size_' + \
        str(training_size) + '_method_' + str(training_method) + '.pth')
    print('TRAINING COMPLETE!')
    

# %% TESTING

with torch.no_grad():
    # Predictions on test data
    outputs = model(test_samples)
    _, predictions = torch.max(outputs.data, 1)
    confusion_mtx = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_mtx[i,j] = sum((test_labels==i) & (predictions==j))

    accuracy = sum(np.diag(confusion_mtx))/sum(sum(confusion_mtx)) * 100

    # Predictions on mesh
    num_samples = 100
    x1, x2 = np.meshgrid(np.linspace(-1, 1, num_samples), \
        np.linspace(-1, 1, num_samples))
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    mesh_samples = np.array([x1, x2]).reshape(2, -1).T
    mesh_samples = utils.numpy_to_torch(mesh_samples)

    mesh_labels = model(mesh_samples)
    _, mesh_labels = torch.max(mesh_labels.data, 1)
    mesh_labels = mesh_labels.reshape(num_samples, num_samples)

# %% PLOTS

os.makedirs('./../results/ex2', exist_ok=True)
path = './../results/ex2/'

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'samples_NN_dataset_' + datafile + '_size_' + str(training_size)\
    + '_method_' + str(training_method)
np.random.seed(34)
random_idx = np.random.randint(num_train_samples, size=1000)
plt.contourf(x1, x2, mesh_labels, alpha=0.2, levels=np.linspace(0, 5, 100))
utils.plot_data2D(train_samples[random_idx], train_labels[random_idx], ax=ax,
    xlimits=[-1,1], ylimits=[-1,1], show=False, save=save_res)

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'accuracy_NN_dataset_' + datafile + '_size_' + str(training_size)\
    + '_method_' + str(training_method)
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=1000,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=False, save=save_res)

plt.figure(figsize=(8,8))
ax = plt.gca()
save_res = path + 'loss_NN_dataset_' + datafile + '_size_' + str(training_size)\
     + '_method_' + str(training_method)
utils.plot_signal(np.arange(1,num_epochs+1), training_error, ax=ax,
    xaxis_label=r'EPOCHS', yaxis_label=r'TRAINING ERROR', xlimits=[0,num_epochs],
    ylimits=[0,20], show=False, save=save_res)

# %%
