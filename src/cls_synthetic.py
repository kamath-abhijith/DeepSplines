'''

DEEP-SPLINE FEEDFORWARD NETWORK FOR CLASSIFICATION

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

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% LOAD DATASET

datafile = 'noise0'
dataset = utils.Synthetic_2D(datafile)

train_samples, test_samples, train_labels, test_labels = \
    utils.train_test_splitter(dataset.samples, dataset.labels)

num_train_samples, input_dim = train_samples.shape
output_dim = len(np.unique(train_labels))

train_samples = utils.numpy_to_torch(train_samples)
test_samples = utils.numpy_to_torch(test_samples)
train_labels = utils.numpy_to_torch(train_labels).type(torch.LongTensor)
test_labels = utils.numpy_to_torch(test_labels).type(torch.LongTensor)

# %% MODEL PARAMETERS

model = models.ds_feedforward_network(2, 500, 2)
# model = models.feedforward_network(2, 500, 2)
criterion = nn.CrossEntropyLoss()

main_optimiser = torch.optim.Adam(model.parameters_no_deepspline(),
    lr=0.001)
aux_optimiser = torch.optim.Adam(model.parameters_deepspline())

# main_optimiser = torch.optim.SGD(model.parameters(),
#     lr=0.001, momentum=0.9)

# %% TRAINING ERROR

num_epochs = 100
training_size = 100
training_error = np.zeros(num_epochs)
lambd = 1e-4

for epoch in tqdm(range(num_epochs)):

    # Forward pass
    outputs = model(train_samples)
    loss = criterion(outputs, train_labels)
    loss = loss + lambd * model.TV2()
    training_error[epoch] = loss.item()
    
    # Backward and optimize
    main_optimiser.zero_grad()
    aux_optimiser.zero_grad()
    loss.backward()
    main_optimiser.step()
    aux_optimiser.step()
    
    if (epoch+1) % num_epochs/10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %% TESTING

with torch.no_grad():
    # Predictions on test data
    outputs = model(test_samples)
    _, predictions = torch.max(outputs.data, 1)
    confusion_mtx = np.zeros((output_dim, output_dim))
    for i in range(output_dim):
        for j in range(output_dim):
            confusion_mtx[i,j] = sum((test_labels==i) & (predictions==j))

    accuracy = sum(np.diag(confusion_mtx))/sum(sum(confusion_mtx)) * 100

    # Predictions on mesh
    num_samples = 100
    x1, x2 = np.meshgrid(np.linspace(0, 2, num_samples), \
        np.linspace(0, 2, num_samples))
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    mesh_samples = np.array([x1, x2]).reshape(2, -1).T
    mesh_samples = utils.numpy_to_torch(mesh_samples)

    mesh_labels = model(mesh_samples)
    _, mesh_labels = torch.max(mesh_labels.data, 1)
    mesh_labels = mesh_labels.reshape(num_samples, num_samples)

# %% PLOTS

plt.figure(figsize=(8,8))
ax = plt.gca()
# save_res = path + 'accuracy_NN_dataset_' + datafile + '_size_' + str(training_size)\
#     + '_method_' + str(training_method)
utils.plot_confusion_matrix(confusion_mtx, ax=ax, map_min=0, map_max=1000,
    title_text=r'ACCURACY: %.2f %%'%(accuracy), show=True)

plt.figure(figsize=(8,8))
ax = plt.gca()
plt.contourf(x1, x2, mesh_labels, alpha=0.2, levels=np.linspace(0, 1, 10))
utils.plot_data2D(train_samples, train_labels, ax=ax,
    xlimits=[0,2], ylimits=[0,1], show=True)

# %%

activation_layer = model.get_deepspline_activations()
locations = activation_layer[0]['locations'][0].numpy()
coefficients = activation_layer[0]['coefficients'].numpy()

plt.figure(figsize=(8,8))
ax = plt.gca()
utils.plot_signal(locations, coefficients[110,:], ax=ax,
    plot_colour='blue', line_width=2,
    xaxis_label=r'$x$',
    xlimits=[-4,4], ylimits=[-1,5])

# %%
