'''

TOOLS FOR NONLINEAR MODELS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import numpy as np
import torch

from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from deepsplines.ds_modules import dsnn

from matplotlib import pyplot as plt


# %% FEEDFORWARD NETWORK

class feedforward_network(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(feedforward_network, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

class deep_network(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(deep_network, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

# %% CONVOLUTIONAL NETWORK

class conv_network(nn.Module):
    def __init__(self):
        super(conv_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 12, 3, 1)
        self.linear1 = nn.Linear(1728, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

class neuroaal_network(nn.Module):
    def __init__(self):
        super(neuroaal_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1)
        self.conv2 = nn.Conv2d(3, 6, 5, 1)
        self.conv3 = nn.Conv2d(6, 12, 5, 1)
        self.conv4 = nn.Conv2d(12, 6, 3, 1)
        self.linear1 = nn.Linear(924, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

class neuropower_network(nn.Module):
    def __init__(self):
        super(neuropower_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1)
        self.conv2 = nn.Conv2d(3, 6, 5, 1)
        self.conv3 = nn.Conv2d(6, 12, 5, 1)
        self.conv4 = nn.Conv2d(12, 6, 3, 1)
        self.linear1 = nn.Linear(630, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

class eeg_network(nn.Module):

    def __init__(self):
        super(eeg_network, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, 3, 1)
        self.conv2 = nn.Conv1d(3, 6, 5, 1)
        self.conv3 = nn.Conv1d(6, 3, 3, 1)
        self.linear1 = nn.Linear(510, 128)
        self.linear2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

# %% AUTOENCODERS

class denoising_ae(nn.Module):
    def __init__(self):
        super(denoising_ae, self).__init__()
        self.linear1 = nn.Linear(28*28, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)

        self.linear4 = nn.Linear(64, 128)
        self.linear5 = nn.Linear(128, 256)
        self.linear6 = nn.Linear(256, 28*28)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(out)
        out = self.relu(out)
        out = self.linear6(out)
        out = self.sigmoid(out)

        return out

# %% DEEP SPLINES

class ds_feedforward_network(dsnn.DSModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc_ds = nn.ModuleList()

        # deepspline parameters
        opt_params = {
            'size': 51,
            'range_': 4,
            'init': 'leaky_relu',
            'save_memory': False
        }

        # fully-connected layer with 120 output units
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_ds.append(dsnn.DeepBSpline('fc', hidden_dim, **opt_params))
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.fc_ds[0](self.fc1(x))
        x = self.fc2(x)

        return x

class ds_denoising_ae(dsnn.DSModule):
    def __init__(self):
        super(ds_denoising_ae, self).__init__()

        self.fc_ds = nn.ModuleList()

        # deepspline parameters
        opt_params = {
            'size': 51,
            'range_': 4,
            'init': 'leaky_relu',
            'save_memory': False
        }

        self.linear1 = nn.Linear(28*28, 64)
        self.fc_ds.append(dsnn.DeepBSpline('fc', 64, **opt_params))
        self.linear2 = nn.Linear(64, 28*28)
        # self.fc_ds.append(dsnn.DeepBSpline('fc', 128, **opt_params))
        # self.linear3 = nn.Linear(128, 64)
        # self.fc_ds.append(dsnn.DeepBSpline('fc', 64, **opt_params))
        # self.linear4 = nn.Linear(64, 128)
        # self.fc_ds.append(dsnn.DeepBSpline('fc', 128, **opt_params))
        # self.linear5 = nn.Linear(128, 256)
        # self.fc_ds.append(dsnn.DeepBSpline('fc', 256, **opt_params))
        # self.linear6 = nn.Linear(256, 28*28)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc_ds[0](self.linear1(x))
        # out = self.fc_ds[1](self.linear2(out))
        # out = self.fc_ds[2](self.linear3(out))
        # out = self.fc_ds[3](self.linear4(out))
        # out = self.fc_ds[4](self.linear5(out))
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out