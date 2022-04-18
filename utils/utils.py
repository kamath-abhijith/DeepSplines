'''

TOOLS FOR NONLINEAR MODELS

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from scipy import io
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

# %% PLOTTING

colour_list = list(mcolors.BASE_COLORS)

def plot_confusion_matrix(data, ax=None, xaxis_label=r'PREDICTED CLASS',
    yaxis_label=r'TRUE CLASS', map_min=0.0, map_max=1.0, title_text=None,
    show=True, save=False):
    ''' Plots confusion matrix '''
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    ax = sns.heatmap(data, vmin=map_min, vmax=map_max, linewidths=0.5,
        annot=True, fmt=".0f")
    # ax.invert_yaxis()
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_hline(level=0, ax=None, line_colour='black', line_style='-',
    alpha=1, line_width=0.5, annotation=None, pos=(1,1)):
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.axhline(level, color=line_colour, linestyle=line_style,
        linewidth=line_width, alpha=alpha)
    if annotation:
        plt.annotate(annotation, xy=pos, color=line_colour)

def plot_signal(x, y, ax=None, plot_colour='blue', alpha=1, xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='lower left', n_col=2, line_style='-', line_width=None,
    xlimits=[-2,2], ylimits=[-2,2], axis_formatter='%0.1f',
    show=False, save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width,
        color=plot_colour, label=legend_label, zorder=0, alpha=alpha)
    if legend_label and legend_show:
        plt.legend(ncol=n_col, loc=legend_loc, frameon=True, framealpha=0.8,
            facecolor='white')
    
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    # if axis_formatter:
    #     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(axis_formatter))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_data2D(samples, labels, ax=None, title_text=None,
    xlimits=[0,2], ylimits=[0,1], show=True, save=False):
    ''' Plots 2D data with labels '''
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    for _, label in enumerate(np.unique(labels)):
        samples_with_label = samples[np.where(labels==label)]

        plt.scatter(samples_with_label[:,0], samples_with_label[:,1],
            color=colour_list[int(label)])

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title_text)

    plt.ylim(ylimits)
    plt.xlim(xlimits)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

# %% DATASETS

class Synthetic_2D(Dataset):

    def __init__(self, datafile):
        if datafile == 'noise0':
            data = np.loadtxt('./../data/2class-Synthetic/data_noise_0.txt',
                delimiter=",", dtype=np.float32)
        elif datafile == 'noise20':
            data = np.loadtxt('./../data/2class-Synthetic/data_noise_20.txt',
                delimiter=",", dtype=np.float32)
        elif datafile == 'noise40':
            data = np.loadtxt('./../data/2class-Synthetic/data_noise_40.txt',
                delimiter=",", dtype=np.float32)

        self.samples = data[:,:2]
        self.labels = data[:,2]
        self.num_samples = data.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples

class Board(Dataset):
    
    def __init__(self, datafile):
        if datafile == 'Board0':
            data = np.loadtxt('./../data/Board/board_data.txt', delimiter=",",
                dtype=np.float32)
        elif datafile == 'Board10':
            data = np.loadtxt('./../data/Board/board_data_10.txt', delimiter=",",
                dtype=np.float32)
        elif datafile == 'Board25':
            data = np.loadtxt('./../data/Board/board_data_25.txt', delimiter=",",
                dtype=np.float32)

        self.samples = data[:,:2]
        self.labels = data[:, 2]
        self.num_samples = data.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples

class mnist(Dataset):

    def __init__(self, datafile, transform=None):
        
        if datafile == 'mnist':
            data = io.loadmat('./../data/MNIST/mnist_training_data.mat')
            labels = io.loadmat('./../data/MNIST/mnist_training_label.mat')

            self.num_samples = len(data['training_data'])
            self.samples = data['training_data'].reshape(self.num_samples, 28, 28)[:, None].astype(np.float32)
            self.labels = labels['training_label'][:,0].astype(np.int64)
            self.transform = transform

        elif datafile == 'mnist-rot':
            data = io.loadmat('./../data/MNIST/mnist-rot_training_data.mat')
            labels = io.loadmat('./../data/MNIST/mnist-rot_training_label.mat')

            self.num_samples = len(data['train_data'])
            self.samples = data['train_data'].reshape(self.num_samples, 28, 28)[:, None].astype(np.float32)
            self.labels = labels['train_label'][0,:].astype(np.int64)
            self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.samples[index]), self.labels[index]
        else:
            return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples

class Neuro_Dataset(Dataset):

    def __init__(self, parcellation):
        pos_samples = []
        pos_filepath = './../data/Neuro_dataset/Normal_Subjects/'
        pos_paths = os.listdir(pos_filepath)
        for _, path in enumerate(pos_paths):
            data = io.loadmat(pos_filepath + path)
            pos_samples.append(data[parcellation])
        pos_samples = np.array(pos_samples, dtype=np.float32)
        num_pos_samples = pos_samples.shape[0]

        neg_samples = []
        neg_filepath = './../data/Neuro_dataset/Alzheimer_Subjects/'
        neg_paths = os.listdir(neg_filepath)
        for _, path in enumerate(neg_paths):
            data = io.loadmat(neg_filepath + path)
            neg_samples.append(data[parcellation])
        neg_samples = np.array(neg_samples, dtype=np.float32)
        num_neg_samples = neg_samples.shape[0]

        self.samples = np.vstack([pos_samples, neg_samples])[:,None]
        self.labels = np.hstack([np.ones(num_pos_samples), np.zeros(num_neg_samples)])
        self.num_samples = num_pos_samples + num_neg_samples

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples

class EEG_Data(Dataset):

    def __init__(self):
        data = np.array(pd.read_csv('./../data/EEG/eeg_data.csv', skiprows=1))[:,1:].astype(np.float32)

        self.num_samples, dim = data.shape
        self.samples = data[:,:dim-1]
        self.labels = data[:,dim-1]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.num_samples

# %% DATA PROCESSING

def add_noise(data, snr=None, sigma=None, jitter=None):
    '''
    Add white Gaussian noise to data according to given SNR or standard deviation

    :param data: input data vector
    :param snr: desired signal to noise ratio
    :param sigma: desired noise variance

    :returns: noisy data

    '''

    if snr:
        noise = np.random.randn(data.shape[0], data.shape[1])
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(data) \
            * 10 ** (-1.0*snr / 20.)
        return data + noise

    elif sigma:
        noise = np.random.normal(scale=sigma, loc=0, size=data.shape)
        return data + noise

    elif jitter:
        noise = np.random.uniform(low=-1.0*jitter/2., high=jitter/2.,
            size=data.shape)
        return data + noise

def normalised_mean_squared_error(a, b):
    '''
    Computes mean-squared-error between a and b

    :param a: input vector
    :param b: input vector

    :return: mse between a and b

    '''

    return np.abs(np.mean((a-b)**2))/np.abs(np.mean((a)**2))

def numpy_to_torch(data):
    return torch.from_numpy(data)

def train_test_splitter(samples, labels, split_fraction=0.5, state=12):
    return train_test_split(samples, labels, test_size=split_fraction,
        random_state=state)

def train_minibatches(data, batch_size=100, shuffle_flag=True):
    return DataLoader(dataset=data, batch_size=batch_size,
        shuffle=shuffle_flag)