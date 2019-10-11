#!/usr/bin/env python
from __future__ import print_function

import os
import json
import sys
import math
import argparse

from utils import mkdir
from dataloader import SimulationDataSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


import numpy as np

def get_dataconf_file(args):
    return args.model_type + '_dataconf.txt'


parser = argparse.ArgumentParser(description='The training script of the flowPredict pytorch implementation')
parser.add_argument('--data', dest='data_dir', required=True, help='Root directory of the generated data.')
parser.add_argument('--model-name', dest='model_name', default='s_res_8', required=False, help='Name of the current model being trained')
parser.add_argument('--config', dest='config', required=True, help='Configuration file for the system')
parser.add_argument('--use-pressure', dest='use_pressure', required=False, action='store_true',
                    default=False, help='Should the pressure field images to considered by the models')
parser.add_argument('--model-type', dest='model_type', action='store', default='c', choices=['c', 'vd', 's', 'o'], required=False,
                    help='Type of model to be build. \'c\' - baseline, \'vd\' - fluid viscosity and density, \'s\' - inflow speed, \'o\' - object')
parser.add_argument('--cuda', dest='cuda', action='store_true', default=False, help='Should CUDA be used or not')
parser.add_argument('--threads', dest='threads', type=int, default=4, help='Number of threads for data loader to use')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=4, help='Training batch size.')
parser.add_argument('--test-train-split', dest='test_train_split', type=float, default=0.8, help='The percentage of the data to be used in the training set.')
parser.add_argument('--shuffle', dest='shuffle', default=False, action='store_true', help='Should the training and testing data be shufffled.')
parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs for which the model will be trained')
parser.add_argument('--seed', dest='seed', type=int, default=123, help='Random seed to use. Default=123')



print('===> Setting up basic structures ')

args = parser.parse_args()

print('--Model name:', args.model_name)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device("cuda:0" if args.cuda else "cpu")



with open(args.config, 'r') as config_handle:    
    config = json.load(config_handle)

model_type = args.model_type
test_train_split = args.test_train_split
batch_size = args.batch_size
shuffle_dataset = args.shuffle
num_epochs = args.epochs
threads = args.threads
random_seed = args.seed
model_name = args.model_name

print('--model type:', model_type)
print('--use pressure:', args.use_pressure)
print('--test-train split:', test_train_split)
print('--batch size:', batch_size)
print('--shuffle dataset:', shuffle_dataset)
print('--num epochs:', num_epochs)
print('--random seed:', random_seed)
print('--worker threads:', threads)
print('--cuda:', args.cuda)
print('--device:', device)

print('===> Loading datasets')

dataconf_file = get_dataconf_file(args)
dataset = SimulationDataSet(args.data_dir, dataconf_file, args)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_train_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
train_indices, test_indices = indices[:split], indices[split:]

print('--training samples count:', len(train_indices))
print('--test samples count:', len(test_indices))

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=threads)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=threads)

print('===> Loading model')

print('===> Starting the training loop')

# 
for epoch in range(num_epochs):
    for iteration, data in enumerate(train_loader, 1):
        real_a, real_b = data[0].to(device), data[1].to(device)
        # trainy things


    #updating learning rate

    if epoch % 50 == 0:
        mkdir("checkpoints")
        mkdir(os.path.join("checkpoints", args.model_name))
        # ...
        print("Checkpoint saved to {}".format(os.path.join("checkpoints", args.model_name)))

