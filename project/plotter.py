#!/usr/bin/env python

import os
import json
import sys
import math
import argparse
import random
import signal
import re

from utils import mkdir, Logger
from dataloader import SimulationDataSet
from models import define_G, define_D, get_scheduler, update_learning_rate, GANLoss
from evaluate import Evaluator
from config import config

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
from matplotlib import pyplot as plt

from torchsummary import summary







def plot_train_losses(losses_txt):

    desc = []
    gen = []
    
    with open(losses_txt) as fp:
        line = fp.readline()
        while line:
            match = re.search('gen:\s*(\d*\.\d+),\s*desc:\s*(\d*\.\d+)', line.strip())
            desc_loss = float(match.group(2))
            gen_loss = float(match.group(1))
            desc.append(desc_loss)
            gen.append(gen_loss)
            line = fp.readline()

    plt.figure(figsize=(10,11), dpi=100)

    plt.suptitle('Model: c_res_l4_nf32_p', fontsize=16)
    
    plt.subplot(2,1,1)
    plt.plot(np.arange(len(desc)), desc, '-b', linewidth=0.9, label='Discriminator')    
    plt.grid(True)
    plt.legend()
    plt.title("Loss over the epochs")
    plt.xlabel("Epoch")
    plt.ylabel('Loss_D')

    plt.subplot(2,1,2)
    plt.plot(np.arange(len(gen)), gen, '-r', linewidth=0.9, label='Generator')    
    plt.grid(True)
    plt.legend()
    plt.title("Loss over the epochs")
    plt.xlabel("Epoch")
    plt.ylabel('Loss_G')

    
    # plt.show()
    plt.savefig("losses_data_plot.png", bbox_inches='tight')
    return 0

def plot_val_res(losses_txt):

    epoch = []
    psnr = []
    mse = []
    
    with open(losses_txt) as fp:
        line = fp.readline()
        while line:
            match = re.search('epoch:\s*(\d*\.?\d+),\s*psnr:\s*(\d*\.\d+)\s*,\s*mse:\s*(\d*\.\d+)', line.strip())

            epoch_cur = float(match.group(1))
            psnr_cur = float(match.group(2))
            mse_cur = float(match.group(3))

            epoch.append(epoch_cur)
            psnr.append(psnr_cur)
            mse.append(mse_cur)
            
            line = fp.readline()

    plt.figure(figsize=(7, 10), dpi=100)

    plt.suptitle('Model: c_res_l4_nf32_p\nValidation set metrics', fontsize=16)
    
    plt.subplot(2,1,1)
    plt.plot(epoch, psnr, '-r', linewidth=0.9, label='PNSR')
    plt.grid(True)
    # plt.legend()
    plt.title("PSNR")
    plt.xlabel("Epoch")
    plt.ylabel('PSNR')
    plt.subplot(2,1,2)

    plt.plot(epoch, mse, '-r', linewidth=0.9, label='MSE')
    plt.grid(True)
    # plt.legend()
    plt.title("MSE")
    plt.xlabel("Epoch")
    plt.ylabel('MSE')

    
    plt.show()
    # plt.savefig("losses_data_plot.png", bbox_inches='tight')
    return 0



def main():
    plot_val_res(sys.argv[1])




if __name__ == '__main__':
    main()
