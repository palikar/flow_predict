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

    plt.subplot(2,1,1)
    plt.figure(figsize=(15,13), dpi=100)
    plt.plot(np.arange(len(desc)), desc, '-b', linewidth=1.0, label='Discriminator')
    
    plt.grid(True)
    plt.legend()
    plt.title("Losses over the epochs")
    plt.xlabel("epoch")
    plt.ylabel('Loss')
    plt.show()

    # plt.savefig(self.directory + "/mdn_data_plot.png", bbox_inches='tight')
    return 0



def main():
    plot_train_losses(sys.argv[1])




if __name__ == '__main__':
    main()
