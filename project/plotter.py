#!/usr/bin/env python

import os
import json
import sys
import math
import argparse
import random
import signal
import re
import glob

from contextlib import suppress

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







def plot_train_losses(losses_txt, model_name, file_loc):
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

    plt.figure(figsize=(10,5), dpi=100)

    plt.suptitle('Loss curves\nModel: {}'.format(model_name), fontsize=16)

    plt.subplot(1,2,1)
    plt.plot(np.arange(len(desc)), desc, '-b', linewidth=0.9, label='Discriminator')
    plt.grid(True)
    plt.legend()
    plt.title("")
    plt.xlabel("Epoch")
    plt.ylabel('Loss_D')

    plt.subplot(1,2,2)
    plt.plot(np.arange(len(gen)), gen, '-r', linewidth=0.9, label='Generator')
    plt.grid(True)
    plt.legend()
    plt.title("")
    plt.xlabel("Epoch")
    plt.ylabel('Loss_G')

    plt.savefig(file_loc, bbox_inches='tight')
    return 0



def plot_val_res(losses_txt, model_name, file_loc):

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

    plt.figure(figsize=(10, 5), dpi=100)

    plt.suptitle('Validation set metrics\nModel: {}'.format(model_name), fontsize=16)

    plt.subplot(1,2,1)
    plt.plot(epoch, psnr, '-r', linewidth=0.9, label='PNSR')
    plt.grid(True)
    plt.legend()
    plt.title("PSNR")
    plt.xlabel("Epoch")
    plt.ylabel('PSNR')
    plt.subplot(1,2,2)

    plt.plot(epoch, mse, '-r', linewidth=0.9, label='MSE')
    plt.grid(True)
    plt.legend()
    plt.title("MSE")
    plt.xlabel("Epoch")
    plt.ylabel('MSE')


    # plt.show()
    plt.savefig(file_loc, bbox_inches='tight')
    return 0



class PlotProcessor():


    def __init__(self, args):
        "docstring"

        self.root_dir = args.root

        for mod in self.get_model_dirs():
            with suppress(FileExistsError):
                os.mkdir(os.path.join(mod, 'vis'))

    def get_model_dirs(self):
        return [os.path.join(self.root_dir, model_dir) for model_dir in os.listdir(self.root_dir)]

    def get_vis_dir(self, mod):
        return os.path.join(mod, 'vis')

    def get_model_name(self, mod):

        l = glob.glob(mod + '/c_*')
        if len(l) != 0:
            return os.path.basename(l[0])

        l = glob.glob(mod + '/s_*')
        if len(l) != 0:
            return os.path.basename(l[0])

        l = glob.glob(mod + '/vd_*')
        if len(l) != 0:
            return os.path.basename(l[0])


    def val_losses(self):
        for mod in self.get_model_dirs():
            model_name = self.get_model_name(mod)
            print('Generating plot for {}'.format(model_name))
            if os.path.isfile(os.path.join(mod, 'val_losses_test.txt')):
                plot_val_res(os.path.join(mod, 'val_losses_test.txt'),
                             model_name,
                             os.path.join(self.get_vis_dir(mod), 'val_losses_plot.png'))
            else:
                print('No val_losses_test.txt file for model {}'.format(mod))

    def train_losses(self):
        for mod in self.get_model_dirs():
            model_name = self.get_model_name(mod)
            print('Generating plot for {}'.format(model_name))
            if os.path.isfile(os.path.join(mod, 'losses.txt')):
                plot_train_losses(os.path.join(mod, 'losses.txt'),
                             model_name,
                             os.path.join(self.get_vis_dir(mod), 'train_losses_plot.png'))
            else:
                print('No losses.txt file for model {}'.format(mod))











def main():

    parser = argparse.ArgumentParser(description='Create some plots with metrics from the evaluation and training')
    parser.add_argument('root', help='Root directory of the models data.')
    args = parser.parse_args()

    plotter = PlotProcessor(args)

    plotter.val_losses()
    plotter.train_losses()


if __name__ == '__main__':
    main()
