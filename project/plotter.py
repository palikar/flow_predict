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

plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})

def rchop(thestring, ending):
  if thestring.endswith(ending):
    return thestring[:-len(ending)]
  return thestring


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

    plt.savefig(file_loc)
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



    plt.savefig(file_loc)
    plt.clf()
    return 0



def plot_recursive(model_name, mse_test, mse_train, ssim_test, ssim_train, test_file, train_file):
    plt.figure(figsize=(10,5), dpi=100)
    plt.suptitle('Recursive Applications\n Model: {}'.format(model_name), fontsize=16)

    plt.subplot(2,1,1)

    plt.plot(np.arange(len(mse_test)), mse_test, 'black', linewidth=0.9, label='MSE')
    plt.grid(True)
    plt.legend()
    plt.title("")
    plt.ylabel('MSE')

    plt.subplot(2,1,2)

    plt.plot(np.arange(len(ssim_test)), ssim_test, 'black', linewidth=0.9, label='SSIM')
    plt.grid(True)
    plt.legend()
    plt.title("")
    plt.xlabel("Recursive application")
    plt.ylabel('SSIM')
    plt.savefig(test_file)

    
    plt.figure(figsize=(10,5), dpi=100)
    plt.suptitle('Recursive Applications\n Model: {}'.format('sadlkj'), fontsize=16)

    plt.subplot(2,1,1)

    plt.plot(np.arange(len(mse_train)), mse_train, 'black', linewidth=0.9, label='MSE')
    plt.grid(True)
    plt.legend()
    plt.title("")
    plt.ylabel('MSE')

    plt.subplot(2,1,2)

    plt.plot(np.arange(len(ssim_train)), ssim_train, 'black', linewidth=0.9, label='SSIM')
    plt.grid(True)
    plt.legend()
    plt.title("")
    plt.xlabel("Recursive application")
    plt.ylabel('SSIM')

    plt.savefig(train_file)
    plt.clf()
        
    

class PlotProcessor():


    def __init__(self, args):
        "docstring"

        self.root_dir = args.root

        for mod in self.get_model_dirs():
            with suppress(FileExistsError):
                os.mkdir(os.path.join(mod, 'vis'))

    def get_model_dirs(self):
        return [os.path.join(self.root_dir, model_dir) for model_dir in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, model_dir))]

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


    def get_model_metric(self, mod, metric='psnr', test=True):
        file_loc = os.path.join(mod,
                                'test' if test else 'train',
                                'metrics_avrg.txt')
        with open(file_loc, 'r') as fh:
            for line in fh.readlines():
                if metric in line:
                    return float(line.split(':')[1].strip())

    def get_recursive_list(self, mod, metric='psnr', test=True):
        file_loc = os.path.join(mod,
                                'test' if test else 'train',
                                'recursive_application.txt')
        with open(file_loc, 'r') as fh:
            for line in fh.readlines():
                if metric in line:
                    return [float(f.strip()) for f in line.split(':')[1].strip().split(',')]


    def val_losses(self):
        for mod in self.get_model_dirs():
            model_name = self.get_model_name(mod)
            print('Generating plot for {}'.format(model_name))
            if os.path.isfile(os.path.join(mod, 'val_losses_test.txt')):
                plot_val_res(os.path.join(mod, 'val_losses_test.txt'),
                             model_name,
                             os.path.join(self.get_vis_dir(mod), 'val_losses_plot.png'))
                plt.cla()
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


    def _metrics_comp(self, file_loc, metric='psnr'):
        x = []
        metr = []
        metr_p = []
        hel_d = {}

        for mod in self.get_model_dirs():
          model_name = self.get_model_name(mod)
          met = hel_d.setdefault(rchop(model_name, '_p'), {})
          if model_name.endswith('_p'):
            met['with_p'] = self.get_model_metric(mod, metric=metric)
          else:
            met['without_p'] = self.get_model_metric(mod, metric=metric)
        for mod, m_d in hel_d.items():
          metr.append(m_d['without_p'])
          metr_p.append(m_d['with_p'])
          x.append(mod)
          

        barWidth = 0.2
        spacing = 0.05
        r1 = np.arange(len(metr))
        r2 = [x + barWidth + spacing for x in r1]

        plt.figure(figsize=(11, 10), dpi=100)
        plt.suptitle("Models comparison")

        # print(r1)
        # print(metr)
        # print('------')
        # print(r2)
        # print(metr_p)

        plt.subplot(2,1,1)
        plt.title("Test")
        plt.barh(r1, metr, barWidth, edgecolor='black', label='Without pressure')
        plt.barh(r2, metr_p, barWidth, edgecolor='black', label='With pressure')
        plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        plt.ylabel('Model', fontweight='bold')
        plt.yticks([r + barWidth - spacing/2 for r in range(len(metr))], x)
        plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.1), ncol=3, fancybox=True, shadow=True)

        x = []
        metr = []
        metr_p = []
        hel_d = {}
        
        for mod in self.get_model_dirs():
          model_name = self.get_model_name(mod)
          met = hel_d.setdefault(rchop(model_name, '_p'), {})
          if model_name.endswith('_p'):
            met['with_p'] = self.get_model_metric(mod, metric=metric, test=False)
          else:
            met['without_p'] = self.get_model_metric(mod, metric=metric, test=False)
        for mod, m_d in hel_d.items():
          metr.append(m_d['without_p'])
          metr_p.append(m_d['with_p'])
          x.append(mod)

          
        barWidth = 0.2
        spacing = 0.05
        r1 = np.arange(len(metr))
        r2 = [x + barWidth + spacing for x in r1]


        plt.subplot(2,1,2)
        plt.title("Train")
        plt.barh(r1, metr, barWidth, edgecolor='black', label='Without pressure')
        plt.barh(r2, metr_p, barWidth, edgecolor='black', label='With pressure')
        plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.2)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.xlabel(metric.upper(), fontweight='bold')
        plt.ylabel('Model', fontweight='bold')
        plt.yticks([r + barWidth - spacing/2 for r in range(len(metr))], x)
        

        plt.savefig(file_loc, bbox_inches='tight')
        plt.clf()


    def matrics_comp(self):
        self._metrics_comp(os.path.join(self.root_dir, 'Models_PSNR.png'), 'psnr')
        self._metrics_comp(os.path.join(self.root_dir, 'Models_SSIM.png'), 'ssim')
        self._metrics_comp(os.path.join(self.root_dir, 'Models_MSE.png'),  'mse')
        self._metrics_comp(os.path.join(self.root_dir, 'Models_COR.png'),  'cor')




    def recursive_plot(self):
        for mod in self.get_model_dirs():
            name = self.get_model_name(mod)
            mse_train = self.get_recursive_list(mod, test=True, metric='mse')
            ssim_train = self.get_recursive_list(mod, test=True, metric='ssim')

            mse_test = self.get_recursive_list(mod, test=True, metric='mse')
            ssim_test = self.get_recursive_list(mod, test=True, metric='ssim')

            test_file = os.path.join(self.get_vis_dir(mod), 'recursive_app_test.png')
            train_file = os.path.join(self.get_vis_dir(mod), 'recursive_app_train.png')

            plot_recursive(name, mse_test, mse_train, ssim_test, ssim_train, test_file, train_file)
        




def main():

    parser = argparse.ArgumentParser(description='Create some plots with metrics from the evaluation and training')
    parser.add_argument('root', help='Root directory of the models data.')
    args = parser.parse_args()

    plotter = PlotProcessor(args)

    plotter.val_losses()
    plotter.train_losses()
    plotter.matrics_comp()
    plotter.recursive_plot()
    


if __name__ == '__main__':
    main()
