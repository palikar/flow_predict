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

import matplotlib
if 'DISPLAY' not in os.environ: matplotlib.use('agg')
from matplotlib import pyplot as plt


from torchsummary import summary

plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17)

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
        


def plot_model_comparision(metric, labels, test_metr, test_metr_p, train_metr, train_metr_p, filename):

  barWidth = 0.2
  spacing = 0.05
  r1 = np.arange(len(test_metr[0]))
  r2 = [x + barWidth + spacing for x in r1]
  
  plt.figure(figsize=(12,8), dpi=100)
  plt.suptitle("Models comparison")

  # plt.subplot(2,1,1)
  plt.title("Test")
  plt.barh(r1, test_metr[0], barWidth, edgecolor='black', label='Without pressure', alpha=0.5, xerr=test_metr[1])
  plt.barh(r2, test_metr_p[0], barWidth, edgecolor='black', label='With pressure' , alpha=0.5, xerr=test_metr_p[1])
  plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.2)
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
  plt.ylabel('Model', fontweight='bold')

  plt.yticks([r + barWidth - spacing/2 for r in range(len(test_metr[0]))], labels)
  
  plt.legend(loc='upper left', bbox_to_anchor=(0.7, 1.1), ncol=3, fancybox=True, shadow=True)


  # plt.subplot(2,1,2)
  # plt.title("Train")
  # plt.barh(r1, train_metr[0], barWidth, edgecolor='black', label='Without pressure', alpha=0.5, xerr=train_metr[1])
  # plt.barh(r2, train_metr_p[0], barWidth, edgecolor='black', label='With pressure', alpha=0.5, xerr=train_metr_p[1])
  # plt.grid(b=True, which='major', color='#999999', linestyle='-', alpha=0.2)
  # plt.minorticks_on()
  # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

  # plt.xlabel(metric.upper(), fontweight='bold')
  # plt.ylabel('Model', fontweight='bold')

  # plt.yticks([r + barWidth - spacing/2 for r in range(len(train_metr[0]))], labels)

  plt.savefig(filename, bbox_inches='tight')
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


    def get_model_metric(self, mod, metric='psnr', test=True, avrg=True):
        file_loc = os.path.join(mod,
                                'test' if test else 'train',
                                'metrics_avrg.txt')

        pref = "Avrg " if avrg else "Var "
        with open(file_loc, 'r') as fh:
            for line in fh.readlines():
                if (pref + metric) in line:
                    return float(line.split(':')[1].strip())


    def get_recursive_list(self, mod, metric='psnr'):
      
      rec_dirs = []

      for di in os.listdir(mod):

        if not os.path.isdir(os.path.join(mod, di)):
          continue

        if os.path.isfile(os.path.join(mod, di, 'recursive_application.txt')):
         rec_dirs.append(di)

      rec_lists = {}
      
      for di in rec_dirs:
        
        with open(os.path.join(mod, di, 'recursive_application.txt')) as fh:
          for line in fh.readlines():
            if line.startswith(metric):
              rec_lists[di] = [float(f.strip()) for f in line.split(':')[1].strip().split(',')]

      return rec_lists


    def val_losses(self):
        for mod in self.get_model_dirs():
            model_name = self.get_model_name(mod)
            print('Generating validation loss plot for {}'.format(model_name))
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
            print('Generating models\' loss plot for {}'.format(model_name))
            if os.path.isfile(os.path.join(mod, 'losses.txt')):
                plot_train_losses(os.path.join(mod, 'losses.txt'),
                             model_name,
                             os.path.join(self.get_vis_dir(mod), 'train_losses_plot.png'))
            else:
                print('No losses.txt file for model {}'.format(mod))


    def _metrics_comp(self, file_loc, metric='psnr'):
        print('Plotting', metric)
        hel_d = {}
        for mod in self.get_model_dirs():

          model_name = self.get_model_name(mod)
          met = hel_d.setdefault(rchop(model_name, '_p'), {})

          met.setdefault('train_with_p', [])
          met.setdefault('train_without_p', [])
          met.setdefault('test_with_p', [])
          met.setdefault('test_without_p', [])
          
          if model_name.endswith('_p'):
            met['test_with_p'].append(self.get_model_metric(mod, metric=metric, test=True))
            met['train_with_p'].append(self.get_model_metric(mod, metric=metric, test=False))
          else:
            met['test_without_p'].append(self.get_model_metric(mod, metric=metric, test=True))
            met['train_without_p'].append(self.get_model_metric(mod, metric=metric, test=False))

        labels = []
        
        test_metr = []
        test_metr_p = []
        test_metr_err = []
        test_metr_p_err = []

        train_metr = []
        train_metr_p = []
        train_metr_err = []
        train_metr_p_err = []

        for mod, m_d in hel_d.items():

          test_metr.append(np.mean(m_d['test_without_p']))
          test_metr_err.append(np.std(m_d['test_without_p']))


          test_metr_p.append(np.mean(m_d['test_with_p']))
          test_metr_p_err.append(np.std(m_d['test_with_p']))

          
          train_metr.append(np.mean(m_d['train_without_p']))
          train_metr_err.append(np.std(m_d['train_without_p']))
          
          train_metr_p.append(np.mean(m_d['train_with_p']))
          train_metr_p_err.append(np.std(m_d['train_with_p']))

        
          labels.append(mod)

          


        plot_model_comparision(metric, labels,
                               (test_metr, test_metr_err), (test_metr_p, test_metr_p_err),
                               (train_metr, train_metr_err), (train_metr_p, train_metr_p_err),
                               file_loc)


    def matrics_comp(self):
        self._metrics_comp(os.path.join(self.root_dir, 'Models_PSNR.png'), 'psnr')
        # self._metrics_comp(os.path.join(self.root_dir, 'Models_SSIM.png'), 'ssim')
        # self._metrics_comp(os.path.join(self.root_dir, 'Models_MSE.png'),  'mse')
        self._metrics_comp(os.path.join(self.root_dir, 'Models_COR.png'),  'cor')

        self._metrics_comp(os.path.join(self.root_dir, 'Models_AVRG_DIFF.png'),  'avrg_diff_perc')
        self._metrics_comp(os.path.join(self.root_dir, 'Models_MAX_DIFF.png'),  'max_diff_perc')
        

    def _averge_recursive_plot(self, metrics):


      model_met = {}
      
      for mod in self.get_model_dirs():
        name = self.get_model_name(mod)
        model_met.setdefault(name, {})
        for metric in metrics:
          model_met[name].setdefault(metric, [])
          model_met[name][metric].append(self.get_recursive_list(mod, metric=metric))


      def _add_plot(rec, metric, i):
          mats = {}
          for vals in rec[metric]:
            mse_lists = vals
            for eval_type, val_list in mse_lists.items():
              if 'recursive' not in eval_type:
                continue
              mats.setdefault(eval_type, [])
              mats[eval_type].append(val_list[0:20])

          # plt.subplot(2,2,i)
          for eval_type, eval_mat in list(mats.items()):
            arr = np.array(eval_mat)
            std = np.std(arr, axis=0)
            avrg = np.average(arr, axis=0)
            plt.plot(np.arange(len(avrg)), avrg, linewidth=1, label=eval_type)
            # plt.fill_between(np.arange(len(avrg)), avrg+0.95*std, avrg-0.95*std, linewidth=4, linestyle='dashdot', antialiased=True, alpha=0.2)
          plt.grid(True)
          plt.legend()
          plt.ylabel(metric)
          plt.title('Average {} drop'.format(metric.upper()), fontsize=14)
          
      
      for mod, rec in list(model_met.items()):
        plt.figure(figsize=(11,10), dpi=100)

        cnt = 1
        for metric in metrics:
          _add_plot(rec, metric, cnt)
          cnt += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.root_dir, 'average_recursive_application_{}.png'.format(mod)))
        plt.clf()

    def averge_recursive_plot(self):
      # self._averge_recursive_plot(['psnr', 'cor', 'diff_avrg', 'diff_max'])
      self._averge_recursive_plot(['diff_avrg'])
       
      

    def recursive_plots(self):

      rec = []
      
      for mod in self.get_model_dirs():
        print('Processing directory:', mod)
        
        name = self.get_model_name(mod)
        
        mse = self.get_recursive_list(mod, metric='mse')
        psnr = self.get_recursive_list(mod, metric='psnr')
        cor = self.get_recursive_list(mod, metric='cor')
        ssim = self.get_recursive_list(mod, metric='ssim')
        diff_avrg = self.get_recursive_list(mod, metric='diff_avrg')
        diff_max = self.get_recursive_list(mod, metric='diff_max')

        rec.append(mse)

        plt.figure(figsize=(9, 10), dpi=100)
        

        def add_metric(metr, name, number):
          # plt.subplot(2,2,number)
          for eval_type, val_list in metr.items():
            plt.plot(np.arange(len(val_list)), val_list, linewidth=0.9, label=eval_type)        
            plt.grid(True)
            plt.legend()
            plt.title('Recursive {}'.format(name))
            plt.ylabel(name)


        # add_metric(mse, 'MSE', 1)
        # add_metric(ssim, 'SSIM', 4)

        # add_metric(psnr, 'PSNR', 1)
        # add_metric(cor, 'Cor', 2)
        add_metric(diff_avrg, 'Diff Avrg', 3)
        # add_metric(diff_max, 'Diff max', 4)
        
        # plt.suptitle('Recursive Applications\n Model: {}'.format(name), fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(mod, 'vis', 'recursive_application.png'))
        plt.clf()
        

        
            




def main():

    parser = argparse.ArgumentParser(description='Create some plots with metrics from the evaluation and training')
    parser.add_argument('root', help='Root directory of the models data.')
    args = parser.parse_args()

    plotter = PlotProcessor(args)

    # plotter.val_losses()
    # plotter.train_losses()
    # plotter.matrics_comp()
    # plotter.recursive_plots() 
    plotter.averge_recursive_plot()

    


if __name__ == '__main__':
    main()
