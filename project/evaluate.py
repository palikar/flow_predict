import os
import json
import sys
import math
import argparse
import random

import torch
import torch.nn as nn

import numpy as np

from utils import mkdir
from utils import correlation
from utils import merge_and_save
from utils import RedirectStdStreams
from dataloader import UnNormalize

from skimage.metrics import structural_similarity as ssim_metr



class Evaluator:


    
    def __init__(self, args, root_dir, device='cpu'):
        self.args = args
        self.root_dir = root_dir
        self.criterionMSE = nn.MSELoss().to(device)
        self.device = device
        self.denormalize = UnNormalize([0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


    def recusive_application_performance(self, net, dataset, split_point, samples=20):
        net.eval()
        
        start_index = int(split_point - samples/2)
        end_index = int(split_point + samples/2)
        
        mse =  []
        cor =  []
        psnr = []
        ssim = []

        prev_img = None
        input_img = dataset[start_index][0].expand(1,-1,-1,-1)

        for index in range(start_index, end_index):
            predicted = net(input_img)
            target = dataset[index][1].expand(1,-1,-1,-1)
            
            cur_mse = self.criterionMSE(predicted, target).item()
            
            predicted_img = predicted.detach().cpu().numpy()
            target_img = target.detach().cpu().numpy()

            mse += [cur_mse]
            psnr += [10 * math.log10(1 / cur_mse)]
            cor += [np.average(np.array([correlation(predicted_img[i], target_img[i]) for i in range(predicted_img.shape[0])]))]
            ssim += [np.average(np.array([ssim_metr(predicted_img[i].T, target_img[i].T, multichannel=True) for i in range(predicted_img.shape[0])]))]
            
            input_img = predicted

        
        
        with open(os.path.join(self.root_dir, 'recursive_application.txt'), 'w') as list_hand:
            list_hand.write('Split index: {}'.format(samples/2))
            list_hand.write('{} {}\n'.format('mse: ' ,  ','.join(str(i) for i in mse)))
            list_hand.write('{} {}\n'.format('cor: ' ,  ','.join(str(i) for i in cor)))
            list_hand.write('{} {}\n'.format('psnr: ',  ','.join(str(i) for i in psnr)))
            list_hand.write('{} {}\n'.format('ssim: ',  ','.join(str(i) for i in ssim)))
        
            
            
    def individual_images_performance(self, net, test_dataloader):

        net.eval()

        mse = []
        cor = []
        psnr = []
        ssim = []

        for batch in test_dataloader:
            real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)
            predicted = net(real_a)
            
            cur_mse = self.criterionMSE(predicted, real_b).item()
            
            predicted = predicted.detach().cpu().numpy()
            real_b = real_b.detach().cpu().numpy()

            mse += [cur_mse]
            psnr += [10 * math.log10(1 / cur_mse)]
            cor += [np.average(np.array([correlation(predicted[i], real_b[i]) for i in range(predicted.shape[0])]))]
            ssim += [np.average(np.array([ssim_metr(predicted[i].T, real_b[i].T, multichannel=True) for i in range(predicted.shape[0])]))]
            
        mse  = np.array(mse)
        cor  = np.array(cor)
        psnr = np.array(psnr)
        ssim = np.array(ssim)
        
        with open(os.path.join(self.root_dir, 'metrics_avrg.txt'), 'w') as avrg_hand:
            avrg_hand.write('{} {}\n'.format('Avrg mse: ',  np.average(mse)))
            avrg_hand.write('{} {}\n'.format('Avrg cor: ',  np.average(cor)))
            avrg_hand.write('{} {}\n'.format('Avrg psnr: ', np.average(psnr)))
            avrg_hand.write('{} {}\n'.format('Avrg ssim: ', np.average(ssim)))
            
        with open(os.path.join(self.root_dir, 'metrics_list.txt'), 'w') as list_hand:
            list_hand.write('{} {}\n'.format('mse: ' ,  ','.join(str(i) for i in mse)))
            list_hand.write('{} {}\n'.format('cor: ' ,  ','.join(str(i) for i in cor)))
            list_hand.write('{} {}\n'.format('psnr: ',  ','.join(str(i) for i in psnr)))
            list_hand.write('{} {}\n'.format('ssim: ',  ','.join(str(i) for i in ssim)))


    def prepare_tensor_img(self, tens_img):
        d = self.denormalize(tens_img).detach().cpu().numpy()
        x = np.transpose(d[:3], (1,2,0))*255
        y = np.transpose(d[3:], (1,2,0))*255
        return x, y
    

    def snapshots(self, net, sampler, dataset, samples=5):

        for index, i in zip(sampler, range(samples)):
            input_img, target = dataset[index]
            predicted = net(input_img.expand(1,-1,-1,-1))

            
            predicted_x, predicted_y = self.prepare_tensor_img(predicted[0])
            input_img_x, input_img_y = self.prepare_tensor_img(input_img)
            target_x, target_y =       self.prepare_tensor_img(target)

            merge_and_save(target_x, predicted_x,
                           'Real image_x', 'Predicred image_x',
                           './checkpoints/snapshots/x_prediction_{}_{}.png'.format(i, random.randint(0, 10000)))

            merge_and_save(target_y, predicted_y,
                           'Real image_y', 'Predicred image_y',
                           './checkpoints/snapshots/y_prediction_{}_{}.png'.format(i, random.randint(0, 10000)))

            merge_and_save(input_img_x, predicted_x,
                           'Time step t (x)', 'Time step t+1 (x)',
                           './checkpoints/snapshots/x_timestep_{}_{}.png'.format(i, random.randint(0, 10000)))

            merge_and_save(input_img_y, predicted_y,
                           'Time step t (y)', 'Time step t+1 (y)',
                           './checkpoints/snapshots/y_timestep_{}_{}.png'.format(i, random.randint(0, 10000)))
