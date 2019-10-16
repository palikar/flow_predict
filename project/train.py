#!/usr/bin/env python
from __future__ import print_function

import os
import json
import sys
import math
import argparse

from utils import mkdir
from dataloader import SimulationDataSet
from models import define_G, define_D, get_scheduler, update_learning_rate
from models import GANLoss
from evaluate import Evaluator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

from PIL import Image, ImageDraw, ImageFont

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
parser.add_argument('--niter', type=int, dest='niter', default=100, help='Number of iterations at starting learning rate')
parser.add_argument('--niter_decay', type=int, dest='niter_decay', default=100, help='Number of iterations to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--print-summeries', default=False, action='store_true', help='Print summeries of the genereated networks')



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
print('--learning rate policiy:', args.lr_policy)
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

net_g = define_G(6, 6, 32, gpu_id=device).float()
net_d = define_D(12, 64, n_layers_D=3, gpu_id=device).float()

optimizer_g = optim.Adam(net_g.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=0.001, betas=(0.9, 0.999))

net_g_scheduler = get_scheduler(optimizer_g, args)
net_d_scheduler = get_scheduler(optimizer_d, args)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

if args.print_summeries:
    print('Generator network:')
    summary(net_g, input_size=(6, 512, 512))
    print('Detector network:')
    summary(net_d, input_size=(12, 512, 512))

print('===> Starting the training loop')

mkdir("./checkpoints")
mkdir(os.path.join("./checkpoints", args.model_name))
mkdir(os.path.join("./checkpoints", 'snapshots'))

train_loader_len = len(train_loader)
losses_path = "./checkpoints/{}/losses.txt".format(args.model_name)
test_losses_path = "./checkpoints/{}/losses_test.txt".format(args.model_name)

evaluator = Evaluator(args, "./checkpoints", device=device)

# evaluator.snapshots(net_g, test_sampler, dataset, samples=1)
evaluator.individual_images_performance(net_g, test_loader)

# for epoch in range(num_epochs):
#     epoch_loss_d = 0
#     epoch_loss_g = 0

#     for iteration, batch in enumerate(train_loader, 1):

#         net_g.train()
#         net_d.train()
        
#         real_a, real_b = batch[0].to(device), batch[1].to(device)
#         fake_b = net_g(real_a)

#         ##############################
#         # Training the descriminator #
#         ##############################
#         optimizer_d.zero_grad()
        
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = net_d(fake_ab.detach())
#         loss_d_fake = criterionGAN(pred_fake, False)

#         real_ab = torch.cat((real_a, real_b), 1)
#         pred_real = net_d(real_ab)
#         loss_d_real = criterionGAN(pred_real, True)

#         loss_d = (loss_d_fake + loss_d_real) * 0.5

#         loss_d.backward()
#         optimizer_d.step()

#         ##############################
#         #   Training the generator   #
#         ##############################

#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = net_d(fake_ab)
#         loss_g_gan = criterionGAN(pred_fake, True)
#         loss_g_l1 = criterionL1(fake_b, real_b) * 10

#         loss_g = loss_g_gan + loss_g_l1

#         loss_g.backward()
#         optimizer_g.step()

#         epoch_loss_d += loss_d.item()
#         epoch_loss_g += loss_g.item()

#         print("> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
#             epoch, iteration, train_loader_len, loss_d.item(), loss_g.item()))

#     update_learning_rate(net_g_scheduler, optimizer_g)
#     update_learning_rate(net_d_scheduler, optimizer_d)

#     epoch_loss_d /= train_loader_len
#     epoch_loss_g /= train_loader_len
#     with open(losses_path, 'w+') as losses_hand:
#         losses_hand.write('epoch: {}, gen:{:.5f}, desc:{:.5f}'.format(epoch, epoch_loss_g, epoch_loss_d))
    
#     if epoch % 10  == 0:

#         net_g_model_out_path = "./checkpoints/{}/netG_model_epoch_{}.pth".format(args.model_name, epoch)
#         net_d_model_out_path = "./checkpoints/{}/netD_model_epoch_{}.pth".format(args.model_name, epoch)        
#         torch.save(net_g, net_g_model_out_path)
#         torch.save(net_d, net_d_model_out_path)
        
#         print("==> Checkpoint saved to {}".format(os.path.join("checkpoints", args.model_name)))

#         avg_psnr = 0
#         for batch in test_loader:
#             input_img, target = batch[0].to(device), batch[1].to(device)
#             prediction = net_g(input_img)
#             mse = criterionMSE(prediction, target)
#             psnr = 10 * math.log10(1 / mse.item())
#             avg_psnr += psnr
#         avg_psnr /= len(test_loader)
#         print("> Avg. PSNR: {:.5} dB".format(avg_psnr))


    
