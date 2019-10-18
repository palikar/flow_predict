#!/usr/bin/env python

import os
import json
import sys
import math
import argparse
import random
import signal


from utils import mkdir
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

from torchsummary import summary


training_started = False
STOP_TRAINING = False

def signal_handler(sig, frame):
    if not training_started:
        sys.exit(0)

    print('Stoping the training...')
    global STOP_TRAINING
    STOP_TRAINING = True


def create_directories():
    mkdir(config['output_dir'])
    mkdir(os.path.join(config['output_dir'], args.model_name))
    mkdir(os.path.join(config['output_dir'], 'test', 'snapshots'))
    mkdir(os.path.join(config['output_dir'], 'train', 'snapshots'))
    

def get_dataconf_file(args):
    return args.model_type + '_dataconf.txt'


def test_validation_test_split(dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_train_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, test_indices = indices[:test_split], indices[test_split:]
    train_size = len(train_indices)
    validation_split = int(np.floor((1 - val_train_split) * train_size))
    train_indices, val_indices = train_indices[ : validation_split], train_indices[validation_split:]
    
    return train_indices, val_indices, test_indices


def save_models(net_g, net_d, args, epoch):
    net_g_model_out_path = "./checkpoints/{}/netG_model_epoch_{}.pth".format(args.model_name, epoch)
    net_d_model_out_path = "./checkpoints/{}/netD_model_epoch_{}.pth".format(args.model_name, epoch)        
    torch.save(net_g, net_g_model_out_path)
    torch.save(net_d, net_d_model_out_path)    


parser = argparse.ArgumentParser(description='The training script of the flowPredict pytorch implementation')
parser.add_argument('--data', dest='data_dir', required=True, help='Root directory of the generated data.')
parser.add_argument('--model-name', dest='model_name', default='s_res_8', required=False, help='Name of the current model being trained')
parser.add_argument('--use-pressure', dest='use_pressure', required=False, action='store_true', default=False, help='Should the pressure field images to considered by the models')
parser.add_argument('--model-type', dest='model_type', action='store', default='c', choices=['c', 'vd', 's', 'o'], required=False,
                    help='Type of model to be build. \'c\' - baseline, \'vd\' - fluid viscosity and density, \'s\' - inflow speed, \'o\' - object')
parser.add_argument('--cuda', dest='cuda', action='store_true', default=False, help='Should CUDA be used or not')
parser.add_argument('--threads', dest='threads', type=int, default=4, help='Number of threads for data loader to use')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=4, help='Training batch size.')
parser.add_argument('--test-train-split', dest='test_train_split', type=float, default=0.8, help='The percentage of the data to be used in the training set.')
parser.add_argument('--val-train-split', dest='val_train_split', type=float, default=0.1, help='The percentage of the train data to be used as validation set')
parser.add_argument('--shuffle', dest='shuffle', default=False, action='store_true', help='Should the training and testing data be shufffled.')
parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs for which the model will be trained')
parser.add_argument('--seed', dest='seed', type=int, default=123, help='Random seed to use. Default=123')
parser.add_argument('--niter', type=int, dest='niter', default=100, help='Number of iterations at starting learning rate')
parser.add_argument('--niter_decay', type=int, dest='niter_decay', default=100, help='Number of iterations to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--print-summeries', dest='print_summeries', default=False, action='store_true', help='Print summeries of the genereated networks')
parser.add_argument('--evaluate', default=False, action='store_true', dest='evaluate' , help='Evaluate the trained model at the end')
parser.add_argument('--no-train', default=False, action='store_true', dest='no_train' , help='Do not train the model with the trianing data')
parser.add_argument('--model-path', default=None, action='store', dest='model_path' , help='Optional path to the model\'s weights.')

print('===> Setting up basic structures ')
args = parser.parse_args()

print('--Model name:', args.model_name)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

random_seed = args.seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed) 
if args.cuda:
    torch.cuda.manual_seed(random_seed)

device = torch.device("cuda:0" if args.cuda else "cpu")

model_type = args.model_type
test_train_split = args.test_train_split
batch_size = args.batch_size
shuffle_dataset = args.shuffle
num_epochs = args.epochs
threads = args.threads
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

if not args.no_train: signal.signal(signal.SIGINT, signal_handler)

print('===> Loading datasets')

dataconf_file = get_dataconf_file(args)
dataset = SimulationDataSet(args.data_dir, dataconf_file, args)
train_indices, val_indices, test_indices = test_validation_test_split(dataset, shuffle=shuffle_dataset,
                                                                      test_train_split=args.test_train_split,
                                                                      val_train_split=args.val_train_split)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=threads)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=threads)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=threads)

print('--training samples count:', len(train_indices))
print('--validation samples count:', len(val_indices))
print('--test samples count:', len(test_indices))


print('===> Loading model')

net_g = define_G(config['g_input_nc'], config['g_output_nc'], config['g_nfg'], n_blocks=config['g_layers'], gpu_id=device, args=args).float()
net_d = define_D(config['d_input_nc'], config['d_nfg'], n_layers_D=config['d_layers'], gpu_id=device).float()

optimizer_g = optim.Adam(net_g.parameters(), lr=config['adam_lr'], betas=(config['adam_b1'], config['adam_b2']))
optimizer_d = optim.Adam(net_d.parameters(), lr=config['adam_lr'], betas=(config['adam_b1'], config['adam_b2']))

net_g_scheduler = get_scheduler(optimizer_g, args)
net_d_scheduler = get_scheduler(optimizer_d, args)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

if args.print_summeries:
    print('Generator network:')
    summary(net_g, input_size=(config['g_input_nc'], config['input_width'], config['input_height']))
    print('Detector network:')
    summary(net_d,input_size=(config['d_input_nc'], config['input_width'], config['input_height']))

if not args.no_train:
    print('===> Starting the training loop')

create_directories()

evaluator = Evaluator(args, config['output_dir'], device=device)

train_loader_len = len(train_loader)
losses_path = os.path.join(config['output_dir'], args.model_name, 'losses.txt')
val_losses_path = os.path.join(config['output_dir'], args.model_name, 'val_losses_test.txt')
test_losses_path = os.path.join(config['output_dir'], args.model_name, 'losses_test.txt')

training_started = True
for epoch in range(num_epochs if not args.no_train else 0):
    epoch_loss_d = 0
    epoch_loss_g = 0

    for iteration, batch in enumerate(train_loader, 1):

        net_g.train()
        net_d.train()
        
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ##############################
        # Training the descriminator #
        ##############################
        optimizer_d.zero_grad()
        
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d(real_ab) 
        loss_d_real = criterionGAN(pred_real, True)

        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
        optimizer_d.step()

        ##############################
        #   Training the generator   #
        ##############################
        optimizer_g.zero_grad()

        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
        loss_g_l1 = criterionL1(fake_b, real_b) * config['lambda_L1']

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()
        optimizer_g.step()

        epoch_loss_d += loss_d.item()
        epoch_loss_g += loss_g.item()

        print("> Epoch[{}]({}/{}): Loss_D: {:.5f} Loss_G: {:.5f}".format(
            epoch, iteration, train_loader_len, loss_d.item(), loss_g.item()))

        if STOP_TRAINING:
            print('Saving the model now...')
            save_models(net_g, net_d, args, epoch)
            print('Model saved.')
            sys.exit(0)

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    epoch_loss_d /= train_loader_len
    epoch_loss_g /= train_loader_len

    with open(losses_path, 'w+') as losses_hand:
        losses_hand.write('epoch: {}, gen:{:.5f}, desc:{:.5f}'.format(epoch, epoch_loss_g, epoch_loss_d))
    
    if epoch % 10  == 0:
        save_models(net_g, net_d, args, epoch)        
        print("==> Checkpoint saved to {}".format(os.path.join("checkpoints", args.model_name)))

        avg_psnr = 0
        avg_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                input_img, target = batch[0].to(device), batch[1].to(device)
                prediction = net_g(input_img)
                mse = criterionMSE(prediction, target)
                psnr = 10 * math.log10(1 / mse.item())
                avg_mse += mse
                avg_psnr += psnr
            avg_psnr /= len(val_loader)
            avg_mse /= len(val_loader)
            print("> Val Avg. PSNR: {:.5} dB".format(avg_psnr))

            with open(val_losses_path, 'w+') as losses_hand:
                losses_hand.write('epoch:{}, psnr:{:.5f}, mse:{:.5f}'.format(epoch,avg_psnr,avg_mse))

training_started = False            
            


if args.evaluate:

    print('===> Evaluating model')

    net_g.eval()
    with torch.no_grad():
        
        print('--Evaluating with test set:')
        evaluator.set_output_name('test')
        evaluator.snapshots(net_g, test_sampler, dataset, samples=config['evaluation_snapshots_cnt'])
        evaluator.individual_images_performance(net_g, test_loader)
        evaluator.recusive_application_performance(net_g, dataset, len(train_indices) + len(val_indices) , samples=config['evaluation_recursive_samples'])

        print('--Evaluating with train set:')
        evaluator.set_output_name('train')
        evaluator.snapshots(net_g, train_sampler, dataset, samples=config['evaluation_snapshots_cnt'])
        evaluator.individual_images_performance(net_g, train_loader)
        evaluator.recusive_application_performance(net_g, dataset, 1, samples=config['evaluation_recursive_samples'])
        
