#!/usr/bin/env python

import os
import json
import sys
import math
import argparse
import random
import signal
import datetime
import itertools

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

from torchsummary import summary

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


training_started = False
STOP_TRAINING = False

s_test_indices = [i for i in range(669, 1003)] + [i for i in range(2339, 2673)] + [i for i in range(4009, 4343)] + [i for i in range(5077, 5411)] + [i for i in range(6747, 7081)] + [i for i in range(8417, 8751)] + [i for i in range(10087, 10421)] + [i for i in range(11757, 12091)] + [i for i in range(12425, 12759)] 

vd_test_indices = [i for i in range(322, 643)] + [i for i in range(1927, 2248)] + [i for i in range(3532, 3853)] + [i for i in range(5137, 5458)] + [i for i in range(6742, 7063)] + [i for i in range(8347, 8668)] + [i for i in range(9952, 10273)] + [i for i in range(11557, 11878)] + [i for i in range(13162, 13483)] + [i for i in range(14767, 15088)] + [i for i in range(16372, 16693)] + [i for i in range(17977, 18298)] + [i for i in range(19582, 19903)] + [i for i in range(21187, 21508)] + [i for i in range(22792, 23113)] + [i for i in range(24397, 24718)] + [i for i in range(26002, 26323)] + [i for i in range(27607, 27928)] + [i for i in range(29212, 29533)] + [i for i in range(30817, 31138)] + [i for i in range(32422, 32743)] + [i for i in range(34027, 34348)] + [i for i in range(35632, 35953)] + [i for i in range(37237, 37558)] + [i for i in range(38842, 39163)] + [i for i in range(40447, 40768)] + [i for i in range(42052, 42373)] + [i for i in range(43657, 43978)] + [i for i in range(45262, 45583)] + [i for i in range(46867, 47188)] + [i for i in range(48472, 48793)] + [i for i in range(50077, 50398)] + [i for i in range(51682, 52003)] + [i for i in range(53287, 53608)] + [i for i in range(54892, 55213)] + [i for i in range(56497, 56818)] + [i for i in range(58102, 58423)] + [i for i in range(59707, 60028)] + [i for i in range(61312, 61633)] + [i for i in range(62917, 63238)] + [i for i in range(64522, 64843)] + [i for i in range(66127, 66448)] + [i for i in range(67732, 68053)] + [i for i in range(69337, 69658)] + [i for i in range(70942, 71165)] + [i for i in range(72449, 72770)] + [i for i in range(74054, 74375)] + [i for i in range(75659, 75980)] + [i for i in range(77264, 77585)] + [i for i in range(78869, 79190)] + [i for i in range(80474, 80795)] + [i for i in range(82079, 82400)] + [i for i in range(83684, 84005)] + [i for i in range(85289, 85610)] + [i for i in range(86894, 87215)] + [i for i in range(88499, 88820)] + [i for i in range(90104, 90425)] + [i for i in range(91709, 92030)] + [i for i in range(93314, 93635)] + [i for i in range(94919, 95240)] + [i for i in range(96524, 96845)] + [i for i in range(98129, 98450)] + [i for i in range(99734, 100055)] + [i for i in range(101339, 101660)] + [i for i in range(102944, 103265)]



def signal_handler(sig, frame):
    if not training_started:
        sys.exit(0)

    print('Stoping the training...')
    global STOP_TRAINING
    STOP_TRAINING = True


def create_directories():
    mkdir(config['output_dir'])
    mkdir(os.path.join(config['output_dir'], args.model_name))


def get_dataconf_file(args):
    return args.model_type + '_dataconf.txt'


def test_validation_test_split(dataset, test_train_split=0.8, val_train_split=0.1, shuffle=False, model = 'c'):

    if model == 'c':
        print('Splitting for c')
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

    if model == 's':
        print('Splitting for s')
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        test_indices = s_test_indices
        train_indices = list(set(indices) - set(s_test_indices))
        if shuffle:
            np.random.shuffle(train_indices)
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))
        train_indices, val_indices = train_indices[ : validation_split], train_indices[validation_split:]
        return train_indices, val_indices, test_indices

    if model == 'vd':
        print('Splitting for vd')
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        test_indices = vd_test_indices
        train_indices = list(set(indices) - set(s_test_indices))
        if shuffle:
            np.random.shuffle(train_indices)
        train_size = len(train_indices)
        validation_split = int(np.floor((1 - val_train_split) * train_size))
        train_indices, val_indices = train_indices[ : validation_split], train_indices[validation_split:]
        return train_indices, val_indices, test_indices


def save_models(net_g, net_d, args, epoch):
    net_g_model_out_path = "./{0}/{1}/netG_{1}_model_epoch_{2}.pth".format(config['output_dir'], args.model_name, epoch)
    net_d_model_out_path = "./{0}/{1}/netD_{1}_model_epoch_{2}.pth".format(config['output_dir'], args.model_name, epoch)
    torch.save(net_g, net_g_model_out_path)
    # torch.save(net_d, net_d_model_out_path)


parser = argparse.ArgumentParser(description='The training script of the flowPredict pytorch implementation')
parser.add_argument('--data', dest='data_dir', required=True, help='Root directory of the generated data.')
parser.add_argument('--model-name', dest='model_name', default='s_res_8', required=False, help='Name of the current model being trained. res or unet')
parser.add_argument('--use-pressure', dest='use_pressure', required=False, action='store_true', default=False, help='Should the pressure field images to considered by the models')
parser.add_argument('--rgb', dest='rgb', required=False, action='store_true', default=False, help='Do we use RGB images or just the grayscale')
parser.add_argument('--model-type', dest='model_type', action='store', default='c', choices=['c', 'vd', 's', 'o'], required=False,
                    help='Type of model to be build. \'c\' - baseline, \'vd\' - fluid viscosity and density, \'s\' - inflow speed, \'o\' - object')
parser.add_argument('--cuda', dest='cuda', action='store_true', default=False, help='Should CUDA be used or not')
parser.add_argument('--threads', dest='threads', type=int, default=4, help='Number of threads for data loader to use')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=4, help='Training batch size.')
parser.add_argument('--test-train-split', dest='test_train_split', type=float, default=0.8, help='The percentage of the data to be used in the training set.')
parser.add_argument('--val-train-split', dest='val_train_split', type=float, default=0.1, help='The percentage of the train data to be used as validation set')
parser.add_argument('--shuffle', dest='shuffle', default=False, action='store_true', help='Should the training and testing data be shufffled.')
parser.add_argument('--seed', dest='seed', type=int, default=123, help='Random seed to use. Default=123')
parser.add_argument('--niter', type=int, dest='niter', default=100, help='Number of iterations at starting learning rate')
parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs for which the model will be trained')
parser.add_argument('--niter_decay', type=int, dest='niter_decay', default=100, help='Number of iterations to linearly decay learning rate to zero')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--print-summeries', dest='print_summeries', default=False, action='store_true', help='Print summeries of the genereated networks')
parser.add_argument('--evaluate', default=False, action='store_true', dest='evaluate' , help='Evaluate the trained model at the end')
parser.add_argument('--no-train', default=False, action='store_true', dest='no_train' , help='Do not train the model with the trianing data')
parser.add_argument('--model-path', default=None, action='store', dest='model_path' , help='Optional path to the model\'s weights.')
parser.add_argument('--g_nfg', type=int, dest='g_nfg', default=-1, help='Number of feature maps in the first layers of ResNet')
parser.add_argument('--g_layers', type=int, dest='g_layers', default=-1, help='ResNet blocks in the middle of the network')
parser.add_argument('--g_output_nc', type=int, dest='g_output_nc', default=-1, help='Number of output channels of the genrator network')
parser.add_argument('--g_input_nc', type=int, dest='g_input_nc', default=-1, help='Number of input channels of the genrator network')
parser.add_argument('--output-dir', dest='output_dir', default=None, help='The output directory for the model files')
parser.add_argument('--no-mask', dest='mask', default=True, action='store_false', help='Disable the mask for the model')
parser.add_argument('--no-noise', dest='noise', default=True, action='store_false', help='Use noise for the input')
parser.add_argument('--no-crops', dest='crops', default=True, action='store_false', help='Use crops for the input')


args = parser.parse_args()

parameterized = args.model_type == 'vd' or args.model_type == 's'

config['g_input_nc'] = 6 if args.rgb else 2
config['g_output_nc'] = 6 if args.rgb else 2

if args.use_pressure:
    config['g_output_nc'] += 3 if args.rgb else 1
    config['g_input_nc'] += 3 if args.rgb else 1

if args.mask:
    config['d_input_nc'] = 2*config['g_input_nc'] + 1
else:
    config['d_input_nc'] = 2*config['g_input_nc']

if parameterized:
    config['g_input_nc'] += 1 if args.model_type == 's' else 2 if args.model_type == 'vd' else 0

if args.mask: config['g_input_nc'] += 1


if args.g_layers != -1: config['g_layers'] = args.g_layers
if args.g_nfg != -1: config['g_nfg'] = args.g_nfg
if args.g_input_nc != -1: config['g_input_nc'] = args.g_input_nc
if args.g_output_nc != -1: config['g_output_nc'] = args.g_output_nc
if args.output_dir is not None: config['output_dir'] = args.output_dir





args.model_name = '{}_{}_l{}_ngf{}'.format(args.model_type, args.model_name, config['g_layers'], config['g_nfg'])
if args.use_pressure:
    args.model_name = '{}_p'.format(args.model_name)
if args.mask:
    args.model_name = '{}_m'.format(args.model_name)
if args.noise:
    args.model_name = '{}_n'.format(args.model_name)
if args.crops:
    args.model_name = '{}_c'.format(args.model_name)

create_directories()

sys.stdout = Logger(os.path.join(config['output_dir'], 'log.txt'))

print('===> Setting up basic structures ')


if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

random_seed = args.seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

if args.cuda:
    torch.cuda.manual_seed(random_seed)

device = torch.device("cuda:0" if args.cuda else "cpu")
# if torch.cuda.device_count() > 1:
#     device = torch.device("cuda:0")

if not args.no_train: signal.signal(signal.SIGINT, signal_handler)

model_type = args.model_type
test_train_split = args.test_train_split
val_train_split = args.val_train_split
batch_size = args.batch_size
shuffle_dataset = args.shuffle
num_epochs = args.epochs
threads = args.threads
model_name = args.model_name


print('--Model name:', args.model_name)
print('--model type:', model_type)
print('--use pressure:', args.use_pressure)
print('--test-train split:', test_train_split)
print('--val-train split:', val_train_split)
print('--batch size:', batch_size)
print('--shuffle dataset:', shuffle_dataset)
print('--num epochs:', num_epochs)
print('--learning rate policiy:', args.lr_policy)
print('--random seed:', random_seed)
print('--worker threads:', threads)
print('--cuda:', args.cuda)
print('--device:', device)
print('--gen. input channels:', config['g_input_nc'])
print('--gen. output channels:', config['g_output_nc'])
print('--desc. input channels:', config['d_input_nc'])
print('--mask:', args.mask)


print('===> Loading datasets')

dataconf_file = get_dataconf_file(args)
dataset = SimulationDataSet(args.data_dir, dataconf_file, args)

train_indices, val_indices, test_indices = test_validation_test_split(dataset,
                                                                      test_train_split=args.test_train_split,
                                                                      val_train_split=args.val_train_split,
                                                                      shuffle=shuffle_dataset,
                                                                      model=model_type)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=threads)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=threads)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=threads)


print('--training samples count:', len(train_indices))
print('--validation samples count:', len(val_indices))
print('--test samples count:', len(test_indices))

now = datetime.datetime.now()
date = now.strftime("%d-%m-%Y:%H:%M:%S")
print('--date:', date)

with open(os.path.join(config['output_dir'], 'date_{}'.format(date)), 'w') as dh:
    dh.write(date)

print('===> Loading model')

net_g = define_G(config['g_input_nc'], config['g_output_nc'], config['g_nfg'], n_blocks=config['g_layers'], gpu_id=device, args=args).float()
net_d = define_D(config['d_input_nc'], config['d_nfg'], n_layers_D=config['d_layers'], gpu_id=device).float()

# if torch.cuda.device_count() > 1:
#   print("--using", torch.cuda.device_count(), "GPUs")
#   net_g = nn.DataParallel(net_g)
#   net_d = nn.DataParallel(net_d)
# net_d.to(device)
# net_g.to(device)


optimizer_g = optim.Adam(net_g.parameters(), lr=config['adam_lr'], betas=(config['adam_b1'], config['adam_b2']))
optimizer_d = optim.Adam(net_d.parameters(), lr=config['adam_lr'], betas=(config['adam_b1'], config['adam_b2']))

net_g_scheduler = get_scheduler(optimizer_g, args)
net_d_scheduler = get_scheduler(optimizer_d, args)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# if args.print_summeries:
#     print('===> Generator network:')

#     if args.model_type == 's':
#         summary(net_g, [(config['g_input_nc'] - 1, config['input_width'], config['input_height']), (1, 1, 1)], batch_size=2, device='cuda')
#     elif args.model_type == 'vd':
#         summary(net_g, [(config['g_input_nc'] - 2, config['input_width'], config['input_height']), (1, 1, 2)])
#     else:
#         summary(net_g, (config['g_input_nc'], config['input_width'], config['input_height']))
#     # net_g.to(device)

#     print('===> Detector network:')
#     summary(net_d, (config['d_input_nc'], config['input_width'], config['input_height']))



train_loader_len = len(train_loader)
losses_path = os.path.join(config['output_dir'], 'losses.txt')
val_losses_path = os.path.join(config['output_dir'], 'val_losses_test.txt')

if not args.no_train:
    print('===> Starting the training loop')

if args.mask:
    MASK = dataset.get_mask().to(device)

training_started = True
dataset.test()
for epoch in range(num_epochs if not args.no_train else 0):
    epoch_loss_d = 0
    epoch_loss_g = 0

    iteration = 1
    for batch in train_loader:
        net_g.train()
        net_d.train()

        real_a, real_b = batch[0].to(device), batch[1].to(device)

        if parameterized:
            params = batch[2].to(device)
            fake_b = net_g((real_a, params))
        else:
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
        iteration += 1

        if STOP_TRAINING:
            print('> Saving the model now...')
            save_models(net_g, net_d, args, epoch)
            print('> Model saved.')
            sys.exit(0)

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    epoch_loss_d /= train_loader_len
    epoch_loss_g /= train_loader_len

    with open(losses_path, 'a') as losses_hand:
        losses_hand.write('epoch: {}, gen:{:.5f}, desc:{:.5f}\n'.format(epoch, epoch_loss_g, epoch_loss_d))

    if epoch % 20  == 0:
        save_models(net_g, net_d, args, epoch)
        print("> Checkpoint saved to {}".format(os.path.join("checkpoints", args.model_name)))

    if epoch % 5  == 0:
        avg_psnr = 0
        avg_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                input_img, target = batch[0].to(device), batch[1].to(device)
                if parameterized:
                    params = batch[2].to(device)
                    prediction = net_g((input_img, params))
                else:
                    prediction = net_g(input_img)

                if args.mask:
                    for i,j in itertools.product(range(prediction.shape[0]), range(prediction.shape[1])):
                        prediction[i][j] = MASK * prediction[i][j]

                mse = criterionMSE(prediction, target)
                psnr = 10 * math.log10(1 / mse.item())
                avg_mse += mse
                avg_psnr += psnr
            avg_psnr /= len(val_loader)
            avg_mse /= len(val_loader)

            print("> Val Avg. PSNR: {:.5} dB".format(avg_psnr))
            with open(val_losses_path, 'a') as losses_hand:
                losses_hand.write('epoch:{}, psnr:{:.5f}, mse:{:.5f}\n'.format(epoch, avg_psnr, avg_mse))


save_models(net_g, net_d, args, num_epochs)
print("> Checkpoint saved to {}".format(os.path.join("checkpoints", args.model_name)))
training_started = False

evaluator = Evaluator(args, config['output_dir'], MASK if args.mask else None, device=device, parameterized = parameterized)
if args.evaluate:
    print('===> Evaluating model')

    net_g.eval()

    dataset.test()
    
    with torch.no_grad():

        print('===> Evaluating with test set:')
        evaluator.set_output_name('test')
        evaluator.snapshots(net_g, test_sampler, dataset, samples=config['evaluation_snapshots_cnt'])
        evaluator.individual_images_performance(net_g, test_loader)
        

        print('===> Evaluating with train set:')
        evaluator.set_output_name('train')
        evaluator.snapshots(net_g, train_sampler, dataset, samples=config['evaluation_snapshots_cnt'])
        evaluator.individual_images_performance(net_g, train_loader)
        

        if args.model_type == 'c':
            print('===> Running simulations for c:')

            evaluator.set_output_name('simulations')
            evaluator.run_full_simulation(net_g, dataset, 20, config['full_simulaiton_samples'], sim_name = 'simulation_i{}'.format(20))
            evaluator.run_full_simulation(net_g, dataset, 100, config['full_simulaiton_samples'], sim_name = 'simulation_i{}'.format(100))
            evaluator.run_full_simulation(net_g, dataset, 200, config['full_simulaiton_samples'], sim_name = 'simulation_i{}'.format(200))
            evaluator.run_full_simulation(net_g, dataset, 300, config['full_simulaiton_samples'], sim_name = 'simulation_i{}'.format(300))

            evaluator.run_full_simulation(net_g, dataset, 100, 300, sim_name = 'simulation_timings', saving_imgs=False)
        
            print('===> Evaluating recursively:')

            evaluator.set_output_name('recursive_i20')
            evaluator.recusive_application_performance(net_g, dataset, 20, samples=config['evaluation_recursive_samples'])
            
            evaluator.set_output_name('recursive_i100')
            evaluator.recusive_application_performance(net_g, dataset, 20, samples=config['evaluation_recursive_samples'])

            evaluator.set_output_name('recursive_i200')
            evaluator.recusive_application_performance(net_g, dataset, 200, samples=config['evaluation_recursive_samples'])

            evaluator.set_output_name('recursive_i300')
            evaluator.recusive_application_performance(net_g, dataset, 300, samples=config['evaluation_recursive_samples'])

        if args.model_type == 's':
            print('===> Running simulations for s:')

            evaluator.set_output_name('simulations')
            evaluator.run_full_simulation(net_g, dataset, 100, 300, sim_name = 'simulation_timings', saving_imgs=False)
            indices = [(1, 's'), (101, 's'), (1436, 's'), (3106, 's'), (6178, 's'), (9518, 's'), (11188, 's'), (12858, 's')]
            for start_index in indices:
                print(start_index)
                evaluator.run_full_simulation(net_g, dataset, start_index[0],
                                              config['full_simulaiton_samples'],
                                              sim_name = 'simulation_{}_i{}'.format(start_index[1], start_index[0]))



            print('===> Evaluating recursively:')

            indices = [(1, 's'), (101, 's'), (1436, 's'), (3106, 's'), (6178, 's'), (9518, 's'), (11188, 's'), (12858, 's')]
            for start_index in indices:
                print(start_index)
                evaluator.set_output_name('recursive_{}_i{}'.format(start_index[1], start_index[0]))
                evaluator.recusive_application_performance(net_g, dataset, start_index[0], samples=config['evaluation_recursive_samples'])


        if args.model_type == 'vd':
            print('===> Running simulations for vd:')
            
            indices = [(7705 + 50, 'vd_50'),
                       (17335 + 150, 'vd_150'),
                       (23755 + 10, 'vd_10'),
                       (31780 + 250, 'vd_250'),
                       (39805 + 2, 'vd_2'),
                       (55855 + 150, 'vd'),
                       (63880 + 150, 'vd'),
                       (71807 + 150, 'vd'),
                       (79832 + 150, 'vd'),
                       (87857 + 150, 'vd'),
                       (95882 + 150, 'vd'),
                       (99092 + 150, 'vd')]
            
            evaluator.set_output_name('simulations')

            evaluator.run_full_simulation(net_g, dataset, 100, 300, sim_name = 'simulation_timings', saving_imgs=False)
            for start_index in indices:
                print(start_index)
                evaluator.run_full_simulation(net_g, dataset, start_index[0],
                                              config['full_simulaiton_samples'],
                                              sim_name = 'simulation_{}_i{}'.format(start_index[1], start_index[0]))
            
            
            print('===> Evaluating recursively:')

            for start_index in indices:
                print(start_index)
                evaluator.set_output_name('recursive_{}_i{}'.format(start_index[1], start_index[0]))
                evaluator.recusive_application_performance(net_g, dataset, start_index[0], samples=config['evaluation_recursive_samples'])


            
