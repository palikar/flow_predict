import os
import json
import sys
import math
import argparse
import random
import time
import itertools

import torch
import torch.nn as nn
import numpy as np

from utils import mkdir
from utils import correlation
from utils import merge_and_save
from utils import RedirectStdStreams
from utils import mkdir
from utils import save_img
from utils import imgs_perc_diff
from config import config
from dataloader import UnNormalize

try:
    from skimage.metrics import structural_similarity as ssim_metr
except ImportError:
    def ssim_metr(*args, **kargs):
        return 0.3



class Evaluator:

    def __init__(self, args, root_dir, MASK, device='cpu', parameterized = False):
        self.args = args
        self.root_dir = root_dir
        self.criterionMSE = nn.MSELoss().to(device)
        self.device = device
        self.parameterized = parameterized

        self.output_name = 'test'
        self.gen_dirs()

        self.MASK = MASK

        if args.rgb:
            if args.use_pressure:
                self.denormalize = UnNormalize([0.5]*9,[0.5]*9)
            else:
                self.denormalize = UnNormalize([0.5]*6,[0.5]*6)
        else:
            if args.use_pressure:
                self.denormalize = UnNormalize([0.5]*3, [0.5]*3)
            else:
                self.denormalize = UnNormalize([0.5]*2, [0.5]*2)


    def set_output_name(self, name):
        self.output_name = name
        self.gen_dirs()


    def gen_dirs(self):
        self.path = os.path.join(config['output_dir'], self.output_name)
        self.path_snaps = os.path.join(config['output_dir'], self.output_name, 'snapshots')
        self.path_full_sim = os.path.join(config['output_dir'], self.output_name, 'full_simulation')
        mkdir(self.path)


    def recusive_application_performance(self, net, dataset, split_point, samples=20):
        print('===> Evaluating performance of recursive application')

        path = os.path.join(config['output_dir'], self.output_name, 'recursive')
        mkdir(path)

        if split_point - samples/2 < 0:
            start_index = 0
            end_index = int(split_point + samples)
        else:
            start_index = int(split_point - samples/2)
            end_index = int(split_point + samples/2)

        print('-- Start index:', start_index)
        print('-- End index:', end_index)

        mse =  []
        cor =  []
        psnr = []
        ssim = []
        diff_avrg = []
        diff_max = []
        diff_x = []
        diff_y = []

        input_img = dataset[start_index][0].expand(1,-1,-1,-1).to(self.device)

        if self.parameterized:
            params = dataset[start_index][2].expand(1,-1,-1,-1).to(self.device)

        for index in range(start_index, end_index):
            
            if self.parameterized:
                predicted = net((input_img, params))
            else:
                predicted = net(input_img)

            for i,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                predicted[i][j] = self.MASK * predicted[i][j]

            input_img = torch.cat((torch.tensor(predicted[0][0:3]).expand(1,-1,-1,-1), self.MASK.expand(1,-1,-1,-1)), 1)


            target = dataset[index][1].expand(1,-1,-1,-1).to(self.device)

            cur_mse = self.criterionMSE(predicted, target).item()

            predicted_img = self.denormalize(predicted).detach().cpu().numpy()
            target_img = self.denormalize(target).detach().cpu().numpy()

            mse += [cur_mse]
            psnr += [10 * math.log10(1 / cur_mse)]
            cor += [np.average(np.array([correlation(predicted_img[i], target_img[i]) for i in range(predicted_img.shape[0])]))]
            ssim += [np.average(np.array([ssim_metr(predicted_img[i].T, target_img[i].T, multichannel=True) for i in range(predicted_img.shape[0])]))]

            diff_avrg_, _, diff_max_ = imgs_perc_diff(target_img, predicted_img)
            diff_avrg.append(diff_avrg_)
            diff_max.append(diff_max_)

            diff_x.append(imgs_perc_diff(target_img[0][0], predicted_img[0][0])[0])
            diff_y.append(imgs_perc_diff(target_img[0][1], predicted_img[0][1])[0])

            if not self.args.use_pressure:
                predicted_x, predicted_y = self._prepare_tensor_img(predicted_img[0])
                target_x, target_y = self._prepare_tensor_img(dataset[index][1])
            else:
                predicted_x, predicted_y, predicted_p = self._prepare_tensor_img(predicted[0])
                target_x, target_y, target_p  = self._prepare_tensor_img(dataset[index][1])


            merge_and_save(target_x, predicted_x,
                           'Real', 'Predicted',
                           os.path.join(path, 'x_recursive_{}.png'.format(index - start_index)))

            merge_and_save(target_y, predicted_y,
                           'Real', 'Predicted',
                           os.path.join(path, 'y_recursive_{}.png'.format(index - start_index)))

            print('> Recursive application {} completed'.format(index - start_index))


        with open(os.path.join(self.root_dir, self.output_name, 'recursive_application.txt'), 'w') as list_hand:
            list_hand.write('Split index: {}\n'.format(str(samples/2)))
            list_hand.write('{} {}\n'.format('mse: ' ,  ','.join(str(i) for i in mse)))
            list_hand.write('{} {}\n'.format('cor: ' ,  ','.join(str(i) for i in cor)))
            list_hand.write('{} {}\n'.format('psnr: ',  ','.join(str(i) for i in psnr)))
            list_hand.write('{} {}\n'.format('ssim: ',  ','.join(str(i) for i in ssim)))
            list_hand.write('{} {}\n'.format('diff_avrg: ',  ','.join(str(i) for i in diff_avrg)))
            list_hand.write('{} {}\n'.format('diff_max: ',  ','.join(str(i) for i in diff_max)))
            list_hand.write('{} {}\n'.format('x_diff_avrg: ',  ','.join(str(i) for i in diff_x)))
            list_hand.write('{} {}\n'.format('y_diff_max: ',  ','.join(str(i) for i in diff_y)))


    def individual_images_performance(self, net, test_dataloader):
        print('===> Evaluating performance on individual images')

        mse = []
        cor = []
        psnr = []
        ssim = []
        diff_avrg = []
        diff_max = []
        diff_x = []
        diff_y = []


        for iteration, batch in enumerate(test_dataloader, 1):
            real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)

            if self.parameterized:
                params = batch[2].to(self.device)
                predicted = net((real_a, params))
            else:
                predicted = net(real_a)

            for i,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                predicted[i][j] = self.MASK * predicted[i][j]

            cur_mse = self.criterionMSE(predicted, real_b).item()

            predicted = self.denormalize(predicted).detach().cpu().numpy()
            real_b = self.denormalize(real_b).detach().cpu().numpy()

            mse += [cur_mse]
            psnr += [10 * math.log10(1 / cur_mse)]
            cor += [np.average(np.array([correlation(predicted[i], real_b[i]) for i in range(predicted.shape[0])]))]
            ssim += [np.average(np.array([ssim_metr(predicted[i].T, real_b[i].T, multichannel=True) for i in range(predicted.shape[0])]))]

            diff_avrg_, _, diff_max_ = imgs_perc_diff(real_b, predicted)
            diff_avrg.append(diff_avrg_)
            diff_max.append(diff_max_)

            for i in range(predicted.shape[0]):
                diff_x.append(imgs_perc_diff(real_b[i][0], predicted[i][0])[0])
                diff_y.append(imgs_perc_diff(real_b[i][1], predicted[i][1])[0])

            if iteration % 10 == 0:
                print('> Evaluation {} completed'.format(iteration))

        mse  = np.array(mse)
        cor  = np.array(cor)
        psnr = np.array(psnr)
        ssim = np.array(ssim)
        diff_avrg = np.array(diff_avrg)
        diff_max = np.array(diff_max)

        with open(os.path.join(self.root_dir, self.output_name, 'metrics_avrg.txt'), 'w') as avrg_hand:
            avrg_hand.write('{} {}\n'.format('Avrg mse: ',  np.average(mse)))
            avrg_hand.write('{} {}\n'.format('Avrg cor: ',  np.average(cor)))
            avrg_hand.write('{} {}\n'.format('Avrg psnr: ', np.average(psnr)))
            avrg_hand.write('{} {}\n'.format('Avrg ssim: ', np.average(ssim)))
            avrg_hand.write('{} {}\n'.format('Avrg avrg_diff_perc: ', np.average(diff_avrg)))
            avrg_hand.write('{} {}\n'.format('Avrg max_diff_perc: ', np.average(diff_max)))
            avrg_hand.write('{} {}\n'.format('Avrg avrt_diff_x: ', np.average(diff_x)))
            avrg_hand.write('{} {}\n'.format('Avrg avrt_diff_y: ', np.average(diff_y)))

            avrg_hand.write('{} {}\n'.format('Var mse: ',  np.var(mse)))
            avrg_hand.write('{} {}\n'.format('Var cor: ',  np.var(cor)))
            avrg_hand.write('{} {}\n'.format('Var psnr: ', np.var(psnr)))
            avrg_hand.write('{} {}\n'.format('Var ssim: ', np.var(ssim)))
            avrg_hand.write('{} {}\n'.format('Var avrg_diff_perc: ', np.var(diff_avrg)))
            avrg_hand.write('{} {}\n'.format('Var max_diff_perc: ', np.var(diff_max)))
            avrg_hand.write('{} {}\n'.format('Var avrt_diff_x: ', np.var(diff_x)))
            avrg_hand.write('{} {}\n'.format('Var avrt_diff_y: ', np.var(diff_y)))

        with open(os.path.join(self.root_dir, self.output_name, 'metrics_list.txt'), 'w') as list_hand:
            list_hand.write('{} {}\n'.format('mse: ' ,  ','.join(str(i) for i in mse)))
            list_hand.write('{} {}\n'.format('cor: ' ,  ','.join(str(i) for i in cor)))
            list_hand.write('{} {}\n'.format('psnr: ',  ','.join(str(i) for i in psnr)))
            list_hand.write('{} {}\n'.format('ssim: ',  ','.join(str(i) for i in ssim)))
            list_hand.write('{} {}\n'.format('diff_avrg: ',  ','.join(str(i) for i in diff_avrg)))
            list_hand.write('{} {}\n'.format('diff_max: ',  ','.join(str(i) for i in diff_max)))


    def _prepare_tensor_img(self, tens_img):
        d = self.denormalize(tens_img).detach().cpu().numpy()

        x = np.transpose(d[:1], (1,2,0))*255

        if self.args.use_pressure:
            y = np.transpose(d[1:2], (1,2,0))*255
            p = np.transpose(d[2:], (1,2,0))*255
            return x, y, p
        else:
            y = np.transpose(d[1:], (1,2,0))*255
            return x, y


    def snapshots(self, net, sampler, dataset, samples=5):
        mkdir(self.path_snaps)

        print('===> Saving {} snapshots'.format(samples))

        for index, i in zip(sampler, range(samples)):

            if self.parameterized:
                input_img, target, params = dataset[index]
                predicted = net((input_img.expand(1,-1,-1,-1).to(self.device),
                                  params.expand(1,-1,-1,-1).to(self.device)))
            else:
                input_img, target = dataset[index]
                predicted = net(input_img.expand(1,-1,-1,-1).to(self.device))

            for i,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                predicted[i][j] = self.MASK * predicted[i][j]


            print('> Snapshot {}'.format(str(i)))

            if not self.args.use_pressure:
                predicted_x, predicted_y = self._prepare_tensor_img(predicted[0])
                input_img_x, input_img_y = self._prepare_tensor_img(input_img)
                target_x, target_y       = self._prepare_tensor_img(target)
            else:
                predicted_x, predicted_y, predicted_p = self._prepare_tensor_img(predicted[0])
                input_img_x, input_img_y, input_img_p = self._prepare_tensor_img(input_img)
                target_x, target_y, target_p          = self._prepare_tensor_img(target)


            merge_and_save(target_x, predicted_x,
                           'Real image_x', 'Predicted image (x)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'x_prediction_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(target_y, predicted_y,
                           'Real image_y', 'Predicted image (y)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'y_prediction_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(input_img_x, predicted_x,
                           'Time step t (x)', 'Time step t+1 (x)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'x_timestep_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(input_img_y, predicted_y,
                           'Time step t (y)', 'Time step t+1 (y)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'y_timestep_{}_{}.png'.format(index, random.randint(0, 10000))))

            if self.args.use_pressure:
                merge_and_save(target_p, predicted_p,
                           'Real image_y', 'Predicted image (p)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'p_prediction_{}_{}.png'.format(index, random.randint(0, 10000))))

                merge_and_save(input_img_p, predicted_p,
                               'Time step t (x)', 'Time step t+1 (p)',
                               os.path.join(config['output_dir'], self.output_name, 'snapshots', 'p_timestep_{}_{}.png'.format(index, random.randint(0, 10000))))


    def run_full_simulation(self, net, dataset, start_index, cnt, sim_name='simulation'):
        print('===> Running simulation with the generator network')

        path = os.path.join(self.path_full_sim, sim_name)
        mkdir(path)

        times = []

        input_img = dataset[start_index][0].expand(1,-1,-1,-1).to(self.device)

        if self.parameterized:
            params = dataset[start_index][2].expand(1,-1,-1,-1).to(self.device)

        for i, index in enumerate(range(start_index, start_index + cnt), 1):

            if self.parameterized:
                t0 = time.time()
                predicted = net((input_img, params))
                t1 = time.time()
            else:
                t0 = time.time()
                predicted = net(input_img)
                t1 = time.time()

            for l,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                predicted[l][j] = self.MASK * predicted[l][j]

            elapsed = int(round(t1*1000 - t0*1000))
            times.append(elapsed)

            if not self.args.use_pressure:
                predicted_x, predicted_y = self._prepare_tensor_img(predicted[0])
            else:
                predicted_x, predicted_y, predicted_p = self._prepare_tensor_img(predicted[0])

            
            save_img(predicted_x, 'x_step_{}'.format(i), '{}/x_step_{}.png'.format(path, i))
            save_img(predicted_y, 'y_step_{}'.format(i), '{}/y_step_{}.png'.format(path, i))        

            if self.args.use_pressure:
                save_img(predicted_p, 'p_step_{}'.format(i), '{}/p_step_{}.png'.format(path, i))

            input_img = torch.cat((torch.tensor(predicted[0][0:3]).expand(1,-1,-1,-1), self.MASK.expand(1,-1,-1,-1)), 1)

        times = np.array(times)

        with open(os.path.join(config['output_dir'], self.output_name, 'full_simulation', 'timing_{}.txt'.format(sim_name)), 'w') as time_hand:
            time_hand.write('{} {}\n'.format('Avrg : ',  np.average(times)))
            time_hand.write('{} {}\n'.format('Var : ',  np.var(times)))
            time_hand.write('{} {}\n'.format('List : ' ,  ','.join(str(i) for i in times)))
