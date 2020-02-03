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
import torchvision.transforms as transforms
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


# http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=95018&copyownerid=127295


class Evaluator:

    def __init__(self, args, root_dir, MASK, device='cpu', parameterized = False):
        self.args = args
        self.root_dir = root_dir
        self.criterionMSE = nn.MSELoss().to(device)
        self.device = device
        self.parameterized = parameterized

        self.output_name = 'test'
        self.gen_dirs()

        if self.args.mask:
            self.MASK = MASK

        if self.args.mask:
            if args.use_pressure:
                self.denormalize_input = UnNormalize([0.5, 0.5, 0.5, 0.0], [0.5, 0.5, 0.5, 1.0])
            else:
                self.denormalize_input = UnNormalize([0.5, 0.5, 0.0], [0.5, 0.5, 1.0])
        else:
            if args.use_pressure:
                self.denormalize_input = UnNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            else:
                self.denormalize_input = UnNormalize([0.5, 0.5], [0.5, 0.5])

        if args.use_pressure:
            self.denormalize_output = UnNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            self.denormalize_output = UnNormalize([0.5, 0.5], [0.5, 0.5])


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

        change_psnr_x = []
        change_psnr_y = []
        change_diff_x = []
        change_diff_y = []
        
        input_img = dataset[start_index][0].expand(1,-1,-1,-1).to(self.device)

        if self.parameterized:
            params = dataset[start_index][2].expand(1,-1,-1,-1).to(self.device)

        for index in range(start_index, end_index):

            pred_input = self._prepare_tensor_img(input_img[0].clone(), is_input=True)
            
            if self.parameterized:
                predicted = net((input_img, params))
            else:
                predicted = net(input_img)

            target = dataset[index][1].expand(1,-1,-1,-1).to(self.device)

            if self.args.mask:
                for i,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                    predicted[i][j] = self.MASK * predicted[i][j]
                    input_img = torch.cat((torch.tensor(predicted.clone().detach()[0][0:3]).expand(1,-1,-1,-1), self.MASK.expand(1,-1,-1,-1)), 1)
            else:
                input_img = predicted.clone().detach()
                

            cur_mse = self.criterionMSE(predicted, target).item()

            
            if not self.args.use_pressure:
                predicted_x, predicted_y = self._prepare_tensor_img(predicted[0])
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


            predicted_img = self.denormalize_output(predicted).detach().cpu().numpy()
            target_img = self.denormalize_input(target).detach().cpu().numpy()

            
            mse += [cur_mse]
            psnr += [10 * math.log10(1 / cur_mse)]
            cor += [np.average(np.array([correlation(predicted_img[i], target_img[i]) for i in range(predicted_img.shape[0])]))]
            ssim += [np.average(np.array([ssim_metr(predicted_img[i].T, target_img[i].T, multichannel=True) for i in range(predicted_img.shape[0])]))]

            diff_avrg_, _, diff_max_ = imgs_perc_diff(target_img, predicted_img)
            diff_avrg.append(diff_avrg_)
            diff_max.append(diff_max_)

            diff_x.append(imgs_perc_diff(target_img[0][0], predicted_img[0][0])[0])
            diff_y.append(imgs_perc_diff(target_img[0][1], predicted_img[0][1])[0])


            real_input = self._prepare_tensor_img(dataset[index][0], True)

            change_x_real = np.abs(      target_x - real_input[0])
            change_x_predicted = np.abs( pred_input[0] - predicted_x)
            change_y_real = np.abs(      target_y - real_input[1])
            change_y_predicted = np.abs( pred_input[1] - predicted_y)
            
            change_mse_x = (np.square(change_x_real - change_x_predicted)).mean(axis=None)
            change_mse_y = (np.square(change_y_real - change_y_predicted)).mean(axis=None)

            change_psnr_x += [10.0 * np.log10(255.0 / np.sqrt(change_mse_x))]
            change_psnr_y += [10.0 * np.log10(255.0 / np.sqrt(change_mse_y))]
            change_diff_x += [imgs_perc_diff(change_x_real, change_x_predicted)[0]]
            change_diff_y += [imgs_perc_diff(change_y_real, change_y_predicted)[0]]

            merge_and_save(change_x_real, change_x_predicted,
                           'Real', 'Predicted',
                           os.path.join(path, 'x_diff_{}.png'.format(index - start_index)))

            merge_and_save(change_y_real, change_y_predicted,
                           'Real', 'Predicted',
                           os.path.join(path, 'y_diff_{}.png'.format(index - start_index)))
        

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

            list_hand.write('{} {}\n'.format('change_psnr_x: ',  ','.join(str(i) for i in change_psnr_x )))
            list_hand.write('{} {}\n'.format('change_psnr_y: ',  ','.join(str(i) for i in change_psnr_y )))
            list_hand.write('{} {}\n'.format('change_diff_x: ',  ','.join(str(i) for i in change_diff_x )))
            list_hand.write('{} {}\n'.format('change_diff_y: ',  ','.join(str(i) for i in change_diff_y )))
            

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


        change_mse_x  = []
        change_mse_y  = []
        change_psnr_x = []
        change_psnr_y = []
        change_diff_x = []
        change_diff_y = []
        change_psnr = []
        
        for iteration, batch in enumerate(test_dataloader, 1):
            real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)

            if self.parameterized:
                params = batch[2].to(self.device)
                predicted = net((real_a, params))
            else:
                predicted = net(real_a)

            if self.args.mask:
                for i,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                    predicted[i][j] = self.MASK * predicted[i][j]

            cur_mse = self.criterionMSE(predicted, real_b).item()

            predicted = self.denormalize_output(predicted).detach().cpu().numpy()
            real_a = self.denormalize_output(real_a).detach().cpu().numpy()
            real_b = self.denormalize_output(real_b).detach().cpu().numpy()

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

            
            # error images

            batch_change_mse  = []
            batch_change_mse_x  = []
            batch_change_mse_y  = []
            batch_change_psnr_x = []
            batch_change_psnr_y = []
            batch_change_diff_x = []
            batch_change_diff_y = []
            
            for ind in range(real_a.shape[0]):

                if self.args.use_pressure:
                    real_change_img = np.abs(real_a[ind][0:3] - real_b[ind])
                else:
                    real_change_img = np.abs(real_a[ind][0:2] - real_b[ind])
                    
                predicted_change_img = np.abs(predicted[ind] - real_b[ind])
                cur_mse = (np.square(real_change_img - predicted_change_img)).mean(axis=None)
                cur_psnr = 10 * np.log10(255.0 / np.sqrt(cur_mse))
                batch_change_mse.append(cur_psnr)
                
                real_change_img_x = np.abs(real_a[ind][0] - real_b[ind][0])
                predicted_change_img_x = np.abs(predicted[ind][0] - real_b[ind][0])

                real_change_img_y = np.abs(real_a[ind][1] - real_b[ind][1])
                predicted_change_img_y = np.abs(predicted[ind][1] - real_b[ind][1])

                x_cur_mse = (np.square(real_change_img_x - predicted_change_img_x)).mean(axis=None)
                y_cur_mse = (np.square(real_change_img_y - predicted_change_img_y)).mean(axis=None)
                x_cur_psnr = 10 * np.log10(255.0 / np.sqrt(x_cur_mse))
                y_cur_psnr = 10 * np.log10(255.0 / np.sqrt(y_cur_mse))
                cur_diff_x, _, _ = imgs_perc_diff(real_change_img_x, predicted_change_img_x)
                cur_diff_y, _, __ = imgs_perc_diff(real_change_img_y, predicted_change_img_y)

                batch_change_mse_x.append(x_cur_mse)
                batch_change_mse_y.append(y_cur_mse)                
                batch_change_psnr_x.append(x_cur_psnr)
                batch_change_psnr_y.append(y_cur_psnr)                
                batch_change_diff_x.append(cur_diff_x)
                batch_change_diff_y.append(cur_diff_y)


            change_psnr.append(np.array(batch_change_mse).mean())
            change_mse_x.append(np.array(batch_change_mse_x ).mean())
            change_mse_y.append(np.array(batch_change_mse_y ).mean())
            change_psnr_x.append(np.array(batch_change_psnr_x).mean())
            change_psnr_y.append(np.array(batch_change_psnr_y).mean())
            change_diff_x.append(np.array(batch_change_diff_x).mean())
            change_diff_y.append(np.array(batch_change_diff_y).mean())

            if iteration % 10 == 0:
                print('> Evaluation {} completed'.format(iteration))

        mse  = np.array(mse)
        cor  = np.array(cor)
        psnr = np.array(psnr)
        ssim = np.array(ssim)
        diff_avrg = np.array(diff_avrg)
        diff_max = np.array(diff_max)

        change_mse_x  = np.array(change_mse_x)
        change_mse_y  = np.array(change_mse_y)
        change_psnr_x = np.array(change_psnr_x)
        change_psnr_y = np.array(change_psnr_y)
        change_diff_x = np.array(change_diff_x)
        change_diff_y = np.array(change_diff_y)

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

            avrg_hand.write('{} {}\n'.format('avrg_Change_mse_x: ',  np.mean(change_psnr)))
            avrg_hand.write('{} {}\n'.format('avrg_Change_mse_x: ',  np.mean(change_mse_x)))
            avrg_hand.write('{} {}\n'.format('avrg_Change_mse_y: ',  np.mean(change_mse_y)))
            avrg_hand.write('{} {}\n'.format('avrg_Change_psnr_x: ',  np.mean(change_psnr_x)))
            avrg_hand.write('{} {}\n'.format('avrg_Change_psnr_y: ',  np.mean(change_psnr_y)))
            avrg_hand.write('{} {}\n'.format('avrg_Change_diff_x: ',  np.mean(change_diff_x)))
            avrg_hand.write('{} {}\n'.format('avrg_Change_diff_y: ',  np.mean(change_diff_y)))

        with open(os.path.join(self.root_dir, self.output_name, 'metrics_list.txt'), 'w') as list_hand:
            list_hand.write('{} {}\n'.format('mse: ' ,  ','.join(str(i) for i in mse)))
            list_hand.write('{} {}\n'.format('cor: ' ,  ','.join(str(i) for i in cor)))
            list_hand.write('{} {}\n'.format('psnr: ',  ','.join(str(i) for i in psnr)))
            list_hand.write('{} {}\n'.format('ssim: ',  ','.join(str(i) for i in ssim)))
            list_hand.write('{} {}\n'.format('diff_avrg: ',  ','.join(str(i) for i in diff_avrg)))
            list_hand.write('{} {}\n'.format('diff_max: ',  ','.join(str(i) for i in diff_max)))


    def _prepare_tensor_img(self, tens_img, is_input=False):

        if is_input:
            d = self.denormalize_input(tens_img).detach().cpu().numpy()
        else:
            d = self.denormalize_output(tens_img).detach().cpu().numpy()

        

        x = np.transpose(d[:1], (1,2,0))*255
                

        if self.args.mask and is_input:
            if self.args.use_pressure:
                y = np.transpose(d[1:2], (1,2,0))*255
                p = np.transpose(d[2:3], (1,2,0))*255
                return x, y, p
            else:
                y = np.transpose(d[1:2], (1,2,0))*255
                return x, y
        else:
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

            if self.args.mask:
                for l,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                    predicted[l][j] = self.MASK * predicted[l][j]


            
            print('> Snapshot {}'.format(str(i)))

            if not self.args.use_pressure:
                predicted_x, predicted_y = self._prepare_tensor_img(predicted[0])
                input_img_x, input_img_y = self._prepare_tensor_img(input_img, is_input=True)
                target_x, target_y       = self._prepare_tensor_img(target)
            else:
                predicted_x, predicted_y, predicted_p = self._prepare_tensor_img(predicted[0])
                input_img_x, input_img_y, input_img_p = self._prepare_tensor_img(input_img, is_input=True)
                target_x, target_y, target_p          = self._prepare_tensor_img(target)


            diff_x_real = np.abs(input_img_x - target_x)
            diff_y_real = np.abs(input_img_y - target_y)

            diff_x_predicted = np.abs(input_img_x - predicted_x)
            diff_y_predicted = np.abs(input_img_y - predicted_y)

            merge_and_save(target_x, predicted_x,
                           'Real image (x)', 'Predicted image (x)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'x_prediction_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(target_y, predicted_y,
                           'Real image (y)', 'Predicted image (y)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'y_prediction_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(input_img_x, predicted_x,
                           'Time step t (x)', 'Time step t+1 (x)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'x_timestep_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(input_img_y, predicted_y,
                           'Time step t (y)', 'Time step t+1 (y)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'y_timestep_{}_{}.png'.format(index, random.randint(0, 10000))))


            merge_and_save(diff_x_real, diff_x_predicted,
                           'Real difference (x)', 'Predicted difference (x)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'x_difference_{}_{}.png'.format(index, random.randint(0, 10000))))

            merge_and_save(diff_y_real, diff_y_predicted,
                           'Real difference (y)', 'Predicted difference (y)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'y_difference_{}_{}.png'.format(index, random.randint(0, 10000))))
            

            if self.args.use_pressure:
                merge_and_save(target_p, predicted_p,
                           'Real image_y', 'Predicted image (p)',
                           os.path.join(config['output_dir'], self.output_name, 'snapshots', 'p_prediction_{}_{}.png'.format(index, random.randint(0, 10000))))

                merge_and_save(input_img_p, predicted_p,
                               'Time step t (x)', 'Time step t+1 (p)',
                               os.path.join(config['output_dir'], self.output_name, 'snapshots', 'p_timestep_{}_{}.png'.format(index, random.randint(0, 10000))))


    def run_full_simulation(self, net, dataset, start_index, cnt, sim_name='simulation', saving_imgs=False):
        print('===> Running simulation with the generator network')

        path = os.path.join(self.path_full_sim, sim_name)
        mkdir(path)

        times = []

        input_img = dataset[start_index][0].expand(1,-1,-1,-1).to(self.device)

        if self.parameterized:
            params = dataset[start_index][2].expand(1,-1,-1,-1).to(self.device)

        for i, index in enumerate(range(start_index, start_index + cnt), 1):

            pred_output = self._prepare_tensor_img(input_img[0].clone(), is_input=True)

            
            if self.parameterized:
                t0 = time.time()
                predicted = net((input_img, params))
                t1 = time.time()
            else:
                t0 = time.time()
                print(input_img.shape)
                predicted = net(input_img)
                t1 = time.time()

            if self.args.mask:
                for l,j in itertools.product(range(predicted.shape[0]), range(predicted.shape[1])):
                    predicted[l][j] = self.MASK * predicted[l][j]

            if self.args.mask:
                input_img = torch.cat((torch.tensor(predicted.clone().detach()[0][0:3]).expand(1,-1,-1,-1), self.MASK.expand(1,-1,-1,-1)), 1)
            else:
                input_img = predicted.detach().clone().detach()

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


            real_input = self._prepare_tensor_img(dataset[index][0], True)
            real_output = self._prepare_tensor_img(dataset[index][1])

            
            diff_x_real = np.abs(      real_output[0] - real_input[0])
            diff_x_predicted = np.abs( pred_output[0] - predicted_x)

            diff_y_real = np.abs(      real_output[1] - real_input[1])
            diff_y_predicted = np.abs( pred_output[1] - predicted_y)

            

            merge_and_save(diff_x_real, diff_x_predicted,
                           'Real difference_x {}'.format(i), 'Predicted difference_x {}'.format(i),
                           os.path.join(path, 'diff_x_step_{}.png'.format(i)), txt_color=(255, 255,255,255))

            merge_and_save(diff_y_real, diff_y_predicted,
                           'Real difference_y {}'.format(i), 'Predicted difference_y {}'.format(i),
                           os.path.join(path, 'diff_y_step_{}.png'.format(i)), txt_color=(255, 255,255,255))
            

        times = np.array(times)

        with open(os.path.join(config['output_dir'], self.output_name, 'full_simulation', 'timing_{}.txt'.format(sim_name)), 'w') as time_hand:
            time_hand.write('{} {}\n'.format('Avrg : ',  np.average(times)))
            time_hand.write('{} {}\n'.format('Var : ',  np.var(times)))
            time_hand.write('{} {}\n'.format('List : ' ,  ','.join(str(i) for i in times)))
