import os

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader

from utils import load_img
from config import config



class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class SimulationDataSet(data.Dataset):

    def __init__(self, root_dir, data_file, args):
        
        data_file = os.path.join(root_dir, data_file)
        print('--Data conf file: ', data_file)
        if not os.path.isfile(data_file):
            print('The data config file is not in the root data directory')
            exit(1)

        self.args = args
        
        self.paths_a = []
        self.paths_b = []

        self.densities = []
        self.viscosities = []
        self.speeds = []

        self.train = True

        self.size = (config['input_width'], config['input_height'])
        
        self.root_dir = root_dir

        if args.mask:
            self.mask_path = os.path.join(root_dir, 'mask.png')
            self.mask_img = load_img(self.mask_path).astype(np.float)
        
        with open(data_file, 'r') as handle:
            first_line = handle.readline().rstrip('\n')
            self.data_mode = first_line

            if first_line == "plain":
                print('--Loading \'plain\' data')
                self.return_func = self._return_plain
                self.handle_func = lambda x, y: None

            elif first_line == "fluid":
                print('--Loading \'fluid\' data')
                self.return_func = self._return_fluid
                self.handle_func = self._handle_fluid_parts

            elif first_line == "speed":
                print('--Loading \'speed\' data')
                self.return_func = self._return_speed
                self.handle_func = self._handle_speed_parts

            if self.args.use_pressure:
                print('--Loading pressure images')
            else:
                print('--Not loading pressure images')
                
            self.read_rest(handle, first_line)


        pre_image_transorms_list = [
            
        ]

        self.pre_image_transorms = transforms.Compose(pre_image_transorms_list)
        

        image_transorms_list = [
            transforms.Lambda(lambda x: x + 0.22*torch.rand_like(x)),
        ]

        self.image_transorms = transforms.Compose(image_transorms_list)
        
        transform_list_a = [transforms.ToTensor()]
        transform_list_b = [transforms.ToTensor()]

        if args.mask:
            
            if args.use_pressure:
                transform_list_a.append(transforms.Normalize((0.5, 0.5, 0.5, 0.0), (0.5, 0.5, 0.5, 1.0)))
                transform_list_b.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                transform_list_a.append(transforms.Normalize((0.5, 0.5, 0.0),(0.5, 0.5, 1.0)))
                transform_list_b.append(transforms.Normalize((0.5, 0.5), (0.5, 0.5)))
        else:
            if args.use_pressure:
                transform_list_a.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
                transform_list_b.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
            else:
                transform_list_a.append(transforms.Normalize((0.5, 0.5),(0.5, 0.5)))
                transform_list_b.append(transforms.Normalize((0.5, 0.5),(0.5, 0.5)))
            
            
            
        self.transform_a = transforms.Compose(transform_list_a)
        self.transform_b = transforms.Compose(transform_list_b)


    def _return_plain(self, a, b, index):
        return a, b


    def _return_fluid(self, a, b, index):
        return a, b, torch.tensor([self.densities[index], self.viscosities[index]]).view(1, 1, 2)


    def _return_speed(self, a, b, index):
        return a, b, torch.tensor([self.speeds[index]]).view(1, 1, 1)


    def read_rest(self, handle, first_line):
        line = handle.readline().rstrip('\n').replace(" ", "")
        while line:
            line_parts = line.split(',')

            if self.args.use_pressure:
                self.paths_a.append((line_parts[0], line_parts[1], line_parts[2]))
                self.paths_b.append((line_parts[3], line_parts[4], line_parts[5]))
            else:
                self.paths_a.append((line_parts[0], line_parts[1]))
                self.paths_b.append((line_parts[3], line_parts[4]))

            if first_line == "fluid":
                self._handle_fluid_parts(line_parts)
            elif first_line == "speed":
                self._handle_speed_parts(line_parts)
                
            line = handle.readline().rstrip('\n').replace(" ", "")


    def _handle_fluid_parts(self, parts):
        self.densities.append(float(parts[6]))
        self.viscosities.append(float(parts[7]))


    def _handle_speed_parts(self, parts):
        self.speeds.append(float(parts[6]))


    def train(self):
        self.train = True

    def test(self):
        self.train = False

        
    def __getitem__(self, index):
        a_path = self.paths_a[index]
        b_path = self.paths_b[index]

        a_x = load_img(os.path.join(self.root_dir, a_path[0]), size=self.size)
        a_y = load_img(os.path.join(self.root_dir, a_path[1]), size=self.size)

        b_x = load_img(os.path.join(self.root_dir, b_path[0]), size=self.size)
        b_y = load_img(os.path.join(self.root_dir, b_path[1]), size=self.size)

        if self.args.use_pressure:
            b_p = load_img(os.path.join(self.root_dir, b_path[2]), size=self.size)
            a_p = load_img(os.path.join(self.root_dir, a_path[2]), size=self.size)


        if self.args.mask: mask = self.mask_img

        if self.args.use_pressure:
            # if self.args.mask:
            #     a = np.concatenate((a_x, a_y, a_p), axis=2)
            # else:
            a = np.concatenate((a_x, a_y, a_p), axis=2)

            b = np.concatenate((b_x, b_y, b_p), axis=2)
        else:
            # if self.args.mask:
            #     a = np.concatenate([a_x, a_y], axis=2)
            # else:
            a = np.concatenate([a_x, a_y], axis=2)                

            b = np.concatenate([b_x, b_y], axis=2)

        if self.train and self.args.crops:
            h, w, _ = a.shape
            th, tw = 128, 512
            if not (w == tw and h == th):
                i = np.random.randint(0, h - th)
                j = np.random.randint(0, w - tw)
                iO = i + th
                jO = j + tw                
                a = a[i:iO, j:jO, :]
                b = b[i:iO, j:jO, :]

                
        
        a = self.transform_b(a).float()
        b = self.transform_b(b).float()

        if self.train and self.args.noise:
            a = self.image_transorms(a).float()
        

        if self.args.mask:
            a = torch.cat((a, self.get_mask()), axis = 0)

        if self.args.mask:
            for i in range(a.shape[0]):
                a[i] = self.get_mask() * a[i]
            for i in range(b.shape[0]):
                b[i] = self.get_mask() * b[i]


        return self.return_func(a, b, index)


    def __len__(self):
        return len(self.paths_a)


    def get_mask(self):
        return transforms.ToTensor()(self.mask_img).float()
        
        
