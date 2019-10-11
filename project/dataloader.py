import os

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import load_img

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

        self.root_dir = root_dir
        
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

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)


    def _return_plain(self, a, b, index):
        return a, b

    def _return_fluid(self, a, b, index):
        return a, b, self.densities[index], self.viscosities[index]

    def _return_speed(self, a, b, index):
        return a, b, self.speeds[index]


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
        self.densities.append(parts[7])
        self.viscosities.append(parts[8])


    def _handle_speed_parts(self, parts):
        self.speeds.append(parts[7])


    def __getitem__(self, index):

        a_path = self.paths_a[index]
        b_path = self.paths_b[index]

        a_x = self.transform(load_img(os.path.join(self.root_dir, a_path[0])))
        a_y = self.transform(load_img(os.path.join(self.root_dir, a_path[1])))

        b_x = self.transform(load_img(os.path.join(self.root_dir, b_path[0])))
        b_y = self.transform(load_img(os.path.join(self.root_dir, b_path[1])))


        if self.args.use_pressure:
            b_p = self.transform(load_img(os.path.join(self.root_dir, b_path[2])))
            a_p = self.transform(load_img(os.path.join(self.root_dir, a_path[2])))

        
        if self.args.use_pressure:
            a = np.concatenate((a_x, a_y, a_p))
            b = np.concatenate((b_x, b_y, b_p))
        else:
            a = np.stack([a_x, a_y])
            b = np.stack([b_x, b_y])
        
        return self.return_func(a, b, index)

    def __len__(self):
        return len(self.paths_a)





# root_path = "./data/generated_data/"
# data_file = "c_dataconf.txt"

# sim_data = SimulationDataLoader(root_path, data_file, {})
# sim_data_loader = DataLoader(dataset=sim_data, num_workers=2, batch_size=20, shuffle=True)


# print(len(sim_data))

# s = sim_data[5][1].shape
# print(s)
# # for batch in sim_data_loader:
# #     print(batch)
