#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

import numpy as np

plt.style.use('ggplot')
plt.rcParams.update({'figure.max_open_warning': 0})


def main():

    matplotlib.rc('xtick', labelsize=17)
    matplotlib.rc('ytick', labelsize=17)

    x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    y = np.linspace(44.6, 33.3, len(x))

    yerr = np.linspace(1, 5, len(x))
    
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='-o', ecolor='black', mfc='white', mec='black', color='black')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Frame index', fontsize=16)
    plt.ylabel('PSNR', fontsize=16)
    plt.title('Constant model (recursive application performance)', fontsize=17)
    plt.show()

if __name__ == '__main__':
    main()
