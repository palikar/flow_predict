#!/usr/bin/env python


import sys
import os

import imageio
from pygifsicle import optimize


def main():
    
    with imageio.get_writer(sys.argv[1], mode='I') as writer:
        for filename in sys.argv[2:]:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    optimize(sys.argv[1])

if __name__ == '__main__':
    main()
