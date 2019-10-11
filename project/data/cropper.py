#!/usr/bin/env python
import os
import sys
import itertools
import PIL
from PIL import Image

HEIGHT_BEGIN = 245
HEIGHT_END = 535

WIDTH_BEGIN = 0
WIDTH_END = -1

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print('Cropping image:', input_file)
    
    im = Image.open(input_file)
    w, h = im.size

    w_begin = 0 if WIDTH_BEGIN == -1 else WIDTH_BEGIN
    w_end = w if WIDTH_END == -1 else WIDTH_END
    h_begin = 0 if HEIGHT_BEGIN == -1 else HEIGHT_BEGIN
    h_end = h if HEIGHT_END == -1 else HEIGHT_END
    
    img_new = im.crop((w_begin, h_begin, w_end, h_end))

    img_new.save(output_file)
    

if __name__ == '__main__':
    main()
