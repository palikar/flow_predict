#!/usr/bin/env python

import os
import sys
import argparse

from config import configuration_classes as CONFIGURATIONS
from config import default_config as DEFAULT_CONFIG


VERSION_MSG = ["experimental" ]



def train(parser, args):
    conf = args.config

    conf_object = CONFIGURATIONS[conf]()

    print(f'config: {conf}')
    

def test(args):
    pass

def visu(args):
    pass

def main():
    
    parser = argparse.ArgumentParser(
        prog="flow", description="Train, evaluate and visualize"
    )

    parser.add_argument("--version", "-v", action="version", version=("\n".join(VERSION_MSG)), help="Print veriosn inormation")
    parser.add_argument('--seed', dest='seed', type=int, default=42, help='Random seed to use.')

    subparsers = parser.add_subparsers(title="Commands", description="All phases of ML", dest="command", metavar="Command",)

    
    parser_train = subparsers.add_parser("train", description="Train the models", help="Training the models")
    parser_eval = subparsers.add_parser("eval", description="Eval the models", help="Evaluating the models")
    parser_visu = subparsers.add_parser("visu", description="Visualize results", help="Visualizign the results")

    parser_train.add_argument('--config', dest='config', default=DEFAULT_CONFIG, help='Configuration')

    args = parser.parse_known_args()

    if args[0].command == 'train':
        train(parser, args[0])
    elif args[0].command == 'eval':
        test(parser, args[0])
    elif args[0].command == 'visu':
        visu(parser, args[0])

if __name__ == '__main__':
    main()
