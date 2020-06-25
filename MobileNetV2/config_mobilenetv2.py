import argparse
import os
import datetime
import torch

parser = argparse.ArgumentParser(description = "Initialize parameters")

parser.add_argument("--batch_size", type = int, default = 64)
parser.add_argument("--epoch", type = int, default = 120)
parser.add_argument("--lr", type = float, default = 0.1)
parser.add_argument("--scheduler_policy", type = str, default = 'multistep', choices = ['multistep', 'plateau'])
parser.add_argument("--momentum", type = float, default = 0.9) 
parser.add_argument("--weight_decay", type = float, default = 0)
parser.add_argument("--width_mult", type = float, default = 1.0)
parser.add_argument("--density_factor", type = float, default = 1)
parser.add_argument("--no_bias_decay", action = 'store_true')
parser.add_argument("--no_cuda", action = 'store_true', default = False)
parser.add_argument("--debug", action = 'store_true', default = False)
parser.add_argument("--optim", type = str, default = 'sgd', choices = ['adam', 'sgd'])
parser.add_argument("--init_weight", type = str, default = None, choices = ['xav', 'kaim'])


args = parser.parse_args()

