import argparse
import sys
import time

from model import *
from utils.datasets import *
from utils.utils import *
from utils import torch_utils
import torch
import os

# Import test.py to get mAP after each epoch
import test

DARKNET_WEIGHTS_FILENAME = 'darknet53.conv.74'
DARKNET_WEIGHTS_URL = 'https://pjreddie.com/media/files/{}'.format(DARKNET_WEIGHTS_FILENAME)


def updateBN(model, s):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add(s * torch.sign(m.weight.data))  # L1 Sparsity


def train(
        net_config_path,
        data_config_path,
        img_size=416,
        resume=False,
        epochs=100,
        batch_size=16,
        weights_path='weights',
        report=False,
        multi_scale=False,
        freeze_backbone=True,
        var=0,
        s=0.0001,
):
    device = torch_utils.select_device()
    print("Using device: \"{}\"".format(device))

    if not multi_scale:
        torch.backends.cudnn.benchmark = True

    os.makedirs(weights_path, exist_ok=True)
    latest_weights_file=os.path.join(weights_path,'latest.pt')
    best_weights_file=os.path.join(weights_path,'best.pt')

    #Configure run
    data_config = parse_config.parse_data_config(data_config_path)
    num_classes = int(data_config['classes'])
    train_path = data_config['train']

    #Initialize model
    model=Darknet(net_config_path,img_size)

    # Get dataloader
    if multi_scale:                 # pass maximum multi_scale size
        img_size=608


