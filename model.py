#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:21:31 2020

@author: djoghurt
"""
import torch.nn as nn
from torch import sigmoid, tanh

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = tanh(self.conv1(x))
        x = tanh(self.conv2(x))
        x = tanh(self.conv3(x))
        x = tanh(self.conv4(x))
        x = tanh(self.conv5(x))
        x = sigmoid(self.pixel_shuffle(self.conv6(x)))
        return x
