#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:22:48 2020

@author: djoghurt
"""

from model import Net
from torchvision import transforms
from data_utils import DatasetFromFolder
import torch
import torch.nn as nn


if __name__ == "__main__":
    UPSCALE_FACTOR = 2
    net = Net(upscale_factor=UPSCALE_FACTOR)
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device == 'cuda':
        net.cuda()
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
    ])
    
    # trainset = torchvision.datasets.ImageFolder(root = './data/train/SRF_3', transform=transforms.ToTensor(),
    #                                  target_transform=None)
    
    trainset = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    
    testset = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    
    # testset = torchvision.datasets.ImageFolder(root = './data/val/SRF_3', transform=transform,
    #                                  target_transform=None)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    
    
    
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    
    " train net "
    for epoch in range(3):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')
    " save "
    PATH = './rik_newTrained.pth'
    torch.save(net.state_dict(), PATH)