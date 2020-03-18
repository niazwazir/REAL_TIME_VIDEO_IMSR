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
import torch.optim
import pylab
import matplotlib.pyplot as plt




if __name__ == "__main__":
    UPSCALE_FACTOR = 2
    net = Net(upscale_factor=UPSCALE_FACTOR)
    print(net)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on', device)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=10e-3)

    " train net "
    epochs = []
    losses = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel('total loss')
    ax.set_xlabel('epoch')
    Ln, = ax.plot([0],[1])
    pylab.show()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1000):  # loop over the dataset multiple times
    
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

            running_loss += loss.item()
            
        print('[%d, %5d] total loss: %.3f' %n
              (epoch + 1, i + 1, running_loss,))
        epochs.append(epoch+1)
        losses.append(running_loss)
        Ln.set_ydata(losses)
        Ln.set_xdata(epochs)
        ax.set_xlim(1,epoch+1)
        ax.set_ylim(0,max(losses))
        fig.canvas.draw()
        plt.show()
        plt.pause(0.1)
        
        running_loss = 0.0

        scheduler.step()
        print('lr: ' + str(scheduler.get_lr()))



    print('Finished Training')
    " save "
    PATH = './Trained.pth'
    torch.save(net.state_dict(), PATH)