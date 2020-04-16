import torch.nn as nn
from torch import sigmoid, tanh

class Net(nn.Module):
    def __init__(self, r, C):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(C, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, r * r * C, 3, padding=1)

    def forward(self, x):
        x = tanh(self.conv1(x))
        x = tanh(self.conv2(x))
        x = self.conv3(x)
        return x