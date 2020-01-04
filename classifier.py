from __future__ import print_function

import glob
import math
import os
from IPython import display
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sb
import sklearn

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.functional as F



class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x)

def train(self, epoch):
    # stuff

def main(self):
    # Load data set into two variables
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Separate between training set and test set
    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)









