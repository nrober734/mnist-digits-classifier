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
import matplotlib.pyplot as plt

# Load data set into two variables
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Separate between training set and test set
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)









