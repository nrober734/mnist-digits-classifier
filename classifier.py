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

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",sep=",",header=None)
dataframe = dataframe.head(10000)
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
dataframe.head()


#print((dataframe.loc[:,240]))

def create_features:







