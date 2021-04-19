# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import blitz
import numpy as numpy
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from blitz.modules import BayesianLinear, BayesianConv1d


# ----------------------------------------------------------------------------------------------------------------------
# Define Bayesian Neural Network
# ----------------------------------------------------------------------------------------------------------------------

class bnn(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.dropout = nn.Dropout(0.5)
        self.conv1 = BayesianConv1d(1,16, 16)
        self.conv2 = BayesianConv1d(16,32, 8)
        self.conv3 = BayesianConv1d(32,64, 8)
        self.fc1   = BayesianLinear(3904, 64)
        self.fc2   = BayesianLinear(64, 1)

    def forward(self, x):
        out = self.dropout(x)
        out = F.relu(self.conv1(out))
        out = F.max_pool1d(out,4)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out,4)
        out = F.relu(self.conv3(out))
        out = F.max_pool1d(out,4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
    
    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
