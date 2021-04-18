# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import blitz
import numpy as numpy
import torch as t
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from blitz.modules import BayesianLinear, BayesianConv1d


# ----------------------------------------------------------------------------------------------------------------------
# Define Bayesian Neural Network
# ----------------------------------------------------------------------------------------------------------------------

class bnn(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            BayesianConv1d(1,16, 16),
            nn.MaxPool1d(4),
            nn.ReLU(),
            BayesianConv1d(16,32, 8),
            nn.MaxPool1d(4),
            nn.ReLU(),
            BayesianConv1d(32,64, 8),
            nn.MaxPool1d(4),
            nn.ReLU(),
            nn.Flatten(1,-1),
            BayesianLinear(256, 120),
            nn.ReLU(),
            BayesianLinear(120, 84),
            nn.ReLU(),
            BayesianLinear(84, 10)
        )

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        # out = F.max_pool1d(4)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool1d(4)
        # out = F.relu(self.conv3(out))
        # out = F.max_pool1d(4)
        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)

        return self.layers(out)

    
    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
