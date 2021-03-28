# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as numpy
import torch as t
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from blitz.modules import BayesianLinear


# ----------------------------------------------------------------------------------------------------------------------
# Define Bayesian Neural Network
# ----------------------------------------------------------------------------------------------------------------------

class bnn(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, 80)
        self.blinear2 = BayesianLinear(80, output_dim)
        
    def forward(self, x):
        x_ = self.linear(x)
        x_ = self.blinear1(x_)
        return self.blinear2(x_)

    
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
