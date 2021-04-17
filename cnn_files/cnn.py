# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as func


# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Convolution Neural Network
# ----------------------------------------------------------------------------------------------------------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.batchnorm1 = nn.BatchNorm1d(1)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.batchnorm4 = nn.BatchNorm1d(64)

        self.conv1 = nn.Conv1d(1, 16, 16)
        self.conv2 = nn.Conv1d(16, 32, 8)
        self.conv3 = nn.Conv1d(32, 64, 8)

        self.fc1 = nn.Linear(3904, 64)
        self.fc2 = nn.Linear(64, 1)

        self.maxpool = nn.MaxPool1d(4, stride=4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.conv1(x)
        x = nn.ReLU()(self.maxpool(x))
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = nn.ReLU()(self.maxpool(x))
        x = self.batchnorm3(x)
        x = self.conv3(x)
        x = nn.ReLU()(self.maxpool(x))
        x = self.batchnorm4(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = nn.ReLU()(self.dropout(self.fc1(x)))
        x = nn.Sigmoid()(self.fc2(x))
        return x

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


# ----------------------------------------------------------------------------------------------------------------------