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
    """
    Convolutional neural network for the classification of gravitation wave signals.

    Structure
    ---------
    Input
    1D Convolution - input channels: 1, output channels: 16, kernel size: 16
    Batch Normalization
    ReLU
    1D Max Pooling - kernel size: 4, stride: 4
    1D Convolution - input channels: 16, output channels: 32, kernel size: 8
    Batch Normalization
    ReLU
    1D Max Pooling - kernel size: 4, stride: 4
    1D Convolution - input channels: 32, output channels: 64, kernel size: 8
    Batch Normalization
    ReLU
    1D Max Pooling - kernel size: 4, stride: 4
    Flatten
    Fully-Connected - input features: 3904, output features: 64
    ReLU
    Dropout - dropout probability: 0.5
    Fully-Connected - input features: 64, output features: 1
    Sigmoid
    Output
    """
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)

        # Fully-connected layers
        self.fc1 = nn.Linear(in_features=3904, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

        # Batch normalization
        self.batchnorm1 = nn.BatchNorm1d(num_features=16)
        self.batchnorm2 = nn.BatchNorm1d(num_features=32)
        self.batchnorm3 = nn.BatchNorm1d(num_features=64)

        # Additional methods
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(p=0.5)


    # Feedforward function
    def forward(self, x):
        # First convolution block
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = func.relu(x)
        x = self.maxpool(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = func.relu(x)
        x = self.maxpool(x)

        # Third convolution block
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = func.relu(x)
        x = self.maxpool(x)

        # First fully-connected block
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = func.relu(x)
        x = self.dropout(x)

        # Output
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


    # Reset fully-connected layers
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


# ----------------------------------------------------------------------------------------------------------------------