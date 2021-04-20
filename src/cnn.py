# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPool1D

# ----------------------------------------------------------------------------------------------------------------------
# Define Convolution Neural Network
# ----------------------------------------------------------------------------------------------------------------------

def CNN(input_shape, dropout):
    """
    Convolutional neural network for the classification of gravitation wave signals.

    Structure
    ---------
    Input
    1D Convolution - input channels: 1, output channels: 16, kernel size: 16
    1D Max Pooling - kernel size: 4, stride: 4
    ReLU
    Batch Normalization
    1D Convolution - input channels: 16, output channels: 32, kernel size: 8
    1D Max Pooling - kernel size: 4, stride: 4
    ReLU
    Batch Normalization
    1D Convolution - input channels: 32, output channels: 64, kernel size: 8
    1D Max Pooling - kernel size: 4, stride: 4
    ReLU
    Batch Normalization
    Flatten
    Dropout - dropout probability: 0.5
    Fully-Connected - input features: 3904, output features: 64
    ReLU
    Batch Normalization
    Fully-Connected - input features: 64, output features: 1
    Sigmoid
    Output
    """
    model = Sequential()

    # First convolution block
    model.add(Conv1D(16, 16, input_shape=input_shape))
    model.add(MaxPool1D(pool_size=4, strides=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())
    
    # Second convolution block
    model.add(Conv1D(32, 8))
    model.add(MaxPool1D(pool_size=4, strides=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())

    # Third convolution block
    model.add(Conv1D(64, 8))
    model.add(MaxPool1D(pool_size=4, strides=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())

    # Fully-connected layer
    model.add(Flatten())
    model.add(Dropout(rate=dropout))
    model.add(Dense(64))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())

    # Output
    model.add(Dense(1))
    model.add(Activation(tf.keras.activations.sigmoid))
    return model

# ----------------------------------------------------------------------------------------------------------------------