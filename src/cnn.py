# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPool1D
from tensorflow.keras.layers import BatchNormalization
sys.path.append('.')

from get_data import get_data


# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Convolution Neural Network
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Implementing argument parser
    parser = argparse.ArgumentParser(description='Description goes here')
    parser.add_argument('--param', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('-v', type=int)
    args = parser.parse_args()

    # Loading neural network parameters
    with open(args.param) as f:
        nn_params = json.load(f)
    f.close()

    n_train = nn_params['n_train']
    n_valid = nn_params['n_valid']
    dropout = nn_params['dropout']
    learn_rate = nn_params['learn_rate']
    n_epochs = nn_params['n_epochs']
    batch_size = nn_params['batch_size']

    # Loading training data
    X, y = get_data(args.data, n_train + n_valid)

    # Batch normalization
    X = normalize(X)

    # Training/validation split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=n_valid)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    y_train = y_train.reshape(y_train.size, 1, 1)
    y_valid = y_valid.reshape(y_valid.size, 1, 1)
    input_shape = X_train.shape[1:]

    # Initializing convolutional neural network
    model = Sequential()
    model.add(Conv1D(16, 16, input_shape=input_shape))
    model.add(MaxPool1D(pool_size=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8))
    model.add(MaxPool1D(pool_size=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 8))
    model.add(MaxPool1D(pool_size=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(rate=dropout))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation(tf.keras.activations.sigmoid))

    # Initializing loss function, optimizer, and performance metrics
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    metrics = ['accuracy', 'AUC']

    # Training and evaluating model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    training_results = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=args.v, validation_data=(X_valid, y_valid))

    # model.evaluate(X_test, y_test, batch_size=batch_size, verbose=args.v)


# ----------------------------------------------------------------------------------------------------------------------