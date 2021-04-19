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
from cnn_funcs import *
from get_data import get_data
from tensorflow_addons.optimizers import AdamW
# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Convolution Neural Network
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    # Implementing argument parser
    parser = argparse.ArgumentParser(description='Description goes here')
    parser.add_argument('--param',default='param/cnn_params.json', type=str)
    parser.add_argument('--data',default='F:\phys490_data\data_2_training', type=str)
    parser.add_argument('-v',default=2, type=int)
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
    weight_decay = nn_params['weight_decay']

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

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
    model=get_model(input_shape,dropout)

    # Initializing loss function, optimizer, and performance metrics
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = AdamW(weight_decay=weight_decay,learning_rate=learn_rate)
    metrics = ['accuracy', 'AUC']
    
    for i, layer in enumerate(model.layers):
        tf.summary.histogram('layer{0}'.format(i), layer)

    # Training and evaluating model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    training_results = model.fit(X_train, y_train, epochs=n_epochs, 
        batch_size=batch_size, verbose=args.v, validation_data=(X_valid, y_valid),
        callbacks=[tensorboard_callback,tf.keras.callbacks.LearningRateScheduler(learning_rate_callback)])

    # model.evaluate(X_test, y_test, batch_size=batch_size, verbose=args.v)


# ----------------------------------------------------------------------------------------------------------------------