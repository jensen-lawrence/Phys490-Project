import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPool1D
from tensorflow.keras.layers import BatchNormalization

def get_model(input_shape,dropout):
    model = Sequential()
    model.add(Conv1D(16, 16, input_shape=input_shape))
    model.add(MaxPool1D(pool_size=4,strides=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())
    
    model.add(Conv1D(32, 8))
    model.add(MaxPool1D(pool_size=4,strides=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())

    model.add(Conv1D(64, 8))
    model.add(MaxPool1D(pool_size=4,strides=4))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(rate=dropout))
    model.add(Dense(64))
    model.add(Activation(tf.keras.activations.relu))
    model.add(BatchNormalization())

    model.add(Dense(1))
    model.add(Activation(tf.keras.activations.sigmoid))
    return model

def learning_rate_callback(epoch,lr):
    if epoch%20==0:
        return lr/10
    else:
        return lr