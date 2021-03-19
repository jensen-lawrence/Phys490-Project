# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalizer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as layers
from get_data import get_data


# ----------------------------------------------------------------------------------------------------------------------
# Data Processing
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Convolution Neural Network
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description goes here')
    parser.add_argument('--param', type='str')
    parser.add_argument('--data', type='str')

    with open(args.param) as f:
        nn_params = json.load(f)
    f.close()

    model = Sequential()
