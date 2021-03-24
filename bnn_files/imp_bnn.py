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
from bnn import bnn
from get_data import get_data


# ----------------------------------------------------------------------------------------------------------------------
# Data Processing
# ----------------------------------------------------------------------------------------------------------------------

def run(num_epoch, display_epoch):
    

    for epoch in range(num_epoch):
        for epoch, (datapoints, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()
            
            loss = regressor.sample_elbo(inputs=datapoints, labels=labels, criterion=criterion, sample_nbr=3)
            loss.backward()
            optimizer.step()
            

            if epoch%display_epoch==1:
                ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor,
                                                                            X_test,
                                                                            y_test,
                                                                            samples=25,
                                                                            std_multiplier=3)
                
                print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))
                print("Loss: {:.4f}".format(loss))

# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Bayesian Neural Network
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description goes here')
    parser.add_argument('--param', type='str')
    parser.add_argument('--data', type='str')

    with open(args.param) as f:
        nn_params = json.load(f)
    f.close()

    model = bnn()

