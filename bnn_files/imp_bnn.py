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
from blitz.utils import variational_estimator
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from get_data import get_data


# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN model
# ----------------------------------------------------------------------------------------------------------------------

def run(param, train_in, train_out):
    
    model = BayesianRegressor(in_dim, out_dim)
    optimizer = optim.SGD(regressor.parameters(), lr)
    loss_func = torch.nn.MSELoss()

    train = torch.utils.data.TensorDataset(train_in, train_out)
    dataloader_train = torch.utils.data.DataLoader(train, batch_size)

 

    for epoch in range(num_epoch):
        optimizer.zero_grad()
            
        loss = regressor.sample_elbo(inputs=datapoints, labels=labels, criterion=loss_func, sample_nbr=3)
        loss.backward()
        optimizer.step()
            

        if epoch%display_epoch==1:
            print("Loss: {:.4f}".format(loss))
    
    print("Final Loss: {:.4f}".format(loss))

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

    data_in, data_out = get_data()
    run(nn_params, data_in, data_out)

