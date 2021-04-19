# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(0,'../src')
from get_data import *

import json
import argparse
import blitz,torch
import numpy as np
import torch.optim as optim
import torch.functional as F
from blitz.utils import variational_estimator
from sklearn.model_selection import train_test_split
from bnn import bnn


# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN model
# ----------------------------------------------------------------------------------------------------------------------

def acc(labels,output):
    output=torch.round(output).cpu().detach().numpy()
    labels=labels.cpu().detach().numpy()
    numequal = np.equal(output,labels).size
    return numequal/labels.size

def run(param, train_in, train_out, test_in, test_out):
    device = torch.device('cuda')
    lr = param['learn_rate']
    batch_size = param['batch_size']
    num_epoch = param['n_epoch']
    display_epoch = param['display_epoch']

    sample_nbr = 3

    model = bnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    loss_func = torch.nn.BCELoss()

    train = torch.utils.data.TensorDataset(train_in, train_out)
    train_data = torch.utils.data.DataLoader(train, batch_size)

    test = torch.utils.data.TensorDataset(test_in, test_out)
    test_data = torch.utils.data.DataLoader(test, batch_size)
    loss_vals=[]
    accuracy_vals=[]

    test_loss_vals=[]
    test_acc_vals=[]
    for epoch in range(num_epoch):
        loss_val=0
        acc_val=0
        for i, (signals_t, labels_t) in enumerate(train_data):
            signals_t=signals_t
            out = model.forward(signals_t)
            optimizer.zero_grad()
                
            #loss = model.sample_elbo(signals_t, labels_t, loss_func, sample_nbr)
            loss = loss_func(out,labels_t)
            loss.backward()
            optimizer.step() 
            loss_val+=loss.item()/len(train_data)
            acc_val+=acc(labels_t,out)
        accuracy_vals.append(acc_val/len(train_data))
            
        test_accuracy=0
        test_loss=0
        with torch.no_grad():
            for (signals_test, labels_test) in test_data:
                signals_test=signals_test.to(device)
                labels_test=labels_test.to(device)

                output = model.forward(signals_t)
                test_loss += loss_func(output,labels_test).item()
                test_accuracy += acc(labels_test, output)
        test_acc_vals.append(test_accuracy/len(test_data))

        #if epoch % display_epoch==1:
        print("Training Loss: {:.4f}".format(loss_val))
        print("Test Loss: {:.4f}".format(test_loss))
    
    print("Final Training Loss: {:.4f}".format(loss))
    print("Final Test Loss: {:.4f}".format(test_loss))

# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Bayesian Neural Network
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description goes here')
    parser.add_argument('--param',default='../param/bnn_params.json', type=str, help='Path where params are stored')
    parser.add_argument('--data',default='F:\phys490_data\data_1_training',type=str, help='Path where data is stored')
    args=parser.parse_args()
    with open(args.param) as f:
        params = json.load(f)
    f.close()
    device = torch.device('cuda')
    n_valid = params['n_valid']
    batch_size=params['batch_size']
    signals, labels = get_data(args.data, 5500)
    x_train, x_valid, y_train, y_valid = train_test_split(signals, labels, test_size=n_valid)
    x_train = torch.from_numpy(x_train.reshape(x_train.shape[0], 1, x_train.shape[1])).float().to(device)
    x_valid = torch.from_numpy(x_valid.reshape(x_valid.shape[0], 1, x_valid.shape[1])).float().to(device)
    y_train = torch.from_numpy(y_train.reshape(y_train.size, 1)).float().to(device)
    y_valid = torch.from_numpy(y_valid.reshape(y_valid.size, 1)).float().to(device)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_data = torch.utils.data.DataLoader(train, batch_size, shuffle=True)

    valid = torch.utils.data.TensorDataset(x_valid, y_valid)
    valid_data = torch.utils.data.DataLoader(valid, batch_size, shuffle=True)
    
    run(params, x_train, y_train, x_valid, y_valid)
