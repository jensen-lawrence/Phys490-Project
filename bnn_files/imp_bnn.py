# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('../src')
from get_data import get_data
import json
import argparse
import blitz
import numpy as np
from blitz.utils import variational_estimator
from bnn import bnn


# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN model
# ----------------------------------------------------------------------------------------------------------------------

def run(param, train_in, train_out):

    lr = param['learn_rate']
    batch_size = param['batch_size']
    num_epoch = param['num_epoch']
    display_epoch = param['display_epoch']
    in_dim = train_in.shape[0]
    out_dim = train_out.shape[0]
    sample_nbr = 3

    
    model = bnn(in_dim, out_dim)
    optimizer = optim.SGD(model.parameters(), lr)
    loss_func = torch.nn.CrossEntropyLoss()

    train = torch.utils.data.TensorDataset(train_in, train_out)
    dataloader_train = torch.utils.data.DataLoader(train, batch_size)

 

    for epoch in range(num_epoch):
        for i, (signals_t, labels_t) in enumerate(train_loader):
            out = model.forward(signals_t)
            optimizer.zero_grad()
                
            loss = model.sample_elbo(signals_t, labels_t, loss_func, sample_nbr)
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
    parser.add_argument('--param', type='str', help='Path where Params are stored')
    parser.add_argument('--data', type='str', help='Path where data is stored')

    with open(args.param) as f:
        params = json.load(f)
    f.close()

    signals, labels = gd.get_data(args.data, 500)
    run(params, signals, labels)

