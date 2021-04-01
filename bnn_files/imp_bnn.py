# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('../src')
import get_data
import json
import argparse
import blitz
import numpy as np
from blitz.utils import variational_estimator
from sklearn.model_selection import train_test_split
from bnn import bnn


# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN model
# ----------------------------------------------------------------------------------------------------------------------

def run(param, train_in, train_out, test_in, test_out):

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
    train_data = torch.utils.data.DataLoader(train, batch_size)

    test = torch.utils.data.TensorDataset(test_in, test_out)
    test_data = torch.utils.data.DataLoader(test, batch_size)


    for epoch in range(num_epoch):
        for i, (signals_t, labels_t) in enumerate(train_data):
            out = model.forward(signals_t)
            optimizer.zero_grad()
                
            loss = model.sample_elbo(signals_t, labels_t, loss_func, sample_nbr)
            loss.backward()
            optimizer.step()
                
            with torch.no_grad():
                for (signals_test, labels_test) in test_data:
                    output = model.forward((signals_t))
                    test_loss = loss(labels_test, output)

            if epoch%display_epoch==1:
                print("Training Loss: {:.4f}".format(loss))
                print("Test Loss: {:.4f}".format(test_loss))
    
    print("Final Training Loss: {:.4f}".format(loss))
    print("Final Test Loss: {:.4f}".format(test_loss))

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

    signals, labels = get_data(args.data, 500)
    run(params, signals, labels)

