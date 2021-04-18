# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.insert(0,'../src')
import get_data
import json
import argparse
import blitz,torch
import numpy as np
from blitz.utils import variational_estimator
from sklearn.model_selection import train_test_split
from bnn import bnn


# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN model
# ----------------------------------------------------------------------------------------------------------------------

def acc(labels,output)
    output=torch.round(output).cpu().detach().numpy()
    labels=labels.cpu().detach().numpy()
    numequal = np.equal(output,labels).size
    return numequal/labels.size

def run(param, train_in, train_out, test_in, test_out):
    device = torch.device('cuda')
    lr = param['learn_rate']
    batch_size = param['batch_size']
    num_epoch = param['num_epoch']
    display_epoch = param['display_epoch']

    in_dim = train_in.shape[0]
    out_dim = train_out.shape[0]
    sample_nbr = 3

    model = bnn(in_dim, out_dim)
    optimizer = optim.Adam(model.parameters(), lr)
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
                    signals_test=signals_test.to(device)
                    labels_test=labels_test.to(device)

                    output = model.forward(signals_t)
                    test_loss = loss(labels_test, output)
                    test_accuracy = acc(labels_test,output)

        #if epoch % display_epoch==1:
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

    n_valid = nn_params['n_valid']

    signals, labels = get_data(args.data, 500)
    x_train, x_valid, y_train, y_valid = train_test_split(signals, labels, test_size=n_valid)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_data = torch.utils.data.DataLoader(train, batch_size, shuffle=True)

    valid = torch.utils.data.TensorDataset(x_valid, y_valid)
    valid_data = torch.utils.data.DataLoader(valid, batch_size, shuffle=True)
    
    run(params, x_train, y_train, x_valid, y_valid)
