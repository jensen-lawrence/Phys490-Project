# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# General imports
import sys
import time
import json
import argparse
import numpy as np

# Machine learning imports
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as func
sys.path.append('../src')
from get_data import get_data
from cnn import CNN


# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------

def run_model(params, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None):
    """
    Docstring goes here
    """
    learn_rate = params['learn_rate']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    display_every = params['display_every']

    device = torch.device('cpu')

    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_data = torch.utils.data.DataLoader(train, batch_size, shuffle=True)

    valid = torch.utils.data.TensorDataset(X_valid, y_valid)
    valid_data = torch.utils.data.DataLoader(valid, batch_size, shuffle=True)

    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_func = nn.BCELoss(reduction='mean')
    train_loss_vals, train_acc_vals, train_auc_vals = [], [], []
    valid_loss_vals, valid_acc_vals, valid_auc_vals = [], [], []

    t_train = time.time()
    print('-'*80)
    print('Training model...')
    print('-'*80)
    for epoch in range(n_epochs):
        t_epoch = time.time()
        model.train()
        train_loss, train_acc, train_auc = 0, 0, 0
        valid_loss, valid_acc, valid_auc = 0, 0, 0

        for i, (train_signals, train_labels) in enumerate(train_data):
            optimizer.zero_grad()
            train_labels_out = model(train_signals)

            step_train_loss = loss_func(train_labels_out, train_labels)
            train_loss += step_train_loss.item()
            step_train_acc = accuracy_score(train_labels, torch.round(train_labels_out).detach().numpy())
            train_acc += step_train_acc
            step_train_auc = roc_auc_score(train_labels, torch.round(train_labels_out).detach().numpy())
            train_auc += step_train_auc

            step_train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                for (valid_signals, valid_labels) in valid_data:
                    valid_labels_out = model(valid_signals)
                    step_valid_loss = loss_func(valid_labels_out, valid_labels)
                    valid_loss += step_valid_loss.item()
                    step_valid_acc = accuracy_score(valid_labels, torch.round(valid_labels_out).detach().numpy())
                    valid_acc += step_valid_acc
                    step_valid_auc = roc_auc_score(valid_labels, torch.round(valid_labels_out).detach().numpy())
                    valid_auc += step_valid_auc

            if (i + 1) % display_every == 0:
                print(f'Epoch {epoch + 1} | [{(i + 1)*batch_size}/{len(train_data.dataset)}]')
                print(f'- Training loss: {step_train_loss.item():.4f} \tValidation loss: {step_valid_loss.item():.4f}')
                print(f'- Training accuracy: {step_train_acc:.4f} \tValidation accuracy: {step_valid_acc:.4f}')
                print(f'- Training AUC score: {step_train_auc:.4f} \tValidation AUC score: {step_valid_auc:.4f}')

        train_loss_vals.append(train_loss/batch_size)
        train_acc_vals.append(train_acc/batch_size)
        train_auc_vals.append(train_auc/batch_size)
        valid_loss_vals.append(valid_loss/(batch_size * len(valid_data.dataset)))
        valid_acc_vals.append(valid_acc/(batch_size * len(valid_data.dataset)))
        valid_auc_vals.append(valid_auc/(batch_size * len(valid_data.dataset)))

        print('-'*80)
        print(f'EPOCH {epoch + 1} COMPLETE')
        print(f'Average training loss: {train_loss/batch_size:.4f} \tAverage validation loss: {valid_loss/(batch_size * len(valid_data.dataset)):.4f}')
        print(f'Average training accuracy: {train_acc/batch_size:.4f} \tAverage validation accuracy: {valid_acc/(batch_size * len(valid_data.dataset)):.4f}')
        print(f'Average training AUC score: {train_auc/batch_size:.4f} \tAverage validation AUC score: {valid_auc/(batch_size * len(valid_data.dataset)):.4f}')
        print(f'Time elapsed: {(time.time() - t_epoch)//60} min {(time.time() - t_epoch) % 60} sec')
        print('-'*80)

        if (epoch + 1) % 10 == 0:
            learn_rate /= 10.0
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    print('TRAINING COMPLETE. SUMMARY OF RESULTS:')
    print(f'Final training loss: {train_loss_vals[-1]:.4f} \tFinal validation loss: {valid_loss_vals[-1]:.4f}')
    print(f'Final training accuracy: {train_acc_vals[-1]:.4f} \tFinal validation accuracy: {valid_acc_vals[-1]:.4f}')
    print(f'Final training AUC score: {train_auc_vals[-1]:.4f} \tFinal validation AUC score: {valid_auc_vals[-1]:.4f}')
    print(f'Time elapsed: {(time.time() - t_train)//60} min {(time.time() - t_train) % 60} sec')
    print('-'*80)

    return 0



params_file = '../param/cnn_params.json'
data_file = '../../gw_data/training_1'

with open(params_file) as f:
    params = json.load(f)
f.close()

n_train = params['n_train']
n_valid = params['n_valid']

X, y = get_data(data_file, n_train + n_valid)

# Training/validation split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=n_valid)
X_train = torch.from_numpy(X_train.reshape(X_train.shape[0], 1, X_train.shape[1])).float()
X_valid = torch.from_numpy(X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])).float()
y_train = torch.from_numpy(y_train.reshape(y_train.size, 1)).float()
y_valid = torch.from_numpy(y_valid.reshape(y_valid.size, 1)).float()

run_model(params, X_train, y_train, X_valid, y_valid)

# model = CNN()
# a = model(X_train)
# print(nn.BCELoss(reduction='mean')(a, y_train))
# print(accuracy_score(y_train, torch.round(a).detach().numpy()))
# print(roc_auc_score(y_train, torch.round(a).detach().numpy()))
# print(torch.from_numpy(y_train).float())