# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
import sys
from get_data import get_data

import json
import argparse
import blitz,torch
import numpy as np
import torch.optim as optim
import torch.functional as F
from blitz.utils import variational_estimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import normalize
from bnn import bnn
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN model
# ----------------------------------------------------------------------------------------------------------------------

def acc(labels,output):
    output=torch.round(output).cpu().detach().numpy().astype(int)
    labels=labels.cpu().detach().numpy().astype(int)
    numequal = np.sum(np.equal(output,labels).astype(int))
    return numequal/labels.size

def bnn_train(param,data_path,v,results_dir):
    with open(param) as f:
        params = json.load(f)
    f.close()
    device = torch.device(params['device'])
    n_valid = params['n_valid']
    batch_size=params['batch_size']
    signals, labels = get_data(args.train, 5500)
    signals=normalize(signals)
    x_train, x_valid, y_train, y_valid = train_test_split(signals, labels, test_size=n_valid)
    train_in = torch.from_numpy(x_train.reshape(x_train.shape[0], 1, x_train.shape[1])).float().to(device)
    test_in = torch.from_numpy(x_valid.reshape(x_valid.shape[0], 1, x_valid.shape[1])).float().to(device)
    train_out = torch.from_numpy(y_train.reshape(y_train.size, 1)).float().to(device)
    test_out = torch.from_numpy(y_valid.reshape(y_valid.size, 1)).float().to(device)
    
    lr = params['learn_rate']
    batch_size = params['batch_size']
    num_epoch = params['n_epoch']
    display_epoch = params['display_epoch']
    cw=1e-7
    sample_nbr = 3

    model = bnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    loss_func = torch.nn.BCELoss()
    complexity=model.nn_kl_divergence()
    train = torch.utils.data.TensorDataset(train_in, train_out)
    train_data = torch.utils.data.DataLoader(train, batch_size)

    test = torch.utils.data.TensorDataset(test_in, test_out)
    test_data = torch.utils.data.DataLoader(test, batch_size)
    loss_vals=[]
    accuracy_vals=[]
    auc_vals=[]
    train_preds=[]

    test_loss_vals=[]
    test_min_acc_vals=[]
    test_mean_acc_vals=[]
    test_max_acc_vals=[]
    test_min_auc_vals=[]
    test_mean_auc_vals=[]
    test_max_auc_vals=[]

    test_meanpreds=[]
    test_maxpreds=[]
    test_minpreds=[]
    for epoch in range(num_epoch):
        loss_val=0
        acc_val=0
        for i, (signals_t, labels_t) in enumerate(train_data):
            out = model.forward(signals_t)
            predictions_t=torch.round(out).cpu().detach().numpy().astype(int)
            train_preds.append(predictions_t)
            auc_val = roc_auc_score(predictions_t,labels_t.cpu().detach().numpy())
            optimizer.zero_grad()  
            loss = model.sample_elbo(signals_t, labels_t, loss_func, sample_nbr,complexity_cost_weight=cw)
            loss.backward()
            optimizer.step() 
            loss_val+=loss.item()/(batch_size*len(train_data))
            acc_val+=acc(labels_t,out)/len(train_data)
        accuracy_vals.append(acc_val)
        auc_vals.append(auc_val)
        test_mean_accuracy=0
        test_max_accuracy=0
        test_min_accuracy=0
        test_loss=0
        test_mean_auc_val=0
        test_min_auc_val=0
        test_max_auc_val=0
        with torch.no_grad():
            for (signals_test, labels_test) in test_data:
                meanpreds,stdpreds = model.mfvi_forward(signals_test,sample_nbr)
                meanpreds,conf_int = (meanpreds.cpu().detach().numpy().shape,1.645*(1/np.sqrt(len(stdpreds)))*stdpreds.cpu().detach().numpy().shape)
                maxpreds=meanpreds+conf_int
                minpreds=meanpreds-conf_int
                #predictions=torch.round(output).cpu().detach().numpy().astype(int)
                test_meanpreds.append(meanpreds)
                test_meanpreds.append(minpreds)
                test_meanpreds.append(maxpreds)
                test_mean_auc_val += roc_auc_score(labels_test.cpu().detach().numpy().astype(int),meanpreds)/len(test_data) 
                test_max_auc_val += roc_auc_score(labels_test.cpu().detach().numpy().astype(int),maxpreds)/len(test_data) 
                test_min_auc_val += roc_auc_score(labels_test.cpu().detach().numpy().astype(int),minpreds)/len(test_data) 
                test_loss += model.sample_elbo(signals_test, labels_test, loss_func, sample_nbr,complexity_cost_weight=cw)/(batch_size*len(test_data))
                test_mean_accuracy += acc(labels_test, meanpreds)/len(test_data) 
                test_min_accuracy += acc(labels_test, minpreds)/len(test_data) 
                test_max_accuracy += acc(labels_test, maxnpreds)/len(test_data) 
        test_mean_acc_vals.append(test_mean_accuracy)
        test_mean_auc_vals.append(test_mean_auc_val)
        test_min_acc_vals.append(test_min_accuracy)
        test_min_auc_vals.append(test_min_auc_val)
        test_max_acc_vals.append(test_max_accuracy)
        test_max_auc_vals.append(test_max_auc_val)
        if epoch % 5 == 0:
            lr /= 2.0
            for g in optimizer.param_groups:
                g['lr'] = lr

        #if epoch % display_epoch==1:
        if args.v >0:
            print("Epoch: {} - Training Loss: {:.4f} - Test Loss: {:.4f}".format(epoch,loss_val,test_loss))
            print("Training Accuracy: {:.4f} - Test Accuracy: {:.4f} - Test AUC: {:.4f}".format(acc_val,test_accuracy,test_auc_val))
            print("-"*40)

    print("Final Training Loss: {:.4f}".format(loss))
    print("Final Test Loss: {:.4f}".format(test_loss))
    training_results = {'train_loss': loss_vals,'test_loss':test_loss_vals,
        'train_acc':accuracy_vals,'test_acc':test_mean_acc_vals,
        'train_auc':auc_vals,'test_auc':test_mean_auc_vals}
    train_roc = roc_curve(train_out, train_preds.flatten())
    fpr_train = train_roc[0]
    tpr_train = train_roc[1]

    # Getting validation ROC curve
    test_mean_roc = roc_curve(test_out, test_meanpredes.flatten())
    fpr_test_mean = test_mean_roc[0]
    tpr_test_mean = test_mean_roc[1]

    # Plotting ROC curves
    print('Plotting ROC curves...')
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    plt.plot(fpr_train, tpr_train, color='dodgerblue', label='Training')
    plt.plot(fpr_test_mean, tpr_test_mean , color='springgreen', label='Validation (Mean)')
    plt.plot(fpr_test_min, tpr_test_min , color='springgreen', label='Validation (Min)')
    plt.plot(fpr_test_max, tpr_test_max , color='springgreen', label='Validation (Max)')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'bnn_roc_curve.png', dpi=400)
    print('Plotted training and validation ROC curves.')
    print('-'*80)
    plt.close()

# def plot_bnn_roc(y_train, train_pred, y_valid, valid_pred, save_as):
#     # Getting training ROC curve
#     train_roc = roc_curve(y_train, train_pred)
#     fpr_train = train_roc[0]
#     tpr_train = train_roc[1]

#     # Getting validation ROC curve
#     valid_roc = roc_curve(y_valid, valid_pred)
#     fpr_valid = valid_roc[0]
#     tpr_valid = valid_roc[1]

#     # Plotting ROC curves
#     print('Plotting ROC curves...')
#     sns.set_style('whitegrid')
#     plt.figure(figsize=(12, 8))
#     plt.plot(fpr_train, tpr_train, color='dodgerblue', label='Training')
#     plt.plot(fpr_valid, tpr_valid, color='springgreen', label='Validation')
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#     plt.legend(loc='best', fontsize=16)
#     plt.savefig(save_as + 'bnn_roc_curve.png', dpi=400)
#     print('Plotted training and validation ROC curves.')
#     print('-'*80)
#     plt.close()

def plot_bnn(training_params,testing_params):
    train_loss=training_params['train_loss']
    test_loss=training_params['test_loss']
    train_acc=training_params['train_acc']
    test_acc=training_params['test_acc']
    train_auc=training_params['train_auc']
    test_auc=training_params['test_auc']

    epochs = range(1, len(train_loss) + 1)
    sns.set_style('whitegrid')

    # Plotting loss
    print('Plotting performance metrics...')
    plt.figure(figsize=(12, 8))
    ax1 = sns.lineplot(x=epochs, y=train_loss, color='dodgerblue', label='Training')
    ax1 = sns.lineplot(x=epochs, y=valid_loss, color='springgreen', label='Validation')
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'bnn_loss.png', dpi=400)
    print('Plotted training and validation loss.')

    # Plotting accuracy
    plt.figure(figsize=(12, 8))
    ax2 = sns.lineplot(x=epochs, y=train_acc, color='dodgerblue', label='Training')
    ax2 = sns.lineplot(x=epochs, y=valid_acc, color='springgreen', label='Validation')
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    ax2.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'bnn_accuracy.png', dpi=400)
    print('Plotted training and validation accuracy.')

    # Plotting AUC score
    plt.figure(figsize=(12, 8))
    ax3 = sns.lineplot(x=epochs, y=train_auc, color='dodgerblue', label='Training')
    ax3 = sns.lineplot(x=epochs, y=valid_auc, color='springgreen', label='Validation')
    ax3.set_xlabel('Epochs', fontsize=16)
    ax3.set_ylabel('AUC Score', fontsize=16)
    ax3.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'bnn_auc_score.png', dpi=400)
    print('Plotted training and validation AUC score.')
    print('-'*80)
    plt.close()
# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Bayesian Neural Network
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description goes here')
    parser.add_argument('--param',default='../param/bnn_params.json', type=str, help='Path where params are stored')
    parser.add_argument('--train',default='F:\phys490_data\data_1_training',type=str, help='Path where data is stored')
    parser.add_argument('--res',default='../results',type=str, help='Path where results are stored')
    parser.add_argument('-v',default=2,type=int, help='Verbosity')
    args=parser.parse_args()
    bnn_train(args.param,args.v,args.train,args.res)
