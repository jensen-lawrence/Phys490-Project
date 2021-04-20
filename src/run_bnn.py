# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# General imports
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
import blitz
import torch
import torch.optim as optim
import torch.functional as F
from blitz.utils import variational_estimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import normalize

# Custom imports
from get_data import get_data
from bnn import bnn

# ----------------------------------------------------------------------------------------------------------------------
# Run Data Through BNN Model
# ----------------------------------------------------------------------------------------------------------------------

# Accuracy
def acc(labels,output):
    """
    Calculates accuracy of model predictions    
    """
    if torch.is_tensor(output):
        output=np.round(output.cpu().detach().numpy().astype(int))
    else:
        output=np.round(output).astype(int)
    labels=labels.cpu().detach().numpy().astype(int)
    numequal = np.sum(np.equal(output,labels).astype(int))
    return numequal/labels.size

def bnn_train(param, data_path, v, results_dir, test_dir=''):
    """
    Trains bayesian convolutional neural network
    Inputs:
    - param (str): path to parameters json file
    - data_path (str): path to gravitational wave data
    - v (int): verbosity level
    - results_dir (str): path to results folder
    - test_dir (str): path to large test dataset
    """

    # internal variable to enable testing mode
    if not(test_dir==''):
        testing_mode=True
    else:
        testing_mode=False

    # get parameters from file
    with open(param) as f:
        params = json.load(f)
    f.close()
    
    #set parameters for model
    device = torch.device(params['device'])
    n_valid = params['n_valid']
    batch_size=params['batch_size']

    # load training data
    signals, labels = get_data(args.train, 5500)
    signals=normalize(signals)
    x_train, x_valid, y_train, y_valid = train_test_split(signals, labels, test_size=n_valid)
    train_in = torch.from_numpy(x_train.reshape(x_train.shape[0], 1, x_train.shape[1])).float().to(device)
    valid_in = torch.from_numpy(x_valid.reshape(x_valid.shape[0], 1, x_valid.shape[1])).float().to(device)
    train_out = torch.from_numpy(y_train.reshape(y_train.size, 1)).float().to(device)
    valid_out = torch.from_numpy(y_valid.reshape(y_valid.size, 1)).float().to(device)

    #load testing data
    if testing_mode:
        testsignals, testlabels = get_data(test_dir, 5001)
        testsignals=normalize(testsignals)
        x_test, _, y_test, __ = train_test_split(signals, labels, test_size=1)
        test_in = torch.from_numpy(x_test.reshape(x_test.shape[0], 1, x_test.shape[1])).float().to(device)
        test_out = torch.from_numpy(y_test.reshape(y_test.size, 1)).float().to(device)
    
    # get training params
    lr = params['learn_rate']
    batch_size = params['batch_size']
    num_epoch = params['n_epoch']
    display_epoch = params['display_epoch']
    cw=5e-7 #weight for KL divergence in ELBO sampling
    sample_nbr = 3 #number of ELBO samples

    #setup model
    model = bnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    loss_func = torch.nn.BCELoss()
    complexity=model.nn_kl_divergence()

    #setup dataloaders
    train = torch.utils.data.TensorDataset(train_in, train_out)
    train_data = torch.utils.data.DataLoader(train, batch_size)

    valid = torch.utils.data.TensorDataset(valid_in, valid_out)
    valid_data = torch.utils.data.DataLoader(valid, batch_size)
    if testing_mode:
        test=torch.utils.data.TensorDataset(test_in,test_out)
        test_data=torch.utils.data.DataLoader(test, batch_size)

    #initialize arrays
    loss_vals=[]
    accuracy_vals=[]
    auc_vals=[]
    train_preds=[]

    valid_loss_vals=[]
    valid_min_acc_vals=[]
    valid_mean_acc_vals=[]
    valid_max_acc_vals=[]
    valid_min_auc_vals=[]
    valid_mean_auc_vals=[]
    valid_max_auc_vals=[]

    valid_meanpreds=[]
    valid_maxpreds=[]
    valid_minpreds=[]

    if testing_mode:
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

    # main training loop
    for epoch in range(num_epoch):
        loss_val=0
        acc_val=0

        t1=time.time()
        
        #---training data---
        for i, (signals_t, labels_t) in enumerate(train_data):
            optimizer.zero_grad()  
            loss = model.sample_elbo(signals_t, labels_t, loss_func, sample_nbr,complexity_cost_weight=cw)
            loss.backward()
            optimizer.step()
            with torch.no_grad():             
                out = model.forward(signals_t)
                predictions_t=torch.round(out).cpu().detach().numpy().astype(int)
                if epoch==num_epoch-1:
                    train_preds.append(predictions_t)
                auc_val = roc_auc_score(predictions_t,labels_t.cpu().detach().numpy())
                loss_val+=loss.item()/(batch_size*len(train_data))
                acc_val+=acc(labels_t,out)/len(train_data)
        #print(acc_val)
        accuracy_vals.append(acc_val/len(train_data))
        auc_vals.append(auc_val)
        loss_vals.append(loss_val)

        #---validation and testing data---

        valid_mean_accuracy=0
        valid_max_accuracy=0
        valid_min_accuracy=0
        valid_loss=0
        valid_mean_auc_val=0
        valid_min_auc_val=0
        valid_max_auc_val=0

        if testing_mode:
            test_mean_accuracy=0
            test_max_accuracy=0
            test_min_accuracy=0
            test_loss=0
            test_mean_auc_val=0
            test_min_auc_val=0
            test_max_auc_val=0

        with torch.no_grad():
            for (signals_valid, labels_valid) in valid_data:
                meanpreds,stdpreds = model.mfvi_forward(signals_valid,sample_nbr)
                meanpreds,conf_int = (meanpreds.cpu().detach().numpy(),1.645*(1/np.sqrt(len(stdpreds)))*stdpreds.cpu().detach().numpy())
                maxpreds=meanpreds+conf_int
                minpreds=meanpreds-conf_int
                if epoch==num_epoch-1:
                    valid_meanpreds.append(meanpreds)
                    valid_minpreds.append(minpreds)
                    valid_maxpreds.append(maxpreds)
                valid_mean_auc_val += roc_auc_score(labels_valid.cpu().detach().numpy().astype(int),meanpreds)/len(valid_data) 
                valid_max_auc_val += roc_auc_score(labels_valid.cpu().detach().numpy().astype(int),maxpreds)/len(valid_data) 
                valid_min_auc_val += roc_auc_score(labels_valid.cpu().detach().numpy().astype(int),minpreds)/len(valid_data) 
                valid_loss += model.sample_elbo(signals_valid, labels_valid, loss_func, sample_nbr,complexity_cost_weight=cw).item()/(batch_size*len(valid_data))
                valid_mean_accuracy += acc(labels_valid, meanpreds)/len(valid_data) 
                valid_min_accuracy += acc(labels_valid, minpreds)/len(valid_data) 
                valid_max_accuracy += acc(labels_valid, maxpreds)/len(valid_data) 
            
        #---testing dataset---
            if testing_mode:
                for (signals_test, labels_test) in test_data:
                    meanpreds_test,stdpreds_test = model.mfvi_forward(signals_test,sample_nbr)
                    meanpreds_test,conf_int_test = (meanpreds_test.cpu().detach().numpy(),1.645*(1/np.sqrt(len(stdpreds_test)))*stdpreds_test.cpu().detach().numpy())
                    maxpreds_test=meanpreds_test+conf_int_test
                    minpreds_test=meanpreds_test-conf_int_test
                    if epoch==num_epoch-1:
                        test_meanpreds.append(meanpreds_test)
                        test_minpreds.append(minpreds_test)
                        test_maxpreds.append(maxpreds_test)
                    test_mean_auc_val += roc_auc_score(labels_test.cpu().detach().numpy().astype(int),meanpreds_test)/len(test_data) 
                    test_max_auc_val += roc_auc_score(labels_test.cpu().detach().numpy().astype(int),maxpreds_test)/len(test_data) 
                    test_min_auc_val += roc_auc_score(labels_test.cpu().detach().numpy().astype(int),minpreds_test)/len(test_data) 
                    test_loss += model.sample_elbo(signals_test, labels_test, loss_func, sample_nbr,complexity_cost_weight=cw).item()/(batch_size*len(test_data))
                    test_mean_accuracy += acc(labels_test, meanpreds_test)/len(test_data) 
                    test_min_accuracy += acc(labels_test, minpreds_test)/len(test_data) 
                    test_max_accuracy += acc(labels_test, maxpreds_test)/len(test_data) 
        
        #append values
        test_loss_vals.append(test_loss)
        test_mean_acc_vals.append(test_mean_accuracy)
        test_mean_auc_vals.append(test_mean_auc_val)
        test_min_acc_vals.append(test_min_accuracy)
        test_min_auc_vals.append(test_min_auc_val)
        test_max_acc_vals.append(test_max_accuracy)
        test_max_auc_vals.append(test_max_auc_val)
        valid_loss_vals.append(valid_loss)
        valid_mean_acc_vals.append(valid_mean_accuracy)
        valid_mean_auc_vals.append(valid_mean_auc_val)
        valid_min_acc_vals.append(valid_min_accuracy)
        valid_min_auc_vals.append(valid_min_auc_val)
        valid_max_acc_vals.append(valid_max_accuracy)
        valid_max_auc_vals.append(valid_max_auc_val)

        #update learning rate
        if epoch % 4 == 0:
            lr /= 2.0
            for g in optimizer.param_groups:
                g['lr'] = lr
        t2=time.time()
        print('Time Elapsed: {}'.format(t2-t1))
        #if epoch % display_epoch==1:
        if args.v >0:
            print("Epoch: {} - Training Loss: {:.4f} - Test Loss: {:.4f}".format(epoch,loss_val,test_loss))
            print("Training Accuracy: {:.4f} - Test Accuracy: {:.4f} - Test AUC: {:.4f}".format(acc_val,test_mean_accuracy,test_mean_auc_val))
            print("-"*40)

    print("Final Training Loss: {:.4f}".format(loss))
    print("Final Test Loss: {:.4f}".format(valid_loss))

    # compute roc curve for training data
    train_preds=np.concatenate(train_preds)
    train_roc = roc_curve(train_out.cpu(), train_preds)
    fpr_train = train_roc[0]
    tpr_train = train_roc[1]

    if testing_mode:
        test_meanpreds=np.concatenate(test_meanpreds)
        test_mean_roc = roc_curve(test_out.cpu(), test_meanpreds)
        fpr_test_mean = test_mean_roc[0]
        tpr_test_mean = test_mean_roc[1]
        test_maxpreds=np.concatenate(test_maxpreds)
        test_max_roc = roc_curve(test_out.cpu(), test_meanpreds)
        fpr_test_max = test_max_roc[0]
        tpr_test_max = test_max_roc[1]
        test_minpreds=np.concatenate(test_minpreds)
        test_min_roc = roc_curve(test_out.cpu(), test_minpreds)
        fpr_test_min = test_min_roc[0]
        tpr_test_min = test_min_roc[1]
        # Plotting ROC curves

        print('Plotting ROC curves...')
        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 8))
        #plt.plot(fpr_train, tpr_train, color='dodgerblue', label='Training')
        plt.plot(fpr_test_mean, tpr_test_mean , color='springgreen', label='Validation (Mean)')
        plt.plot(fpr_test_min, tpr_test_min, label='Validation (Lower CI)')
        plt.plot(fpr_test_max, tpr_test_max, label='Validation (Upper CI)')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc='best', fontsize=16)
        plt.savefig(results_dir + '/bnn_roc_curve_test_1.png', dpi=400)
        print('Plotted testing ROC curves.')
        plt.close()
        epochs = range(1, len(loss_vals) + 1)

        # Plotting accuracy
        plt.figure(figsize=(12, 8))
        #ax2 = sns.lineplot(x=epochs, y=train_acc_vals, color='dodgerblue', label='Training')
        ax2 = sns.lineplot(x=epochs, y=test_mean_acc_vals, color='springgreen', label='Validation (Mean)')
        ax2 = sns.lineplot(x=epochs, y=test_max_acc_vals, label='Validation (Lower CI)')
        ax2 = sns.lineplot(x=epochs, y=test_min_acc_vals, label='Validation (Upper CI)')
        ax2.set_xlabel('Epochs', fontsize=16)
        ax2.set_ylabel('Accuracy', fontsize=16)
        ax2.legend(loc='best', fontsize=16)
        plt.savefig(results_dir + '/bnn_accuracy_test_1.png', dpi=400)
        print('Plotted testing accuracy.')

        plt.close()
        plt.figure(figsize=(12, 8))
        ax1 = sns.lineplot(x=epochs, y=loss_vals, color='dodgerblue', label='Training')
        ax1 = sns.lineplot(x=epochs, y=test_loss_vals, color='springgreen', label='Validation')
        ax1.set_xlabel('Epochs', fontsize=16)
        ax1.set_ylabel('Loss', fontsize=16)
        ax1.legend(loc='best', fontsize=16)
        plt.savefig(results_dir + '/bnn_loss_test_1.png', dpi=400)
        plt.close()
        print('Plotting testing loss')

        # Plotting AUC score
        plt.figure(figsize=(12, 8))
        #ax3 = sns.lineplot(x=epochs, y=auc_vals, color='dodgerblue', label='Training')
        ax3 = sns.lineplot(x=epochs, y=test_mean_auc_vals, color='springgreen', label='Validation (Mean)')
        ax3 = sns.lineplot(x=epochs, y=test_max_auc_vals, label='Validation (Upper CI)')
        ax3 = sns.lineplot(x=epochs, y=test_min_auc_vals, label='Validation (Lower CI)')
        ax3.set_xlabel('Epochs', fontsize=16)
        ax3.set_ylabel('AUC Score', fontsize=16)
        ax3.legend(loc='best', fontsize=16)
        plt.savefig(results_dir + '/bnn_auc_score_test_1.png', dpi=400)
        print('Plotted testing AUC score.')
        print('-'*80)
        plt.close()

    # Getting validation ROC curve
    valid_meanpreds=np.concatenate(valid_meanpreds)
    valid_mean_roc = roc_curve(valid_out.cpu(), valid_meanpreds)
    fpr_valid_mean = valid_mean_roc[0]
    tpr_valid_mean = valid_mean_roc[1]
    valid_maxpreds=np.concatenate(valid_maxpreds)
    valid_max_roc = roc_curve(valid_out.cpu(), valid_meanpreds)
    fpr_valid_max = valid_max_roc[0]
    tpr_valid_max = valid_max_roc[1]
    valid_minpreds=np.concatenate(valid_minpreds)
    valid_min_roc = roc_curve(valid_out.cpu(), valid_minpreds)
    fpr_valid_min = valid_min_roc[0]
    tpr_valid_min = valid_min_roc[1]

    # Plotting ROC curves
    print('Plotting ROC curves...')
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    plt.plot(fpr_valid_mean, tpr_valid_mean , color='springgreen', label='Validation (Mean)')
    plt.plot(fpr_valid_min, tpr_valid_min, label='Validation (Lower CI)')
    plt.plot(fpr_valid_max, tpr_valid_max, label='Validation (Upper CI)')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(results_dir + '/bnn_roc_curve_2.png', dpi=400)
    print('Plotted training and validation ROC curves.')
    print('-'*80)
    plt.close()

    epochs = range(1, len(loss_vals) + 1)

    # Plotting accuracy
    plt.figure(figsize=(12, 8))
    #ax2 = sns.lineplot(x=epochs, y=train_acc_vals, color='dodgerblue', label='Training')
    ax2 = sns.lineplot(x=epochs, y=valid_mean_acc_vals, color='springgreen', label='Validation (Mean)')
    ax2 = sns.lineplot(x=epochs, y=valid_max_acc_vals, label='Validation (Lower CI)')
    ax2 = sns.lineplot(x=epochs, y=valid_min_acc_vals, label='Validation (Upper CI)')
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    ax2.legend(loc='best', fontsize=16)
    plt.savefig(results_dir + '/bnn_accuracy_2.png', dpi=400)
    print('Plotted training and validation accuracy.')
    plt.close()

    #plotting loss
    plt.figure(figsize=(12, 8))
    ax1 = sns.lineplot(x=epochs, y=loss_vals, color='dodgerblue', label='Training')
    ax1 = sns.lineplot(x=epochs, y=valid_loss_vals, color='springgreen', label='Validation')
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.legend(loc='best', fontsize=16)
    plt.savefig(results_dir + '/bnn_loss_2.png', dpi=400)
    plt.close()

    # Plotting AUC score
    plt.figure(figsize=(12, 8))
    #ax3 = sns.lineplot(x=epochs, y=auc_vals, color='dodgerblue', label='Training')
    ax3 = sns.lineplot(x=epochs, y=valid_mean_auc_vals, color='springgreen', label='Validation (Mean)')
    ax3 = sns.lineplot(x=epochs, y=valid_max_auc_vals, label='Validation (Upper CI)')
    ax3 = sns.lineplot(x=epochs, y=valid_min_auc_vals, label='Validation (Lower CI)')
    ax3.set_xlabel('Epochs', fontsize=16)
    ax3.set_ylabel('AUC Score', fontsize=16)
    ax3.legend(loc='best', fontsize=16)
    plt.savefig(results_dir + '/bnn_auc_score_2.png', dpi=400)
    print('Plotted validation AUC score.')
    print('-'*80)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------