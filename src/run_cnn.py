# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# General imports
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Machine learning imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

# Custom imports
from get_data import get_data
from cnn import CNN

# ----------------------------------------------------------------------------------------------------------------------
# Implementation of Convolution Neural Network
# ----------------------------------------------------------------------------------------------------------------------

# Learning rate callback
def learning_rate_callback(epoch, learn_rate):
    """
    Callback function for reducing learning rate during training.
    """
    if epoch % 20 == 0:
        return learn_rate/2
    else:
        return learn_rate

# Get training and validation data for the CNN
def get_cnn_train_valid(train_data, n_train, n_valid):
    """
    Using the gravitational wave simulation data in the .hdf files at the path train_data, a set of training data and
    labels of length n_train and a set of validation data and labels of length n_valid is returned.
    """
    X, y = get_data(train_data, n_train + n_valid)
    X = normalize(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=n_valid)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
    y_train = y_train.reshape(y_train.size, 1, 1)
    y_valid = y_valid.reshape(y_valid.size, 1, 1)
    input_shape = X_train.shape[1:]
    return X_train, X_valid, y_train, y_valid, input_shape

# Get testing data for the CNN
def get_cnn_test(test_data, n_test):
    """
    Using the gravitational wave simulation data in the .hdf files at the path test_data, a set of testing data and
    labels oflength n_test is returned.
    """
    X, y = get_data(test_data, n_test)
    X = normalize(X)
    X_test = X.reshape(X.shape[0], X.shape[1], 1)
    y_test = y.reshape(y.size, 1, 1)
    return X_test, y_test

# Per-class accuracy
def per_class_accuracy(labels, predictions, class_label):
    assert len(labels) == len(predictions), 'Mismatched label and prediction size.'
    class_label_idx = np.where(labels==class_label)
    preds_at_idx = predictions[class_label_idx]
    correct = len([i for i in preds_at_idx if i == class_label])
    return correct/len(class_label_idx[0])

# Train, validate, and test the CNN
def run_cnn(params, train_data, v, test_data=''):
    """
    Using the hyperparameters in params, run_cnn trains and validates the CNN in cnn.py using the data in train_data.
    Output verbosity is determined by v. If test_data is provided, the model evaluates its performance on the data
    in test_data.
    """
    # Extracting CNN hyperparameters from source file
    with open(params) as f:
        nn_params = json.load(f)
    f.close()

    n_train = nn_params['n_train']
    n_valid = nn_params['n_valid']
    dropout = nn_params['dropout']
    learn_rate = nn_params['learn_rate']
    n_epochs = nn_params['n_epochs']
    batch_size = nn_params['batch_size']

    # Getting training and validation data
    X_train, X_valid, y_train, y_valid, input_shape = get_cnn_train_valid(train_data, n_train, n_valid)

    # Initializing convolutional neural network, loss function, optimizer, and performance metrics
    model = CNN(input_shape, dropout)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = Adam(learning_rate=learn_rate)
    metrics = ['accuracy', 'AUC']
    
    for i, layer in enumerate(model.layers):
        tf.summary.histogram('layer{0}'.format(i), layer)

    # Initializing model callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    lr_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_callback)

    # Training and validating model
    print('-'*80)
    print('Training convolutional neural network...')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    training_results = model.fit(
        x = X_train,
        y = y_train,
        epochs = n_epochs,
        batch_size = batch_size,
        verbose = v,
        validation_data = (X_valid, y_valid),
        callbacks = [tensorboard_callback, lr_callback]
    )

    # Determining final training and validation predictions
    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)

    # Reshaping true and predicted labels
    y_train = y_train.reshape(y_train.shape[0])
    train_pred = train_pred.reshape(train_pred.shape[0])
    y_valid = y_valid.reshape(y_valid.shape[0])
    valid_pred = valid_pred.reshape(valid_pred.shape[0])

    # Determining per-class accuracy
    train_noise_acc = per_class_accuracy(y_train, np.around(train_pred), 0)
    train_gw_acc = per_class_accuracy(y_train, np.around(train_pred), 1)
    valid_noise_acc = per_class_accuracy(y_valid, np.around(valid_pred), 0)
    valid_gw_acc = per_class_accuracy(y_valid, np.around(valid_pred), 1)

    # Printing results
    print('-'*80)
    print('TRAINING RESULTS')
    print('Training:')
    print(f'- Loss: {training_results.history["loss"][-1]:.4f}')
    print(f'- Noise accuracy: {train_noise_acc:.4f}')
    print(f'- Gravitational wave accuracy: {train_gw_acc:.4f}')
    print(f'- Total accuracy: {training_results.history["accuracy"][-1]:.4f}')
    print(f'- AUC score: {training_results.history["auc"][-1]:.4f}')
    print('Validation:')
    print(f'- Loss: {training_results.history["val_loss"][-1]:.4f}')
    print(f'- Noise accuracy: {valid_noise_acc:.4f}')
    print(f'- Gravitational wave accuracy: {valid_gw_acc:.4f}')
    print(f'- Total accuracy: {training_results.history["val_accuracy"][-1]:.4f}')
    print(f'- AUC score: {training_results.history["val_auc"][-1]:.4f}')
    print('-'*80)

    # Testing if provided with testing data
    if test_data != '':
        n_test = nn_params['n_test']
        X_test, y_test = get_cnn_test(test_data, n_test)

        print('Testing convolutional neural network...')
        test_results = model.evaluate(X_test, y_test, batch_size=batch_size)
        print('-'*80)
        print('TESTING RESULTS')
        print(f'Loss: {test_results[0]:.4f}')
        print(f'Accuracy: {test_results[1]:.4f}')
        print(f'AUC score: {test_results[2]:.4f}')
        print('-'*80)

    return training_results, y_train, train_pred, y_valid, valid_pred

# ----------------------------------------------------------------------------------------------------------------------
# Plotting Results
# ----------------------------------------------------------------------------------------------------------------------

# Plotting CNN performance metrics
def plot_cnn_metrics(training_results, save_as):
    """
    Generates plots of the loss, accuracy, and AUC scores for the training and validation results from training the CNN.
    """
    # Getting performance metrics
    train_loss = training_results.history['loss']
    train_acc = training_results.history['accuracy']
    train_auc = training_results.history['auc']
    valid_loss = training_results.history['val_loss']
    valid_acc = training_results.history['val_accuracy']
    valid_auc = training_results.history['val_auc']

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
    plt.savefig(save_as + 'loss.png', dpi=400)
    print('Plotted training and validation loss.')

    # Plotting accuracy
    plt.figure(figsize=(12, 8))
    ax2 = sns.lineplot(x=epochs, y=train_acc, color='dodgerblue', label='Training')
    ax2 = sns.lineplot(x=epochs, y=valid_acc, color='springgreen', label='Validation')
    ax2.set_xlabel('Epochs', fontsize=16)
    ax2.set_ylabel('Accuracy', fontsize=16)
    ax2.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'accuracy.png', dpi=400)
    print('Plotted training and validation accuracy.')

    # Plotting AUC score
    plt.figure(figsize=(12, 8))
    ax3 = sns.lineplot(x=epochs, y=train_auc, color='dodgerblue', label='Training')
    ax3 = sns.lineplot(x=epochs, y=valid_auc, color='springgreen', label='Validation')
    ax3.set_xlabel('Epochs', fontsize=16)
    ax3.set_ylabel('AUC Score', fontsize=16)
    ax3.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'auc_score.png', dpi=400)
    print('Plotted training and validation AUC score.')
    print('-'*80)
    plt.close()

# Plotting CNN ROC curves
def plot_cnn_roc(y_train, train_pred, y_valid, valid_pred, save_as):
    """
    Generates a plot of the ROC curve for the training and validation results from training the CNN.
    """
    # Getting training ROC curve
    train_roc = roc_curve(y_train, train_pred)
    fpr_train = train_roc[0]
    tpr_train = train_roc[1]

    # Getting validation ROC curve
    valid_roc = roc_curve(y_valid, valid_pred)
    fpr_valid = valid_roc[0]
    tpr_valid = valid_roc[1]

    # Plotting ROC curves
    print('Plotting ROC curves...')
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 8))
    plt.plot(fpr_train, tpr_train, color='dodgerblue', label='Training')
    plt.plot(fpr_valid, tpr_valid, color='springgreen', label='Validation')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    plt.savefig(save_as + 'roc_curve.png', dpi=400)
    print('Plotted training and validation ROC curves.')
    print('-'*80)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# Main Convolutional Neural Network Function
# ----------------------------------------------------------------------------------------------------------------------

# Main CNN function
def cnn_main(params, train_data, v, results_dir, test_data=''):
    """
    Main function for training, validating, and testing the CNN, and then plotting the results.
    """
    # Training, validating, and testing CNN
    training_results, y_train, train_pred, y_valid, valid_pred = run_cnn(params, train_data, v, test_data=test_data)

    # Plotting performance metrics
    save_as = results_dir + '/' + train_data.split('/')[-1] + '_'
    plot_cnn_metrics(training_results, save_as)

    # Plotting ROC curve
    plot_cnn_roc(y_train, train_pred, y_valid, valid_pred, save_as)

# ----------------------------------------------------------------------------------------------------------------------