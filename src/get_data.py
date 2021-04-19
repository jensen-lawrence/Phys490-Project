# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np 
import h5py
from sklearn.utils import shuffle


# ----------------------------------------------------------------------------------------------------------------------
# Extracting data from .hdf5 files
# ----------------------------------------------------------------------------------------------------------------------

def get_data(data_path, n_signals, site='Hanford'):
    """
    Docstring goes here
    """
    assert (site == 'Hanford') or (site == 'Livingston'), 'Invalid site chosen.'
    if site == 'Hanford':
        strain = 'h1_strain'
    elif site == 'Livingston':
        strain = 'l1_strain'

    files_list = os.listdir(data_path)
    data_files = [file for file in files_list if file.split('.')[-1] in ['hdf', 'hdf5', 'h5', 'he5']]

    gw_data, noise_data = [], []
    for file in data_files:
        with h5py.File(data_path + '/' + file, 'r') as f:
            gw_data.append(np.array(f['injection_samples'][strain]))
            noise_data.append(np.array(f['noise_samples'][strain]))
        f.close()

    gw_data = np.concatenate(gw_data)
    noise_data = np.concatenate(noise_data)

    random_indices = np.random.choice(gw_data.shape[0], size=int(n_signals/2), replace=False)
    gw_data = gw_data[random_indices, :]
    noise_data = noise_data[random_indices, :]

    gw_labels = np.ones(gw_data.shape[0])
    noise_labels = np.zeros(noise_data.shape[0])

    data = np.concatenate((gw_data, noise_data))
    labels = np.concatenate((gw_labels, noise_labels))
    data, labels = shuffle(data, labels)
    
    return data, labels


# ----------------------------------------------------------------------------------------------------------------------