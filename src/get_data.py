# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import h5py
import numpy as np 
from sklearn.utils import shuffle


# ----------------------------------------------------------------------------------------------------------------------
# Extracting data from .hdf5 files
# ----------------------------------------------------------------------------------------------------------------------

def get_data(data_path, n_signals, site='Hanford'):
    """
    Using the gravitational wave simulation data in the .hdf files at data_path, get_data returns an array of length
    n_signals containing a 50/50 split of gravitational wave signals and noise. A corresponding array of labels is also
    returned, where 1 represents gravitational waves and 0 represents noise.
    """
    # Determining strain value
    assert (site == 'Hanford') or (site == 'Livingston'), 'Invalid site chosen.'
    if site == 'Hanford':
        strain = 'h1_strain'
    elif site == 'Livingston':
        strain = 'l1_strain'

    # Finding all .hdf files in the specified path
    files_list = os.listdir(data_path)
    data_files = [file for file in files_list if file.split('.')[-1] in ['hdf', 'hdf5', 'h5', 'he5']]

    # Extracting gravitational wave and noise data from .hdf files
    gw_data, noise_data = [], []
    for file in data_files:
        with h5py.File(data_path + '/' + file, 'r') as f:
            gw_data.append(np.array(f['injection_samples'][strain]))
            noise_data.append(np.array(f['noise_samples'][strain]))
        f.close()

    # Combining all gravitational wave and noise data into single arrays
    gw_data = np.concatenate(gw_data)
    noise_data = np.concatenate(noise_data)

    # Randomly selecting samples from gravitational wave and noise data
    random_indices = np.random.choice(gw_data.shape[0], size=int(n_signals/2), replace=False)
    gw_data = gw_data[random_indices, :]
    noise_data = noise_data[random_indices, :]

    # Creating labels for gravitational wave and noise data
    gw_labels = np.ones(gw_data.shape[0])
    noise_labels = np.zeros(noise_data.shape[0])

    # Combining data and labels into single arrays
    data = np.concatenate((gw_data, noise_data))
    labels = np.concatenate((gw_labels, noise_labels))

    # Randomizing data and label arrays
    data, labels = shuffle(data, labels)

    return data, labels


# ----------------------------------------------------------------------------------------------------------------------