# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np 
import h5py


# ----------------------------------------------------------------------------------------------------------------------
# Array Operations
# ----------------------------------------------------------------------------------------------------------------------

def _label_array(array, label):
    """
    _label_array : numpy.array, float -> numpy.array
        Takes an array of arrays, where all the sub-arrays have the same length, and appends the specified label to
        each sub-array. Returns an array of labelled sub-arrays.

    array : numpy.array
        The array of arrays to be labelled.

    label : float
        The label to be appended to each array.
    """
    assert [i.size for i in array] == [array[0].size] * array.shape[0], 'All arrays must have the same length'
    classified = np.zeros((array.shape[0], array.shape[1] + 1))
    classified[:,:-1] = array
    classified[:,-1] = label
    return classified


def _data_label_split(array):
    """
    _data_label_split : numpy.array -> numpy.array, numpy.array
        Splits an array of arrays into two arrays, where the elements of the first array are the arrays from the
        original array with their final elements removed, and the elements of the second array are the final elements
        of each of the arrays in the original array. Returns the array of non-final element arrays and the array of
        final elements.

    array : numpy.array
        The array of arrays to be split.
    """
    data = np.zeros((array.shape[0], array.shape[1] - 1))
    labels = np.zeros((array.shape[0], 1))
    data = array[:,:-1]
    labels = array[:,-1]
    return data, labels


def _two_array_shuffle(array1, array2):
    """
    _two_array_shuffle : numpy.array, numpy.array -> numpy.array
        Takes two arrays of arrays, where all the sub-arrays have the same length, and randomly shuffles their
        elements into one combined array. Returns the combined array.

    array1 : numpy.array
        The first array used in the shuffling. Size m x n.

    array2 : numpy.array
        The second array used in the shuffling. Size k x n.
    """
    assert array1.shape[1] == array2.shape[1], 'Array elements must have the same length'
    L1, L2 = array1.shape[0], array2.shape[0]
    array1_count, array2_count = 0, 0

    shuffled = np.zeros((L1 + L2, array1.shape[1]))
    shuffled_count = 0

    np.random.shuffle(array1)
    np.random.shuffle(array2)
    
    for i in range(L1 + L2):
        if (array1_count > L1) or (array2_count > L2):
            break
            
        r = np.random.randint(1, 3)
        if r == 1:
            if array1_count < L1:
                shuffled[shuffled_count] = array1[array1_count]
            else:
                shuffled[shuffled_count:] = array2[array2_count:]
            array1_count += 1

        elif r == 2:
            if array2_count < L2:
                shuffled[shuffled_count] = array2[array2_count]
            else:
                shuffled[shuffled_count:] = array1[array1_count:]
            array2_count += 1
            
        shuffled_count += 1
        
    return shuffled


# ----------------------------------------------------------------------------------------------------------------------
# Data Extraction
# ----------------------------------------------------------------------------------------------------------------------

def _extract_data(hdf5_file, site):
    """
    _extract_data : str, str -> numpy.array, numpy.array
        Extracts gravitational wave signal data and noise data from a .hdf5 file containing gravitational wave
        simulation data produced by ggwd. Returns the array containing the gravitational wave signal data and the
        array containing the noise data.

    hdf5_file : str
        Path to the .hdf5 file containing the gravitational wave simulation data.

    site : str
        Specifies the site whose signals will be used. Options are 'Hanford' and 'Livingston'.
    """
    assert (site == 'Hanford') or (site == 'Livingston'), 'Invalid site chosen.'
    if site == 'Hanford':
        strain = 'h1_strain'
    elif site == 'Livingston':
        strain = 'l1_strain'

    with h5py.File(hdf5_file, 'r') as f:
        gw_data = np.array(f['injection_samples'][strain])
        noise_data = np.array(f['noise_samples'][strain])
    f.close()
    return gw_data, noise_data


def get_data(hdf5_file, site='Hanford'):
    """
    get_nn_data : str, str -> numpy.array, numpy.array
        Extracts the signal data from a .hdf5 file containing gravitational wave simulation data produced by ggwd, and
        converts the data to a form usable for training neural networks. Returns an array containing the signal data
        and an array containing the label corresponding to each signal (0 for noise, 1 for gravitational wave).

    hdf5_file : str
        Path to the .hdf5 file containing the gravitational wave simulation data.

    site : str
        Specifies the site whose signals will be used. Options are 'Hanford' and 'Livingston'.
    """
    gw_data, noise_data = _extract_data(hdf5_file, site)
    labelled_gw_data = _label_array(gw_data, 1)
    labelled_noise_data = _label_array(noise_data, 0)
    signal_data = _two_array_shuffle(labelled_gw_data, labelled_noise_data)
    signals, labels = _data_label_split(signal_data)
    return signals, labels


# ----------------------------------------------------------------------------------------------------------------------