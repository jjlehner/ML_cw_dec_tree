import numpy
import os


def _load_data_set(dataset_name: str) -> numpy.ndarray:
    """ Loads a data set from /wifi_db

    Arguments
    ---------
    dataset_name: str
        name of the file to load
    
    Returns
    -------
    output: numpy.ndarray
        dimensional array containing dataset information
    """

    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, 'wifi_db', dataset_name)
    return numpy.loadtxt(path)


def load_clean() -> numpy.ndarray:
    """ Loads the clean dataset

    Returns
    -------
    clean_data: numpy.ndarray
        the clean data set 
    """

    return _load_data_set('clean_dataset.txt')


def load_noisy() -> numpy.ndarray:
    """ Loads the noisy dataset

    Returns
    -------
    noisy_data: numpy.ndarray
        the noisy data set 
    """
    return _load_data_set('noisy_dataset.txt')