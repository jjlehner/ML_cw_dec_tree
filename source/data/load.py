import numpy
import os


def load_dataset(path: str) -> numpy.ndarray:
    """ Loads a data set from path

    Arguments
    ---------
    path: str
        path of the file to load

    Returns
    -------
    output: numpy.ndarray
        dimensional array containing dataset information
    """
    return numpy.loadtxt(path)


def load_clean() -> numpy.ndarray:
    """ Loads the clean dataset

    Returns
    -------
    clean_data: numpy.ndarray
        the clean data set
    """
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, 'wifi_db', 'clean_dataset.txt')
    return load_dataset(path)


def load_noisy() -> numpy.ndarray:
    """ Loads the noisy dataset

    Returns
    -------
    noisy_data: numpy.ndarray
        the noisy data set
    """
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, 'wifi_db', 'noisy_dataset.txt')
    return load_dataset(path)
