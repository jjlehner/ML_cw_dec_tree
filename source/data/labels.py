import numpy
import typing


def labels(dataset: numpy.ndarray) -> typing.List:
    """ Returns a list of unique labels from a dataset

    Arguments
    ---------
    dataset: numpy.ndarray
        the dataset to read the labels from
    
    Returns
    -------
    labels: typing.List
        a list of unique labels
    """

    width, _ = dataset.shape
    label_column = dataset[:, width - 1]
    return numpy.unique(label_column)