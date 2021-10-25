import numpy
import typing


def split(dataset: numpy.ndarray, split: float) \
        -> typing.Tuple[numpy.ndarray, numpy.ndarray]:

    """ Splits a single dataset about a split point

    Arguments
    ---------
    dataset: numpy.ndarray
        the dataset to split
    split: float
        the point to split about

    Returns
    -------
    sets: typing.Tuple[numpy.ndarray, numpy.ndarray]
        the (left, right) sets, about the split
    """

    return ([], [])
