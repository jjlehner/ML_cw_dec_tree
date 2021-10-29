import typing

import matplotlib
import numpy

from matplotlib import patches


def box(axes: matplotlib.axes,
        origin: typing.Tuple[int, int],
        size: typing.Tuple[int, int]) -> typing.Tuple[int, int]:

    """ Draw a box

    Arguments
    ---------
    axes: matplotlib.axes
        axes on which to draw the box
    origin: typing.Tuple[int, int]
        origin about which to draw the box
    size: typing.Tuple[int, int]
        size of the box to draw

    Returns
    -------
    size: typing.Tuple[int, int]
        size of the box rendered
    """

    # Offset the box's position by half its size, to centre it about the origin
    offset = numpy.divide(size, [2, 2])
    position = numpy.subtract(origin, offset)

    # Draw the box
    box = patches.Rectangle(position,
            size[0],
            size[1],
            fill=True,
            color=[0, 0, 0])
    axes.add_patch(box)

    return size
