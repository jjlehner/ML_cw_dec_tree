import typing

import matplotlib
import numpy
from matplotlib.patches import Polygon


def line(axes: matplotlib.axes,
        source: typing.Tuple[int, int],
        target: typing.Tuple[int, int],
        colour: typing.List = [0, 0, 0]) -> typing.Tuple[int, int]:

    """ Draw a line

    Arguments
    ---------
    axes: matplotlib.axes
        the axes on which to draw the text
    source: typing.Tuple[int, int]
        the start of the line segment
    target: typing.Tuple[int, int]
        the end of the line segment

    Returns
    -------
    size: typing.Tuple[int, int]
        size of the line rendered
    """

    vertices = numpy.array([source, target])
    points = Polygon(vertices, edgecolor=colour, lw=1)
    axes.add_patch(points)
