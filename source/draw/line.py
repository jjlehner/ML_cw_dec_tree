import typing

import matplotlib


def line(axes: matplotlib.axes,
        source: typing.Tuple[int, int],
        target: typing.Tuple[int, int]) -> typing.Tuple[int, int]:

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

    axes.plot([source[0], target[0]], [source[1], target[1]],
            color=[0, 0, 0],
            linewidth=1)
