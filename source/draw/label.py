import typing

import matplotlib
import numpy

from matplotlib.textpath import TextPath
from matplotlib.path import Path
from matplotlib import patches

# Height of all labels rendered, in points
_height = 0.5


def label(axes: matplotlib.axes,
        origin: typing.Tuple[int, int],
        text: str,
        colour: typing.List = [1, 1, 1]) -> typing.Tuple[int, int]:

    """ Draw a label

    This function, unlike matplotlib.text(), draws text as a rendered series
    of polygon segments. This means it scales and zooms appropriately, rather
    than by staying a fixed size relative to the screen.

    Arguments
    ---------
    axes: matplotlib.axes
        the axes on which to draw the text
    origin: typing.Tuple[int, int]
        the origin about which to draw the text
    text: str
        the text to write

    Returns
    -------
    size: typing.Tuple[int, int]
        size of the label rendered
    """

    # Generate an array of points for a given string
    path = TextPath((0, 0), text, size=1)

    # Evaluate the glyph's size in terms of its bottom-left and top-right
    # corners
    bottom_left = numpy.amin(numpy.array(path.vertices), axis=0)
    top_right = numpy.amax(numpy.array(path.vertices), axis=0)

    # Calculate the text's scale and size
    # scale = numpy.subtract(top_right, bottom_left)
    # size = [_height / numpy.prod(scale), _height]
    size = [0, _height]
    scale = [0, 0]
    scale[0] = (top_right[0] - bottom_left[0])
    scale[1] = (top_right[1] - bottom_left[1])
    size[0] = size[1] / scale[1] * scale[0]

    # Scale each vertex's position relative to the origin
    vertices = []
    for vertex in path.vertices:

        # Calculate the text's scaled position
        position = numpy.multiply(vertex, size)
        position = numpy.divide(position, scale)
        position = numpy.add(position, origin)

        # Centre the text about the origin
        offset = numpy.divide(size, [2, 2])
        position = numpy.subtract(position, offset)

        vertices.append(position)

    # Create a patch from the vertex points
    path = Path(numpy.array(vertices), path.codes)
    patch = patches.PathPatch(path, color=colour, lw=0, zorder=10)
    axes.add_patch(patch)

    return size
