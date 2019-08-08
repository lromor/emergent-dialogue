# coding: utf-8

"""Image with shapes generation module.

This module contains simple classes
to generate an image containing a set of
simple shapes of different size and position.
The images are subdivided in a grid where
each cell is the basic positional unit.
"""

from enum import Enum
from enum import IntEnum
from typing import List, Tuple, Optional

import numpy as np


class Shape(Enum):
    CIRCLE = 1
    SQUARE = 2
    TRIANGLE = 3


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Size(IntEnum):
    SMALL = 1
    BIG = 2


def _get_shape_unicode(shape, color, size):
    shape = Shape(shape)
    color = Color(color)
    size = Size(size)

    chars = {
        Size.SMALL: {
            Shape.CIRCLE: "•",
            Shape.SQUARE: "▪",
            Shape.TRIANGLE: "▴"
        },
        Size.BIG: {
            Shape.CIRCLE: "●",
            Shape.SQUARE: "■",
            Shape.TRIANGLE: "▲"
        }
    }
    colors = {
        Color.RED: "\033[31m",
        Color.GREEN: "\033[32m",
        Color.BLUE: "\033[34m"
    }
    return colors[color] + chars[size][shape] + "\033[0m"


class ShapesImage():
    """Represents a single datapoint of the data-set.
    In this case it represents an image containing different
    sprites of various size, shape and color in a grid.
    """

    SHAPES_CLS = Shape
    COLOR_CLS = Color
    SIZE_CLS = Size

    # Rendering constants:
    # Size of the "big" radius shape
    # within a cell
    BIG_RADIUS = 0.75 / 2

    # Size of the "small" radius shape
    # within a cell
    SMALL_RADIUS = 0.5 / 2

    def __init__(self, data: np.ndarray):
        """Initialize an image containing some shapes in a grid.

        Arguments:
        - data: a numpy array of 3 channels containing
                the features channels (shape, color, size)
        """
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    @classmethod
    def create_random_image(cls, grid_size: Tuple[int, int]=(3, 3),
                            nshapes: int=1, seed: Optional[int] = None):
        w, h = grid_size
        ncells = w * h

        r = np.random.RandomState(seed)

        if nshapes > ncells:
            raise RuntimeError("The number of shapes of is greater than"
                               "the number of cells.")

        # Our image with 3 channels (shape, color, size)
        data = np.zeros((w, h, 3), np.int8)

        # Fill the grid with nshapes shapes
        for i in r.choice(ncells, nshapes, replace=False):
            # x is the cell coordinate starting from left
            x = i % w
            # y is the cell coordinate starting from top
            y = i // w

            # Shape channel
            data[x, y, :] = (
                r.choice(cls.SHAPES_CLS).value,
                r.choice(cls.COLOR_CLS).value,
                r.choice(cls.SIZE_CLS)
            )
        return cls(data)

    def _render_unicode(self):
        """Just a fancy unicode art representation
        of the image.
        """
        w, h, _ = self.shape
        s = "\n┏" +"━┯" * (w - 1)  + "━┓\n"
        for r in range(h):
            s += "┃"
            for c in range(w):
                shape, color, size = self._data[r, c, :]
                value = " "
                if shape:
                    value = _get_shape_unicode(shape, color, size)
                end_c = "│" if c < (w - 1) else "┃"
                s+= value + end_c

            s += "\n"
            if r < (h - 1):
                s += "┠" +"─┼" * (w - 1)  + "─┨\n"
        return s + "┗" +"━┷" * (w - 1)  + "━┛\n"

    def __str__(self):
        return self._render_unicode()

    def to_rgb(self, size: Tuple[int, int]):
        """Returns a numpy RGB array image"""
        raise NotImplementedError

    def to_symb(self):
        """Return the symbolic representation of the data-point"""
        return self._data


if __name__ == "__main__":
    image = ShapesImage.create_random_image(nshapes=4, seed=None)
    print(str(image))
