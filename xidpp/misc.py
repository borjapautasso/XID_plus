# Dont love the idea of a misc but the build pixel grid is too widespread to be within prior anymore

import numpy as np
from numpy.typing import NDArray

def build_pix_grid(map_x, map_y, values: NDArray, fill: int|float, pad: int = 0) -> tuple[NDArray, int, int]:
    """
    TODO: UPDATE DOCS, no longer within Prior
    Lays out a flattened list into a dense 2D pixel grid.

    Creates a (optionally padded) 2D grid, filled with ``fill``.
    For each map pixel position, replaces the value with the equivalent ``values``.

    Parameters
    ----------
    values : array_like
        Values to place into the grid, one per pixel in ``self.map_x``/``self.map_y``.
    fill : int or float
        Value the grid is initialised to. Pixels not covered by ``self.map_x``/``self.map_y``
        will keep their ``fill`` value.
    pad : int, optional
        Extra margin, in map pixels, to add to each side of the grid. Default is 0 (i.e. no
        padding)

    Returns
    -------
    grid :np.ndarray
        Dense 2D grid
    x_origin : int
        Map x-coordinate corresponding to column 0 of ``grid``.
    y_origin : int
        Map y-coordinate corresponding to row 0 of ``grid``.
    """
    x_origin = np.min(map_x) - pad
    y_origin = np.min(map_y) - pad

    shape = (np.max(map_y) - y_origin + pad + 1,
                np.max(map_x) - x_origin + pad + 1)

    grid = np.full(shape, fill, dtype = np.result_type(values, fill))

    grid[map_y - y_origin, map_x - x_origin] = values

    return grid, x_origin, y_origin