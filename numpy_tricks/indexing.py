# -*- coding: utf-8 -*-
# =============================================================================
# Created on Sun Sep  3 13:35:52 2017
#
# @author: Subhy
#
# module: index_tricks
# =============================================================================
"""
Tools for messing with array shapes
"""

import numpy as np
from sl_py_tools.containers import (same_shape, identical_shape, broadcastable,
                                    ShapeTuple)
assert all((same_shape, identical_shape, broadcastable, ShapeTuple))


def mesh_stack(*arrays):
    """
    Expands and combines vectors to a single multidimensional array, with one
    axis for each array, stacked across first (extra) axis.
    Each layer (wrt last axis) constant along all axes except its own.

    Parameters
    ----------
    *arrays
        1D ndarrays, shapes ((N1,),(N2,),(N3,),...(Nk,))

    Returns
    -------
    new_array
        (k+1)D ndarray, of shape (k,N1,N2,N3,...,Nk)
    """
    return np.stack(np.broadcast_arrays(*np.ix_(*arrays)))


def mesh_flat(*arrays):
    """
    Expands and combines vectors to a single two-dimensional array.
    Each row constant along all strides except its own.

    Parameters
    ----------
    *arrays
        1D ndarrays, shapes ((N1,),(N2,),(N3,),...(Nk,))

    Returns
    -------
    new_array
        2D ndarray, of shape (k,N1*N2*N3*...*Nk)
    """
    return mesh_stack(*arrays).reshape((len(arrays), -1))


def estack(arrays, axis=-1):
    """Same as `np.stack`, except default `axis`=-1

    According to `np.linalg` broadcasting, `np.stack` combines (lists of)
    matrices into a list of matrices (as default `axis`=0).
    This function builds up the (list of) matrices/vectors from (lists of)
    vectors/scalars.

    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional, default=-1
        The axis in the result array along which the input arrays are stacked.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    np.stack, np.concatenate
    """
    return np.stack(arrays, axis=axis)


def flattish(array, start, stop):
    """Flatten a subset of axes

    Takes axes in the range [start:stop) and flattens them into one axis.

    Parameters
    ----------
    array
        ndarray to be partially flattened, n-dimensional
    start
        axis at which to start flattening (inclusive)
    stop
        axis at which to stop flattening (exclusive)

    Returns
    -------
    flattish_array
        array with flattened axes, (n-stop+start+1)-dimensional
    """
    return array.reshape(array.shape[:start] + (-1,) + array.shape[stop:])


def expand_dims(array, axis):
    """Expand the shape of the array

    Alias of numpy.expand_dims.
    If `axis` is a sequence, axes are added one at a time, left to right.
    The numbering is wrt the shape at each axis addition.

    Parameters
    ----------
    array
        ndarray to be expanded
    axis
        Position of added axis. If it is a tuple, axes are added one at a time,
        left to right. The numbering is wrt the shape at each axis addition.
    """
    if isinstance(axis, int):
        return np.expand_dims(array, axis).view(type(array))
    elif not isinstance(axis, tuple):
        raise TypeError("axis must be an int or a tuple of ints")
    elif len(axis) == 0:
        return array
    return expand_dims(expand_dims(array, axis[0]), axis[1:])
