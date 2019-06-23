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


class BroadcastType():
    """numpy.broadcast with dtype info.

    Parameters
    ----------
    *arrays: array_like
        Arrays to be broadcast

    Keyword only
    ------------
    shape: Tuple[int] or int
        Minimum shape of result (as if it were included in broadcasting).
    dtype: dtype or dtype specifier
        Minimum scalar type of result (as if it were included in upcasting).

    Attributes
    ----------
    dtype : np.dtype
        Scalar type of result.
    bcast : np.broadcast
        Object encapsulating shape of broadcast result.
    shape, ndim, size, etc
        Obtained from `self.bcast`.

    See Also
    --------
    `np.broadcast` : used for broadcasting
    `np.result_type` : used for type promotion/upcasting.
    """
    bcast: np.broadcast
    dtype: np.dtype

    def __init__(self, *arrays, shape=None, dtype='?'):
        self.bcast = np.broadcast(*arrays, np.empty(shape))
        self.dtype = np.result_type(*arrays, dtype)

    def __getattr__(self, name):
        return getattr(self.bcast, name)

    def __repr__(self):
        return f"BroadcastType(shape={self.shape}, dtype='{self.dtype}')"

    def empty_like(self, **kwds):
        """Return emtpty array of appropriate shape and dtype

        See Also
        --------
        `np.empty`, `np.empty_like`
        """
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)
        return np.empty(**kwds)

    def ones_like(self, **kwds):
        """Return array of ones of appropriate shape and dtype

        See Also
        --------
        `np.ones`, `np.ones_like`
        """
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)
        return np.ones(**kwds)

    def zeros_like(self, **kwds):
        """Return array of zeros of appropriate shape and dtype

        See Also
        --------
        `np.zeros`, `np.zeros_like`
        """
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)
        return np.zeros(**kwds)

    def full_like(self, fill, **kwds):
        """Return array with constant value of appropriate shape and dtype

        See Also
        --------
        `np.full`, `np.full_like`
        """
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('fill', fill)
        kwds.setdefault('dtype', self.dtype)
        return np.full(**kwds)
