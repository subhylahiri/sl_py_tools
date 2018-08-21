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
    This function builds up the (list of) matrices from (lists of) vectors or
    scalars.

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


class Slicify(object):
    """Store index info in variables.

    Converts colon notation to slice objects, so that it can be stored, passed
    to functions, etc. Everything else is passed through unchanged:

    >>> a = slicify[4,...,5:9]
    (4, Ellipsis, slice(5, 9, None))

    Output is always a tuple, even with only one argument:

    >>> a = slicify[:]
    (slice(None, None, None),)

    Parameters
    ----------
    *indexinfo
        Anything that can be used as a set of indices: integers, ndarray[int],
        ndarray[bool], Ellipsis, slices objects or colons & ints.

    Returns
    -------
    indexinfo
        tuple of [integers, ndarrays of int/bool, Ellipsis, slice objects].
    """
    def __getitem__(self, *indexinfo):
        return indexinfo

    def __init__():
        pass


slicify = Slicify()
