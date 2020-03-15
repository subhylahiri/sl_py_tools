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
from ..containers import ShapeTuple, same_shape, identical_shape, broadcastable
from ..slice_tricks import slice_str, slice_to_range, SliceRange, srange
from ..slice_tricks import (last_value, stop_step, in_slice, is_subslice,
                            disjoint_slice)
from ..arg_tricks import default, default_non_eval, Export, args_to_kwargs

Export[slice_to_range, SliceRange, srange, slice_str]
Export[last_value, stop_step, in_slice, is_subslice, disjoint_slice]
Export[same_shape, identical_shape, broadcastable, ShapeTuple]


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

    Following `np.linalg` broadcasting, `np.stack` combines arrays of
    matrices into an array of matrices, as default `axis`=0.
    This function builds up the matrix/vector array from arrays of
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


def _posify(ind, size):
    if ind < 0:
        return ind + size
    return ind


def _negify(ind, size):
    if ind >= 0:
        return ind - size
    return ind


def slice_to_inds(the_slice: slice, size: int = 0):
    """Convert a slice object to an array of indices.

    Parameters
    ----------
    the_slice
        The `slice` to convert.
    size
        Upper limit of `range`s, used if `the_slice.stop` is `None` or negative.

    Returns
    -------
    inds
        `np.ndarray` of indices, as produced by `np.arange` with `start`,
        `stop` and `step` taken from `the_slice`.
    """
    if isinstance(the_slice, int):
        return the_slice
    return np.arange(_posify(default(the_slice.start, 0), size),
                     _posify(default(the_slice.stop, size), size),
                     default(the_slice.step, 1), int)


def take_slice(array: np.ndarray, the_slice: slice, axis: int = None, **kwds):
    """Take a slice along a given axis.

    Equivalent to `np.take`, except it takes a `slice` object instead of an
    array of indices.

    See Also
    --------
    `np.take`
    """
    size = default_non_eval(axis, lambda x: array.shape[x], array.size)
    return np.take(array, slice_to_inds(the_slice, size), axis=axis, **kwds)


def flattish(arr: np.ndarray, start: int = 0, stop: int = None) -> np.ndarray:
    """Partial flattening.

    Flattens those axes in the range `[start:stop)`. If `start == stop` it will
    insert a singleton.

    Parameters
    ----------
    arr : np.ndarray (...,L,M,N,...,P,Q,R,...)
        Array to be partially flattened.
    start : int, optional, default: 0
        First axis of group to be flattened.
    stop : int or None, optional, default: None
        First axis *after* group to be flattened. Goes to end if it is None.

    Returns
    -------
    new_arr : np.ndarray (...,L,M*N*...*P*Q,R,...)
        Partially flattened array.

    Raises
    ------
    ValueError
        If `start > stop`.
    """
    start = _posify(start, arr.ndim)
    stop = _posify(default(stop, arr.ndim), arr.ndim)
    if start > stop:
        raise ValueError(f"start={start} > stop={stop}")
    newshape = arr.shape[:start] + (-1,) + arr.shape[stop:]
    return np.reshape(arr, newshape)


def expand_dims(arr: np.ndarray, *axis) -> np.ndarray:
    """Expand the shape of the array with length one axes.

    Alias of `numpy.expand_dims` when `*axis` is a single `int`. If `axis`
    is a sequence of `int`, axis numbers are relative to the *final* shape.

    Parameters
    ----------
    arr : np.ndarray (...,L,M,N,...,P,Q,...)
        Array to be expanded.
    *axis : int
        Positions where new axes are inserted. Negative numbers are allowed.
        Numbers are with respect to the final shape.

    Returns
    -------
    new_arr : np.ndarray (...,L,1,M,1,N,...,P,1,Q,...)
        Expanded array, `new_arr.shape[ax] == 1` for all `ax in axis` and
        `new_arr.ndim = arr.ndim + len(axis)`.

    Raises
    ------
    ValueError
        If any of `axis` are repeated or not in `range(new_arr.ndim)`.

    See Also
    --------
    numpy.expand_dims
    """
    if len(axis) == 0:
        return arr
    if len(axis) == 1:
        return np.expand_dims(arr, axis[0]).view(type(arr))
    ndim = arr.ndim + len(axis)
    axes_sort = np.unique([_posify(x, ndim) for x in axis])
    if np.any(axes_sort < 0) or np.any(axes_sort >= ndim):
        raise ValueError(f'Axes out of range in: {axis}, ndim={ndim}')
    if len(axes_sort) < len(axis):
        raise ValueError(f'Repeated axes in: {axis}, ndim={ndim}')
    return expand_dims(expand_dims(arr, axes_sort[0]), *axes_sort[1:].tolist())


class BroadcastType():
    """numpy.broadcast with dtype info.

    Parameters
    ----------
    *arrays: array_like
        Arrays to be broadcast
    shape: Tuple[int] or int
        Minimum shape of result (as if it were included in broadcasting).
        Keyword only.
    dtype: dtype or dtype specifier
        Minimum scalar type of result (as if it were included in upcasting).
        Keyword only.

    Attributes
    ----------
    dtype : np.dtype
        Scalar type of result.
    bcast : np.broadcast
        Object/iterable encapsulating shape of broadcast result.
    shape, ndim, size, etc
        Obtained from `self.bcast`.

    See Also
    --------
    `np.broadcast` : used for broadcasting
    `np.result_type` : used for type promotion/upcasting.
    """
    bcast: np.broadcast
    dtype: np.dtype

    def __init__(self, *arrays, shape=None, dtype=None):
        min_dtype = default_non_eval(dtype, lambda x: (x,), ())
        min_shape = default_non_eval(
            shape, lambda x: (np.empty(x, *min_dtype),), ())
        self.bcast = np.broadcast(*arrays, *min_shape)
        self.dtype = np.result_type(*arrays, *min_dtype)

    def __getattr__(self, name):
        return getattr(self.bcast, name)

    def __repr__(self):
        return f"BroadcastType(shape={self.shape}, dtype='{self.dtype}')"

    def __iter__(self):
        return iter(self.bcast)

    def __array_function__(self, func, types, args, kwargs):
        if types:
            return NotImplemented
        if func.__name__.endswith('_like') and hasattr(self, func.__name__):
            return getattr(self, func.__name__)(*args[1:], **kwargs)
        return NotImplemented

    def _like_kwds(self, kwds):
        """Prepare kwds for ????_like method"""
        kwds.pop('subok')
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)

    def empty_like(self, *args, **kwds):
        """Return emtpty array of appropriate `shape` and `dtype`

        See Also
        --------
        `np.empty`, `np.empty_like`
        """
        args_to_kwargs(args, kwds, ['dtype', 'order', 'subok', 'shape'])
        self._like_kwds(kwds)
        return np.empty(**kwds)

    def ones_like(self, *args, **kwds):
        """Return array of ones of appropriate `shape` and `dtype`

        See Also
        --------
        `np.ones`, `np.ones_like`
        """
        args_to_kwargs(args, kwds, ['dtype', 'order', 'subok', 'shape'])
        self._like_kwds(kwds)
        return np.ones(**kwds)

    def zeros_like(self, *args, **kwds):
        """Return array of zeros of appropriate `shape` and `dtype`

        See Also
        --------
        `np.zeros`, `np.zeros_like`
        """
        args_to_kwargs(args, kwds, ['dtype', 'order', 'subok', 'shape'])
        self._like_kwds(kwds)
        return np.zeros(**kwds)

    def full_like(self, fill_value, *args, **kwds):
        """Return array with constant value of appropriate `shape` and `dtype`

        See Also
        --------
        `np.full`, `np.full_like`
        """
        kwds['fill_value'] = fill_value
        args_to_kwargs(args, kwds, ['dtype', 'order', 'subok', 'shape'])
        self._like_kwds(kwds)
        return np.full(**kwds)
