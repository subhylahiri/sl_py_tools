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
from sl_py_tools.arg_tricks import default, non_default_eval
from sl_py_tools.iter_tricks import slice_to_range
from sl_py_tools.containers import same_shape, identical_shape, broadcastable
from sl_py_tools.containers import ShapeTuple
from sl_py_tools.arg_tricks import Export
Export[slice_to_range, same_shape, identical_shape, broadcastable, ShapeTuple]


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


def slice_str(*sliceobjs: slice) -> str:
    """String representation of slice

    Converts `np.s_[a:b:c]` to `'a:b:c'`, `np.s_[a:b, c:]` to `'a:b, c:'`,
    `*np.s_[::c, :4]` to `'::c, :4'`, `np.s_[:]` to `':'`, etc.

    Parameters
    ----------
    sliceobj : slice
        Slice instance(s) to represent.

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    if len(sliceobjs) == 0:
        return ''
    if len(sliceobjs) > 1:
        return ', '.join(slice_str(s) for s in sliceobjs)
    sliceobjs = sliceobjs[0]
    if isinstance(sliceobjs, tuple):
        return slice_str(*sliceobjs)
    if isinstance(sliceobjs, int):
        return str(sliceobjs)
    if isinstance(sliceobjs, Ellipsis):
        return '...'
    slc_str = non_default_eval(sliceobjs.start, str, '') + ':'
    slc_str += non_default_eval(sliceobjs.stop, str, '')
    slc_str += non_default_eval(sliceobjs.step, lambda x: f':{x}', '')
    return slc_str


def slice_to_inds(the_slice: slice, size: int = 0):
    """Convert a slice object to an array of indices.

    Parameters
    ----------
    the_slice
        The `slice` to convert.
    size
        Upper limit of `range`s, used if `the_slice.stop` is `None`.

    Returns
    -------
    inds
        `np.ndarray` of indices, as produced by `np.arange` with `start`,
        `stop` and `step` taken from `the_slice`.
    """
    if isinstance(the_slice, int):
        return the_slice
    return np.arange(default(the_slice.start, 0),
                     default(the_slice.stop, size),
                     default(the_slice.step, 1), int)


class SliceInds():
    """Class for converting a slice to an array of indices.

    You can build an array of indices by calling `si_[start:stop:step]`,
    where `si_` is an instance of `SliceInds`.

    Parameters
    ----------
    size
        Upper limit of `range`s, used if `the_slice.stop` is `None`.
    """
    size: int

    def __init__(self, size: int = 0):
        """
        Parameters
        ----------
        size
            Upper limit of `range`s, used if `the_slice.stop` is `None`.
        """
        self.size = size

    def __getitem__(self, arg):
        """
        Parameters
        ----------
        the_slice
            The `slice` to convert.

        Returns
        -------
        inds
            `np.ndarray` of indices, as produced by `np.arange` with `start`,
            `stop` and `step` taken from `the_slice`.
        """
        return slice_to_inds(arg, self.size)


si_ = SliceInds()


def take_slice(array: np.ndarray, the_slice: slice, axis: int = None, **kwds):
    """Take a slice along a given axis.

    Equivalent to `np.take`, except it takes a `slice` object instead of an
    array of indices.

    See Also
    --------
    `np.take`
    """
    size = array.size
    if axis is not None:
        size = array.shape[axis]
    return np.take(array, slice_to_inds(the_slice, size), axis=axis, **kwds)


def flattish(arr: np.ndarray, start: int = 0, stop: int = None) -> np.ndarray:
    """Partial flattening.

    Flattens those axes in the range [start:stop)

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
    """
    stop = default(stop, arr.ndim)
    newshape = arr.shape[:start] + (-1,) + arr.shape[stop:]
    if len(newshape) > arr.ndim + 1:
        raise ValueError(f"start={start} > stop={stop}")
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
        Expanded array.

    See Also
    --------
    numpy.expand_dims
    """
    if len(axis) == 0:
        return arr
    if len(axis) == 1:
        return np.expand_dims(arr, axis[0]).view(type(arr))
    axes_sort = tuple(np.sort(np.mod(axis, arr.ndim + len(axis))))
    axes_same = np.nonzero(np.diff(axes_sort) == 0)
    if axes_same.size > 0:
        raise ValueError(f'repeated axes, arguments: {axes_same}')
    return expand_dims(expand_dims(arr, axes_sort[0]), *axes_sort[1:])


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
        min_dtype = non_default_eval(dtype, lambda x: (x,), ())
        min_shape = non_default_eval(
                            shape, lambda x: (np.empty(x, *min_dtype),), ())
        self.bcast = np.broadcast(*arrays, *min_shape)
        self.dtype = np.result_type(*arrays, *min_dtype)

    def __getattr__(self, name):
        return getattr(self.bcast, name)

    def __repr__(self):
        return f"BroadcastType(shape={self.shape}, dtype='{self.dtype}')"

    def __iter__(self):
        return iter(self.bcast)

    def empty_like(self, **kwds):
        """Return emtpty array of appropriate shape and dtype

        See Also
        --------
        `np.empty`, `np.empty_like`
        """
        kwds.pop('subok')
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)
        return np.empty(**kwds)

    def ones_like(self, **kwds):
        """Return array of ones of appropriate shape and dtype

        See Also
        --------
        `np.ones`, `np.ones_like`
        """
        kwds.pop('subok')
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)
        return np.ones(**kwds)

    def zeros_like(self, **kwds):
        """Return array of zeros of appropriate shape and dtype

        See Also
        --------
        `np.zeros`, `np.zeros_like`
        """
        kwds.pop('subok')
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('dtype', self.dtype)
        return np.zeros(**kwds)

    def full_like(self, fill, **kwds):
        """Return array with constant value of appropriate shape and dtype

        See Also
        --------
        `np.full`, `np.full_like`
        """
        kwds.pop('subok')
        kwds.setdefault('shape', self.shape)
        kwds.setdefault('fill', fill)
        kwds.setdefault('dtype', self.dtype)
        return np.full(**kwds)
