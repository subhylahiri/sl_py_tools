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
import typing as ty

import numpy as np

import sl_py_tools.numpy_tricks.subclass as sc

from ..arg_tricks import Export, args_to_kwargs, default, eval_or_default
from ..containers import ShapeTuple, broadcastable, identical_shape, same_shape
from ..slice_tricks import (SliceRange, disjoint_slice, in_slice, is_subslice,
                            slice_str, slice_to_range, srange)

_EXPORTED = Export[same_shape, identical_shape, broadcastable, ShapeTuple]
_EXPORTED = Export[slice_to_range, SliceRange, srange, slice_str]
_EXPORTED = Export[in_slice, is_subslice, disjoint_slice]

# =============================================================================


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


def _posify(ind: int, size: int):
    if ind < 0:
        return ind + size
    return ind


def _negify(ind: int, size: int):
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
        Upper limit of `range`s. Used if `the_slice.stop` is `None` or negative
        or if `the_slice.start` is negative.

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


def take_slice(arr: np.ndarray, the_slice: slice, axis: int = None, **kwds):
    """Take a slice along a given axis.

    Equivalent to `np.take`, except it takes a `slice` object instead of an
    array of indices.

    See Also
    --------
    `np.take`
    """
    size = eval_or_default(axis, lambda x: arr.shape[x], arr.size)
    return np.take(arr, slice_to_inds(the_slice, size), axis=axis, **kwds)


def last_ind(arr: np.ndarray, axis: ty.Optional[int] = None,
             out: ty.Optional[np.ndarray] = None,
             **kwds) -> ty.Union[int, np.ndarray]:
    """Last index where boolean array is true

    Other keywords passed to `np.argmax`.

    Parameters
    ----------
    arr : np.ndarray[(...,N), bool]
        Boolean area to find last true value
    axis : int|None, optional
        Axis along which to find last true value, by default `None` -> use
        flattened array.
    out : np.ndarray[(...,), int]|None, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype.

    Returns
    -------
    ind : int|np.ndarray[(...,), int]
        Larges index such that `arr` is true.
    """
    size = arr.size if axis is None else arr.shape[axis]
    return size - np.argmax(np.flip(arr.astype(bool), axis), axis, out, **kwds)


def ravelaxes(arr: np.ndarray, start: int = 0, stop: int = None) -> np.ndarray:
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


def unravelaxis(arr: np.ndarray, axis: int, shape: ty.Tuple[int, ...]
                ) -> np.ndarray:
    """Partial unflattening.

    Folds an `axis` into `shape`.

    Parameters
    ----------
    arr : np.ndarray (...,L,M*N*...*P*Q,R,...)
        Array to be partially folded.
    axis : int
        Axis to be folded.
    shape : Tuple[int, ...]
        Shape to fold `axis` into. One element can be -1, like `numpy.reshape`.

    Returns
    -------
    new_arr : np.ndarray (...,L,M,N,...,P,Q,R,...)
        Partially unflattened array.

    Raises
    ------
    ValueError
        If multiple elements of `shape` are -1.
        If `arr.shape[axis] != prod(shape)` (unless one element is -1).
    """
    minus_one = np.count_nonzero([siz == -1 for siz in shape])
    if minus_one > 1:
        raise ValueError(f"Axis size {arr.shape[axis]} cannot fold to {shape}")
    if minus_one == 0 and np.prod(shape) != arr.shape[axis]:
        raise ValueError(f"Axis size {arr.shape[axis]} cannot fold to {shape}")
    axis = _posify(axis, arr.ndim)
    newshape = arr.shape[:axis] + shape + arr.shape[axis+1:]
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
    return np.expand_dims(arr, axis)


BCAST_FNS = {}
# pylint: disable=invalid-name
implements = sc.make_implements_decorator(BCAST_FNS)


def _minimal(shape, *args, **kwds) -> np.ndarray:
    """Make a minimal array of the desired shape
    """
    return np.broadcast_to(np.empty((), *args, **kwds), shape)


class BroadcastType:
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
        _dtype = eval_or_default(dtype, lambda x: (x,), ())
        _shape = eval_or_default(shape, lambda x: (_minimal(x, *_dtype),), ())
        self.bcast = np.broadcast(*arrays, *_shape)
        self.dtype = np.result_type(*arrays, *_dtype)

    def __getattr__(self, name):
        return getattr(self.bcast, name)

    def __repr__(self):
        return f"BroadcastType(shape={self.shape}, dtype='{self.dtype}')"

    def __iter__(self):
        return iter(self.bcast)

    def __array_function__(self, func, types, args, kwds):
        return sc.array_function_help(self, BCAST_FNS, func, types, args, kwds)


def _like_kwds(obj: BroadcastType, args, kwds):
    """Prepare kwds for ????_like function"""
    args_to_kwargs(args, kwds, ['dtype', 'order', 'subok', 'shape'])
    kwds.pop('subok')
    kwds.setdefault('shape', obj.shape)
    kwds.setdefault('dtype', obj.dtype)


@implements(np.empty_like)
def empty_like(obj: BroadcastType, *args, **kwds) -> np.ndarray:
    """Return emtpty array of appropriate `shape` and `dtype`

    See Also
    --------
    `np.empty`, `np.empty_like`
    """
    _like_kwds(obj, args, kwds)
    return np.empty(**kwds)


@implements(np.ones_like)
def ones_like(obj: BroadcastType, *args, **kwds) -> np.ndarray:
    """Return array of ones of appropriate `shape` and `dtype`

    See Also
    --------
    `np.ones`, `np.ones_like`
    """
    _like_kwds(obj, args, kwds)
    return np.ones(**kwds)


@implements(np.zeros_like)
def zeros_like(obj: BroadcastType, *args, **kwds) -> np.ndarray:
    """Return array of zeros of appropriate `shape` and `dtype`

    See Also
    --------
    `np.zeros`, `np.zeros_like`
    """
    _like_kwds(obj, args, kwds)
    return np.zeros(**kwds)

@implements(np.full_like)
def full_like(obj: BroadcastType, fill_value, *args, **kwds) -> np.ndarray:
    """Return array with constant value of appropriate `shape` and `dtype`

    See Also
    --------
    `np.full`, `np.full_like`
    """
    kwds['fill_value'] = fill_value
    _like_kwds(obj, args, kwds)
    return np.full(**kwds)
