# -*- coding: utf-8 -*-
"""Utilities for module markov_param
"""
from __future__ import annotations

import typing as _ty
from collections.abc import Sequence

import numpy as np

from numpy_linalg import lnarray as array
from numpy_linalg import zeros

# from numpy import ndarray as array
# from numpy import zeros


def diff_like(fun: _ty.Callable[[ArrayType, ArrayType], ArrayType],
              arr: ArrayType, step: int = 1, axis: int = -1) -> ArrayType:
    """Perform an operation on adjacent elements in an array

    Parameters
    ----------
    fun : callable
        Function to perform on elements, as in `fun(arr[i + step], arr[i])`.
    arr : array (...,n,...)
        array to perform operation on
    step : int, optional
        Perform operation on elemnts `step` apart, by default: 1.
    axis : int, optional
        Elements are separated by `step` along this axis, by default: -1.

    Returns
    -------
    out_arr : array (...,n-step,...)
        Output of `fun` for each pair of elements.
    """
    arr = np.moveaxis(arr, axis, -1)
    if step > 0:
        out_arr = fun(arr[..., step:], arr[..., :-step])
    elif step < 0:
        out_arr = fun(arr[..., :step], arr[..., -step:])
    else:
        out_arr = fun(arr, arr)
    return np.moveaxis(out_arr, -1, axis)


# =============================================================================
# Fixing non-parameter parts
# =============================================================================


def stochastify_c(mat: np.ndarray):  # make cts time stochastic
    """
    Make a matrix the generator of a continuous time Markov process.
    Changes diagonal to make row sums zero.
    **Modifies** in place, **does not** return.

    Parameters
    ----------
    mat : la.lnarray (...,n,n)
        square matrix with non-negative off-diagonal elements.
        **Modified** in place.
    """
    mat -= mat.sum(axis=-1, keepdims=True) * np.identity(mat.shape[-1])


def stochastify_pd(mat: np.ndarray):  # make dscr time stochastic
    """
    Make a matrix the generator of a discrete time Markov process.
    Shifts diagonals to make row sums one.

    Parameters
    ----------
    mat : la.lnarray (...,n,n)
        square matrix with non-negative elements.
        **Modified** in place
    """
    mat += (1 - mat.sum(axis=-1, keepdims=True)) * np.identity(mat.shape[-1])


def stochastify_d(mat: np.ndarray):  # make dscr time stochastic
    """
    Make a matrix the generator of a discrete time Markov process.
    Scales rows to make row sums one.

    Parameters
    ----------
    mat : la.lnarray (...,n,n)
        square matrix with non-negative elements.
        **Modified** in place
    """
    mat /= mat.sum(axis=-1, keepdims=True)


stochastify = stochastify_c
# =============================================================================
# Counts & types
# =============================================================================


def _unpack_nest(nest: IntOrSeq) -> int:
    """Get one element of (nested) sequence"""
    while isinstance(nest, Sequence):
        nest = nest[0]
    return nest


def _get_size(arr: np.ndarray, kwds: dict, is_params: bool) -> int:
    """get size by pulling axes argument from keywords"""
    args = ('num_param', 'npar') if is_params else ('num_st', 'nst')
    siz = kwds.get(args[0], kwds.get(args[1], None))
    if siz is not None:
        return siz
    args = ('paxis', 'axis', -1) if is_params else ('maxes', 'axes', (-2, -1))
    axis = _unpack_nest(kwds.get(args[0], kwds.get(*args[1:])))
    return arr.shape[axis]


def _drn_mult(drn: IntOrSeq) -> int:
    """Get factor of two if drn == 0"""
    return 1 if _unpack_nest(drn) else 2


def num_param(states: Sized, *, serial: bool = False, ring: bool = False,
              uniform: bool = False, drn: IntOrSeq = 0, **kwds) -> int:
    """Number of independent rates per matrix

    Parameters
    ----------
    states : int or ndarray (n,...)
        Number of states, or array over states.
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    uniform : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `uni_ring_params_to_mat`?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    params : int
        Number of rate parameters.
    """
    if isinstance(states, np.ndarray):
        states = _get_size(states, kwds, False)
    mult = _drn_mult(drn)
    if uniform:
        return mult
    if serial:
        return mult * (states - 1)
    if ring:
        return mult * states
    return mult * states * (states - 1) // 2


def num_state(params: Sized, *, serial: bool = False, ring: bool = False,
              uniform: bool = False, drn: IntOrSeq = 0, **kwds) -> int:
    """Number of states from rate vector

    Parameters
    ----------
    params : int or ndarray (n,)
        Number of rate parameters, or vector of rates.
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: True
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    states : int
        Number of states.
    """
    if uniform:
        raise ValueError("num_states is ambiguous when uniform")
    if isinstance(params, np.ndarray):
        params = _get_size(params, kwds, True)
    params *= 2 // _drn_mult(drn)
    if serial:
        return params // 2 + 1
    if ring:
        return params // 2
    return np.rint(0.5 + np.sqrt(0.25 + params)).astype(int)


def mat_type_siz(params: Sized, states: Sized, **kwds) -> _ty.Tuple[bool, ...]:
    """Is process (uniform) ring/serial/... inferred from array shapes

    If `uniform`, we cannot distinguish `general`, `seial` and `ring` without
    looking at matrix elements.

    Parameters
    ----------
    params : int or ndarray (np,)
        Number of rate parameters, or vector of rates.
    states : int or ndarray (n,...)
        Number of states, or array over states.

    Returns
    -------
    serial : bool
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: bool
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
        Can only determine `|drn|`, not its sign.
    uniform : bool
        Is the rate vector for `*_params_to_mat` or `uni_*_params_to_mat`?
        * = general, serial or ring.
    """
    if isinstance(params, np.ndarray):
        params = _get_size(params, kwds, True)
    if isinstance(states, np.ndarray):
        states = _get_size(states, kwds, False)
    uniform = params in {1, 2}
    drn = params in {1, states, states - 1, states * (states - 1) // 2}
    ring = uniform or (params in {2 * states, states})
    serial = uniform or (params in {2 * (states - 1), states - 1})
    return serial, ring, drn, uniform


# =============================================================================
# Helpers for broadcasting drn/axis
# =============================================================================


def _posify(ndim: int, axes: AxesOrSeq) -> AxesOrSeq:
    """normalise axes"""
    if isinstance(axes, int):
        return axes % ndim
    return [_posify(ndim, axs) for axs in axes]


def _negify(ndim: int, axes: AxesOrSeq) -> AxesOrSeq:
    """normalise axes"""
    if isinstance(axes, int):
        return (axes % ndim) - ndim
    return [_negify(ndim, axs) for axs in axes]


def _sort_axes(ndim: int, fun_axes: AxesOrSeq, drn_axes: IntOrSeq,
               to_mat: bool) -> _ty.Tuple[AxesOrSeq, IntOrSeq]:
    """order axes so that fun_axes is increasing"""
    fun_axes, drn_axes = _negify(ndim, fun_axes), _negify(ndim, drn_axes)
    if isinstance(drn_axes, int):
        drn_axes = (drn_axes,) * len(fun_axes)
    faxes, daxes = np.array(fun_axes), np.array(drn_axes)
    inds = np.argsort(faxes) if to_mat else np.argsort(faxes[:, -1])
    return faxes[inds].tolist(), daxes[inds].tolist()


def bcast_axes(fun: _ty.Callable[..., ArrayType], arr: ArrayType, *args,
               drn: IntOrSeq = 0, drn_axis: IntOrSeq = 0,
               fun_axis: OrSeqOf[Axies] = -1, **kwds) -> ArrayType:
    """broadcast over axes"""
    outarr = np.asanyarray(arr)
    to_mat = kwds.get('to_mat', False)
    fun_axis, drn_axis = _sort_axes(fun_axis, drn_axis, to_mat, arr.ndim)
    fkey = 'axis' if to_mat else 'axes'
    for daxis, faxis in zip(drn_axis, fun_axis):
        kwds[fkey] = faxis
        outarr = fun(outarr, *args, drn=drn, daxis=daxis, **kwds)
    return outarr


def bcast_drns(fun: _ty.Callable[..., ArrayType], arr: ArrayType, *args,
               drn: IntOrSeq = 0, drn_axis: IntOrSeq = 0,
               fun_axis: OrSeqOf[Axies] = -1, **kwds) -> ArrayType:
    """broadcast an axis wrt drn"""
    to_mat = kwds.pop('to_mat', False)
    fkey = 'axis' if to_mat else 'axes'
    kwds[fkey] = fun_axis
    arr = np.asanyarray(arr)
    drn_axis = _posify(arr.ndim, drn_axis)
    narr = np.moveaxis(arr, drn_axis, 0)
    result = [fun(slc, *args, drn=way, **kwds) for slc, way in zip(narr, drn)]
    return np.moveaxis(np.stack(result), 0, drn_axis)


def bcast_update(updater: _ty.Callable[..., None],
                 arrays: _ty.Tuple[np.ndarray, ...],
                 drn: IntOrSeq,
                 fun_axes: _ty.Tuple[AxesOrSeq, ...],
                 drn_axes: _ty.Tuple[IntOrSeq, ...],
                 *args, **kwds):
    """Update arrays with other arrays.

    Returns
    -------
    None
        modifies `mat` in place.
    """
    num = len(arrays)
    if not isinstance(drn_axes[0], int):
        for axes in zip(*fun_axes, *drn_axes):
            bcast_update(updater, arrays, drn, axes[:num], axes[num:], *args,
                         **kwds)
    elif not isinstance(drn, int):
        narr = [np.moveaxis(arr, dax, 0) for arr, dax in zip(arrays, drn_axes)]
        for arrd in zip(*narr, drn):
            bcast_update(updater, arrd[:-1], arrd[-1], fun_axes, (0,) * num,
                         *args, **kwds)
    else:
        updater(arrays, drn, fun_axes, drn_axes, *args, **kwds)


# =============================================================================
# Parameters to matrices
# =============================================================================


def params_to_mat(params: np.ndarray, fun: IndFun, drn: IntOrSeq,
                  axis: IntOrSeq, daxis: IntOrSeq, **kwds) -> array:
    """Helper function for *_params_to_mat

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    nst : int
        Number of states.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0
    Other key word parameters for `num_state`.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.
    """
    kwds.update(drn=drn, fun_axis=axis, drn_axis=daxis, to_mat=True)
    if isinstance(axis, Sequence):
        return bcast_axes(params_to_mat, params, fun, **kwds)
    if isinstance(drn, Sequence):
        return bcast_drns(params_to_mat, params, fun, **kwds)
    params = np.asanyarray(params)
    npar = _get_size(params, kwds, True)
    nst = num_state(npar, **kwds)
    axis = _posify(params.ndim, axis)
    params = np.moveaxis(params, axis, -1)
    shape = params.shape[:-1]
    mat = zeros(shape + (nst**2,))
    mat[..., fun(nst, drn)] = params
    mat = mat.reshape(shape + (nst, nst))
    stochastify(mat)
    return np.moveaxis(mat, (-2, -1), (axis, axis+1))


def uni_to_any(params: np.ndarray, nst: int, axis: IntOrSeq, **kwds
               ) -> np.ndarray:
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    params : ndarray (1,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        If `drn == 0`, you must provide 2 parameters, one for each direction.
        See docs for `*_inds` for details.
    nst : int
        Number of states.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    Other key word parameters for `num_param`.

    Returns
    -------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags above.
        See docs for `*_inds` for details.
    """
    if isinstance(axis, Sequence):
        kwds.update(fun_axis=axis, to_mat=True)
        return bcast_axes(uni_to_any, params, nst, **kwds)
    params = np.asanyarray(params)
    axis = _posify(params.ndim, axis)
    params = np.moveaxis(params, axis, -1)
    # if drn == 0, then params.shape[axis] == 2, so each needs expanding by
    # half of the real num_param
    kwds.update(drn=1, uniform=False)
    npar = num_param(nst, **kwds)
    full = np.broadcast_to(params[..., None], params.shape + (npar,))
    shape = full.shape[:-2] + (-1,)
    return np.moveaxis(full.reshape(shape), -1, axis)


# =============================================================================
# Matrices to parameters
# =============================================================================


def _out_axis(ndim: int, axes: Axes) -> int:
    """Which matrix axis to use for parameters"""
    return min(_posify(ndim, axes))


def mat_to_params(mat: ArrayType, fun: IndFun, drn: IntOrSeq, axes: Axes,
                  daxis: IntOrSeq, **kwds) -> ArrayType:
    """Helper function for *_mat_to_params

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int]
        Axes to treat as (from, to) axes.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0
    """
    kwds.update(drn=drn, fun_axis=axes, drn_axis=daxis)
    if isinstance(axes[0], Sequence):
        return bcast_axes(mat_to_params, mat, fun, **kwds)
    if isinstance(drn, Sequence):
        return bcast_drns(mat_to_params, mat, fun, **kwds)
    mat = np.asanyarray(mat)
    oaxis = _out_axis(mat.ndim, axes)
    nst = mat.shape[axes[0]]
    mat = np.moveaxis(mat, axes, [-2, -1])
    mat = mat.reshape(mat.shape[:-2] + (-1,))
    return np.moveaxis(mat[..., fun(nst, drn)], -1, oaxis)


def to_uni(params: ArrayType, drn: IntOrSeq, grad: bool, axes: AxesOrSeq,
           **kwds) -> ArrayType:
    """Helper for uni_*_mat_to_params

    Parameters
    ----------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or half of <-
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of values in each direction, else, return mean.
    axes : Tuple[int, int] or None
        Original axes to treat as (from, to) axes.
    """
    if isinstance(axes[0], Sequence):
        kwds.update(drn=drn, fun_axis=axes, grad=grad)
        return bcast_axes(to_uni, params, **kwds)
    # Ensure the same oaxis here as in mat_to_params
    oaxis = _out_axis(params.ndim + 1, axes)
    npar = params.shape[oaxis] / _drn_mult(drn)
    new_shape = params.shape[:oaxis] + (-1, npar) + params.shape[oaxis+1:]
    params = params.reshape(new_shape).sum(axis=oaxis+1)
    if not grad:
        params /= npar
    return params


# =============================================================================
# Type hints
# =============================================================================
ArrayType = _ty.TypeVar('ArrayType', bound=np.ndarray)
Sized = _ty.Union[int, np.ndarray]
Axes = _ty.Tuple[int, int]
IndFun = _ty.Callable[[int, int], np.ndarray]
Axies = _ty.Union[int, Axes]
AxType = _ty.TypeVar('AxType', int, Axes, Axies)
OrSeqOf = _ty.Union[AxType, _ty.Sequence[AxType]]
IntOrSeq = OrSeqOf[int]
AxesOrSeq = OrSeqOf[Axes]
