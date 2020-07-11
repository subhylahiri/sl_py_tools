# -*- coding: utf-8 -*-
"""Utilities for module markov_param
"""
from __future__ import annotations
import typing as _ty

import numpy as np
# from numpy import ndarray as array
# from numpy import zeros

from numpy_linalg import lnarray as array
from numpy_linalg import zeros

from .markov import stochastify_c

# =============================================================================
# Counts & types
# =============================================================================


def num_param(states: Sized, *, serial: bool = False, ring: bool = False,
              uniform: bool = False, drn: int = 0) -> int:
    """Number of independent rates

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
        states = states.shape[0]
    mult = 2 - bool(drn)  # double if drn == 0
    if uniform:
        return mult
    if serial:
        return mult * (states - 1)
    if ring:
        return mult * states
    return mult * states * (states - 1) // 2


# =============================================================================
# Helper for broadcasting
# =============================================================================


def _posify(ndim: int, axes: OrSeqOf[Axies], default: Axies, zax: bool = False
            ) -> OrSeqOf[Axies]:
    """normalise axes"""
    if isinstance(default, bool):
        default = -1 if default else (-2, -1)
    if isinstance(axes, int, type(None)):
        return default if axes is None else axes % ndim
    if isinstance(default, int):
        return [_posify(ndim, axs, default, False) for axs in axes]
    if zax:
        return [_posify(ndim, axs, dfl) for axs, dfl in zip(axes, default)]
    return [_posify(ndim, axs, default, True) for axs in axes]


def _sort_axes(fun_axes: OrSeqOf[Axies], drn_axes: OrSeqOf[int], to_mat: bool,
               ndim: int) -> (OrSeqOf[Axies], OrSeqOf[int]):
    """order axes so that fun_axes is decreasing"""
    drn_axes = _posify(ndim, drn_axes, 0)
    fun_axes = _posify(ndim, fun_axes, to_mat)
    faxes, daxes = np.array(fun_axes), np.array(drn_axes)
    inds = np.argsort(-faxes) if to_mat else np.argsort(-faxes[:, -1])
    return faxes[inds].tolist(), daxes[inds].tolist()


def bcast_drns(fun: _ty.Callable, arr: ArrayType,
               drns: OrSeqOf[int], drn_axis: OrSeqOf[int],
               fun_axis: OrSeqOf[Axies], *args, **kwds) -> ArrayType:
    """broadcast an axis wrt drn"""
    if not isinstance(drn_axis, (int, type(None))):
        narr = arr
        to_mat = kwds.get('to_mat', False)
        fun_axis, drn_axis = _sort_axes(fun_axis, drn_axis, to_mat, arr.ndim)
        for daxis, faxis in zip(drn_axis, fun_axis):
            narr = bcast_drns(fun, narr, drns, daxis, faxis, *args, **kwds)
        return narr
    to_mat = kwds.pop('to_mat', False)
    if to_mat:
        kwds['axis'] = fun_axis
    else:
        kwds['axes'] = fun_axis
    if isinstance(drns, int):
        return fun(arr, *args, drn=drns, **kwds)
    drn_axis = 0 if drn_axis is None else drn_axis % arr.ndim
    narg = np.moveaxis(arr, drn_axis, 0)
    result = [fun(slc, *args, drn=drn, **kwds) for slc, drn in zip(narg, drns)]
    return np.moveaxis(np.stack(result), 0, drn_axis)


# =============================================================================
# Parameters to matrices
# =============================================================================


def params_to_mat(fun: IndFun, params: np.ndarray, nst: int, drn: int,
                  axis: int = -1) -> array:
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

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    axis = params.ndim - 1 if axis is None else axis % params.ndim
    params = np.moveaxis(params, axis, -1)
    shape = params.shape[:-1]
    mat = zeros(shape + (nst**2,))
    mat[..., fun(nst, drn)] = params
    mat = mat.reshape(shape + (nst, nst))
    stochastify_c(mat)
    return np.moveaxis(mat, (-2, -1), (axis, axis+1))


def uni_to_any(params: np.ndarray, nst: int, axis: int = -1, **kwds
               ) -> np.ndarray:
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    params : ndarray (1,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    nst : int
        Number of states.
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags above.
        See docs for `*_inds` for details.
    """
    params = np.asanyarray(params)
    axis = params.ndim - 1 if axis is None else axis % params.ndim
    params = np.moveaxis(params, axis, -1)
    kwds.update({'drn': 1, 'uniform': False})
    npr = num_param(nst, **kwds)
    full = np.broadcast_to(params[..., None], params.shape + (npr,))
    shape = params.shape[:-1] + (-1,)
    return np.moveaxis(full.reshape(shape), -1, axis)


# =============================================================================
# Matrices to parameters
# =============================================================================


def mat_to_params(fun: IndFun, mat: ArrayType, drn: int, axes: Axes
                  ) -> ArrayType:
    """Helper function for *_mat_to_params

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes.
    """
    axes = (-2, -1) if axes is None else axes
    oaxis = min(axes) % mat.ndim
    nst = mat.shape[axes[0]]
    param = np.moveaxis(mat, axes, [-2, -1])
    param = param.reshape(param.shape[:-2] + (-1,))
    return np.moveaxis(param[..., fun(nst, drn)], -1, oaxis)


def to_uni(params: ArrayType, drn: int, grad: bool, axes: Axes) -> ArrayType:
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
        If True, return sum of transitions in each direction.
        If False, return the mean.
    axes : Tuple[int, int] or None
        Original axes to treat as (from, to) axes.
    """
    axis = params.ndim - 1 if axes is None else min(axes) % (params.ndim + 1)
    npar = params.shape[axis] / (1 + (drn == 0))
    new_shape = params.shape[:axis] + (-1, npar) + params.shape[axis+1:]
    params = params.reshape(new_shape).sum(axis=axis+1)
    if not grad:
        params /= npar
    return params


def bcast_update(updater: _ty.Callable[..., None],
                 arrays: _ty.Tuple[ArrayType],
                 drn: OrSeqOf[int],
                 fun_axes: _ty.Tuple[OrSeqOf[Axies]],
                 drn_axes: _ty.Tuple[OrSeqOf[int]],
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
            bcast_update(updater, arrays, drn, axes[:num],
                         axes[num:], *args, **kwds)
    elif not isinstance(drn, int):
        narr = [np.moveaxis(arr, dax, 0) for arr, dax in zip(arrays, drn_axes)]
        for arrd in zip(*narr, drn):
            bcast_update(arrd[:-1], arrd[-1], fun_axes, [0]*num, *args, **kwds)
    else:
        kwds.update(zip(kwds.pop('keys'), tuple(fun_axes) + tuple(drn_axes)))
        updater(*arrays, *args, drn=drn, **kwds)


# =============================================================================
# Type hints
# =============================================================================
ArrayType = _ty.TypeVar('ArrayType', bound=np.ndarray)
Sized = _ty.Union[int, np.ndarray]
Axes = _ty.Optional[_ty.Tuple[int, int]]
IndFun = _ty.Callable[[int, int], np.ndarray]
Axies = _ty.Union[int, Axes]
AxType = _ty.TypeVar('AxType', int, Axes)
OrSeqOf = _ty.Union[AxType, _ty.Sequence[AxType]]
