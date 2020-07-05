# -*- coding: utf-8 -*-
"""Utilities for parameterising Markov processes

For general topology, the parameters are:
    mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
For a serial topology:
    mat_01, mat_12, ..., mat_n-2,n-1,
    mat_10, mat_21, ..., mat_n-1,n-2.
For a ring topology:
    mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
    mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
If `drn` is positive/negative, we only consider the upper/lower triangle.
If `drn == 0`, we consider both.

When `uniform`, we assume that all non-zero elements in the upper/lower
triangle are equal. When extracting `uniform` parameters we average them,
unless `grad` is `True` when we take the sum.
"""
import typing as _ty

import numpy as np
# from numpy import ndarray as array
# from numpy import zeros

from numpy_linalg import lnarray as array
from numpy_linalg import zeros

from .markov import stochastify_c

ArrayType = _ty.TypeVar('ArrayType', bound=np.ndarray)
Sized = _ty.Union[int, np.ndarray]
Axes = _ty.Optional[_ty.Tuple[int, int]]
IndFun = _ty.Callable[[int, int], np.ndarray]

# =============================================================================
# Indices of parameters
# =============================================================================


def offdiag_inds(nst: int, drn: int = 0) -> np.ndarray:
    """Indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : ndarray (n(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
    """
    if drn > 0:
        return np.ravel_multi_index(np.triu_indices(nst, 1), (nst, nst))
    if drn < 0:
        return np.ravel_multi_index(np.tril_indices(nst, -1), (nst, nst))
    # when put into groups of size nst+1:
    # M[0,0], M[0,1], M[0,2], ..., M[0,n-1], M[1,0],
    # M[1,1], M[1,2], ..., M[1,n-1], M[2,0], M[2,1],
    # ...
    # M[n-2,n-2], M[n-2,n-1], M[n-1,0], ..., M[n-1,n-2],
    # unwanted elements are 1st in each group
    k_1st = np.arange(0, nst**2, nst+1)  # ravel ind of 1st element in group
    k = np.arange(nst**2)
    return np.delete(k, k_1st)


def serial_inds(nst: int, drn: int = 0) -> np.ndarray:
    """Ravel indices of independent parameters of serial transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : ndarray (2(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-1,n-2.
    """
    pos = np.arange(1, nst**2, nst+1)
    if drn > 0:
        return pos
    neg = np.arange(nst, nst**2, nst+1)
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def ring_inds(nst: int, drn: int = 0) -> np.ndarray:
    """Ravel indices of independent parameters of ring transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : ndarray (2n,)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    """
    pos = np.hstack((np.arange(1, nst**2, nst+1), [nst*(nst-1)]))
    if drn > 0:
        return pos
    neg = np.hstack(([nst-1], np.arange(nst, nst**2, nst+1)))
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def _ind_fun(serial: bool, ring: bool) -> IndFun:
    """which index function to use"""
    if serial:
        return serial_inds
    if ring:
        return ring_inds
    return offdiag_inds


def param_inds(nst: int, serial: bool = False, ring: bool = False,
               drn: int = 0) -> np.ndarray:
    """Ravel indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
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
    mat : ndarray (k,k), k in (n(n-1), 2(n-1), 2n, 2)
        Indices of independent elements. For the order, see docs for `*_inds`.
    """
    return _ind_fun(serial, ring)(nst, drn)


# =============================================================================
# Counts & types
# =============================================================================


def num_param(states: Sized, serial: bool = False, ring: bool = False,
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


def num_state(params: Sized, serial: bool = False, ring: bool = False,
              drn: int = 0) -> int:
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
    if isinstance(params, np.ndarray):
        params = params.shape[-1]
    if drn:
        params *= 2
    if serial:
        return params // 2 + 1
    if ring:
        return params // 2
    return (0.5 + np.sqrt(0.25 + params)).astype(int)


def mat_type(params: Sized, states: Sized) -> _ty.Tuple[bool, ...]:
    """Is it a (uniform) ring

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
        params = params.shape[-1]
    if isinstance(states, np.ndarray):
        states = states.shape[-1]
    uniform = params in {1, 2}
    drn = params in {1, states, states - 1, states * (states - 1) // 2}
    ring = uniform or (params in {2 * states, states})
    serial = uniform or (params in {2 * (states - 1), states - 1})
    return serial, ring, drn, uniform


def mat_type_dict(params: Sized, states: Sized) -> _ty.Tuple[bool, ...]:
    """Is it a (uniform) ring

    Parameters
    ----------
    params : int or ndarray (np,)
        Number of rate parameters, or vector of rates.
    states : int or ndarray (n,...)
        Number of states, or array over states.

    Returns
    -------
    dict containing:
        serial : bool
            Is the rate vector meant for `serial_params_to_mat` or
            `gen_params_to_mat`?
        ring : bool
            Is the rate vector meant for `ring_params_to_mat` or
            `gen_params_to_mat`?
        drn: bool
            If nonzero, only include transitions in direction `i->i+sgn(drn)`.
            Can only determine `|drn|`, not its sign.
        uniform : bool
            Is the rate vector for `*_params_to_mat` or `uni_*_params_to_mat`?
            * = general, serial or ring.
    """
    serial, ring, drn, uniform = mat_type(params, states)
    return {'serial': serial, 'ring': ring, 'drn': drn, 'uniform': uniform}


# =============================================================================
# Parameters to matrices
# =============================================================================


def _params_to_mat(fun: IndFun, params: np.ndarray, nst: int, drn: int,
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


def _uni_to_any(params: np.ndarray, nst: int, axis: int = -1,
                **kwds) -> np.ndarray:
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
    npr = num_param(nst, drn=1, uniform=False, **kwds)
    full = np.broadcast_to(params[..., None], params.shape + (npr,))
    shape = params.shape[:-1] + (-1,)
    return np.moveaxis(full.reshape(shape), -1, axis)


def gen_params_to_mat(params: np.ndarray, drn: int = 0,
                      axis: int = -1) -> array:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (n(n-1),)
        Vector of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    nst = num_state(params, drn=drn)
    return _params_to_mat(offdiag_inds, params, nst, drn, axis)


def uni_gen_params_to_mat(params: np.ndarray, num_st: int, drn: int = 0,
                          axis: int = -1) -> array:
    """Uniform transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2,) or (1,)
        Vector of independent elements, in order:
        mat_01 = ... = mat_0n-1 = mat_12 = ... mat_1n-1 = ... = mat_n-2,n-1,
        mat_10 = mat_20 = mat_21 = mat_30 = ... = mat_n-10 = ... = mat_n-1,n-2.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    gen_params = _uni_to_any(params, num_st, axis)
    if drn:
        return gen_params_to_mat(gen_params, drn, axis)
    # pylint: disable=unbalanced-tuple-unpacking
    pos, neg = np.split(gen_params, 2)
    return gen_params_to_mat(pos, 1, axis) + gen_params_to_mat(neg, -1, axis)


def serial_params_to_mat(params: np.ndarray, drn: int = 0,
                         axis: int = -1) -> array:
    """Serial transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2(n-1),)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-1,n-2.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    nst = num_state(params, serial=True, drn=drn)
    return _params_to_mat(serial_inds, params, nst, drn, axis)


def uni_serial_params_to_mat(params: np.ndarray, num_st: int, drn: int = 0,
                             axis: int = -1) -> array:
    """Uniform serial transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2,)
        Vector of independent elements, in order:
        mat_01 = mat_12 = ... = mat_n-2,n-1,
        mat_10 = mat_21 = ... = mat_n-1,n-2.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    return serial_params_to_mat(_uni_to_any(params, num_st, axis, serial=True),
                                drn, axis)


def ring_params_to_mat(params: np.ndarray, drn: int = 0,
                       axis: int = -1) -> array:
    """Ring transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2n,)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    nst = num_state(params.size, ring=True, drn=drn)
    return _params_to_mat(ring_inds, params, nst, drn, axis)


def uni_ring_params_to_mat(params: np.ndarray, num_st: int, drn: int = 0,
                           axis: int = -1) -> array:
    """Ring transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2,)
        Vector of independent elements, in order:
        mat_01 = mat_12 = ... = mat_n-2,n-1 = mat_n-1,0,
        mat_0,n-1 = mat_10 = mat_21 = ... = mat_n-1,n-2.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    return ring_params_to_mat(_uni_to_any(params, num_st, axis, ring=True),
                              drn, axis)


def params_to_mat(params: np.ndarray, *, serial: bool = False,
                  ring: bool = False, uniform: bool = False,
                  drn: int = 0, nst: int = 2, axis: int = -1) -> array:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    uniform : bool, optional, default: False
        Is the rate vector for `*_params_to_mat` or `uni_*_params_to_mat`?
        * = general, serial or ring.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    nst : int, optional, default: 2
        Number of states. Only needed when `uniform` is True
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The (from,to) axes are inserted in position `axis`, if given.
    """
    if uniform:
        params = _uni_to_any(params, nst, axis, serial=serial, ring=ring)
    else:
        nst = num_state(params, serial=serial, ring=ring, drn=drn)
    return _params_to_mat(_ind_fun(serial, ring), params, nst, drn, axis)


def matify(params_or_mat: np.ndarray, *args, **kwds) -> array:
    """Transition matrix from independent parameters, if not already so.

    Parameters
    ----------
    params_or_mat : ndarray (np,) or (n,n)
        Either vector of independent elements (in order that depends on flags,
        see docs for `params_to_mat`) or continuous time stochastic matrix.
    other arguments passed to `params_to_mat`

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
    """
    if params_or_mat.ndim >= 2:
        return params_or_mat
    return params_to_mat(params_or_mat, *args, **kwds)


# =============================================================================
# Matrices to parameters
# =============================================================================


def _mat_to_params(fun: IndFun, mat: ArrayType, drn: int,
                   axes: Axes) -> ArrayType:
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


def _to_uni(params: ArrayType, drn: int, grad: bool,
            axes: Axes) -> ArrayType:
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


def gen_mat_to_params(mat: ArrayType, drn: int = 0,
                      axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (...,n(n-1),)
        Vector of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    return _mat_to_params(offdiag_inds, mat, drn, axes)


def uni_gen_mat_to_params(mat: ArrayType, grad: bool = True,
                          drn: int = 0, axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of uniform transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of left/right transitions.
        If False, return the mean.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (2,)
        Vector of independent elements, in order (grad=False):
        mat_01 = ... = mat_0n-1 = mat_12 = ... mat_1n-1 = ... = mat_n-2,n-1,
        mat_10 = mat_20 = mat_21 = mat_30 = ... = mat_n-10 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
        mat_01 + ... + mat_0n-1 + mat_12 + ... mat_1n-1 + ... + mat_n-2,n-1,
        mat_10 + mat_20 + mat_21 + mat_30 + ... + mat_n-10 + ... + mat_n-1,n-2.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    if drn:
        return _to_uni(gen_mat_to_params(mat, drn, axes), drn, grad, axes)
    # need to separate pos, neg
    params_pos = gen_mat_to_params(mat, 1, axes)
    params_neg = gen_mat_to_params(mat, -1, axes)
    return _to_uni(np.hstack((params_pos, params_neg)), drn, grad, axes)


def serial_mat_to_params(mat: ArrayType, drn: int = 0,
                         axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of serial transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (2(n-1),)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-2,n-1.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    return _mat_to_params(serial_inds, mat, drn, axes)


def uni_serial_mat_to_params(mat: ArrayType, grad: bool = True, drn: int = 0,
                             axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of uniform serial transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of left/right transitions.
        If False, return the mean.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (2,)
        Vector of independent elements, in order (grad=False):
            mat_01 = mat_12 = ... = mat_n-2,n-1,
            mat_10 = mat_21 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
            mat_01 + mat_12 + ... + mat_n-2,n-1,
            mat_10 + mat_21 + ... + mat_n-1,n-2.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    return _to_uni(serial_mat_to_params(mat, drn, axes), drn, grad, axes)


def ring_mat_to_params(mat: ArrayType, drn: int = 0,
                       axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of ring transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (2n,)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    return _mat_to_params(ring_inds, mat, drn, axes)


def uni_ring_mat_to_params(mat: ArrayType, grad: bool = True, drn: int = 0,
                           axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of ring transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of (anti)clockwise transitions.
        If False, return the mean.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (2,)
        Vector of independent elements, in order (grad=False):
            mat_01 = mat_12 = ... = mat_n-2,n-1 = mat_n-10,
            mat_0n-1 = mat10 = mat_21 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
            mat_01 + mat_12 + ... + mat_n-2,n-1 + mat_n-10,
            mat_0n-1 + mat10 + mat_21 + ... + mat_n-1,n-2.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    return _to_uni(ring_mat_to_params(mat, drn, axes), drn, grad, axes)


def mat_to_params(mat: ArrayType, *,
                  serial: bool = False, ring: bool = False, drn: int = 0,
                  uniform: bool = False, grad: bool = True,
                  axes: Axes = (-2, -1)) -> ArrayType:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    uniform : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `uni_ring_params_to_mat`?
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of each group of equal transitions.
        If False, return the mean.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of them
        Vector of independent elements. For the order, see docs for `*_inds`.
        Elements lie across the axis given by `min(axes)`, if given.
    """
    params = _mat_to_params(_ind_fun(serial, ring), mat, drn, axes)
    if uniform:
        return _to_uni(params, drn, grad, axes)
    return params


def paramify(params_or_mat: ArrayType, *args, **kwds) -> ArrayType:
    """Independent parameters of transition matrix, if not already so.

    Parameters
    ----------
    params_or_mat : ndarray (np,) or (n,n)
        Either vector of independent elements (in order that depends on flags,
        see docs for `params_to_mat`) or continuous time stochastic matrix.
    other arguments passed to `mat_to_params`

    Returns
    -------
    params : ndarray (np,)
        Vector of independent elements (in order that depends on flags,
        see docs for `*_inds` for details).
    """
    if params_or_mat.ndim >= 2:
        return params_or_mat
    return mat_to_params(params_or_mat, *args, **kwds)


def mat_update_params(mat: ArrayType, params: np.ndarray,
                      **kwds) -> ArrayType:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of them
        Vector of independent elements. For the order, see docs for `*_inds`.

    Keyword only
    ------------
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    uniform : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `uni_ring_params_to_mat`?
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of each group of equal transitions.
        If False, return the mean.

    Returns
    -------
    None
        modifies `mat` in place.
    """
    nst = mat.shape[-1]
    uniform = kwds.pop('uniform', False)
    if uniform:
        params = _uni_to_any(params, nst)
    inds = param_inds(nst, **kwds)
    mat.ravel()[inds] = params
    stochastify_c(mat)


def tens_to_mat(tens: ArrayType, *,
                serial: bool = False, ring: bool = False,
                drn: _ty.Tuple[int, int] = (0, 0),
                uniform: bool = False, grad: bool = True) -> ArrayType:
    """Independent parameters of 4th rank tensor.

    Parameters
    ----------
    tens : ndarray (n,n,n,n)
        Continuous time stochastic matrix.
    serial : bool, optional, default: False
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: Tuple[int], optional, default: (0,0)
        Directions for first/last pair of axes.
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    uniform : bool, optional, default: False
        Is the rate vector meant for `ring_params_to_mat` or
        `uni_ring_params_to_mat`?
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of each group of equal transitions.
        If False, return the mean.

    Returns
    -------
    mat : ndarray (n**2,n**2)
        Matrix of independent elements, each axis in order `*_mat_to_params`.
    """
    nst = tens.shape[0]
    mat = tens.reshape((nst**2, nst**2))
    inds = [param_inds(nst, serial=serial, ring=ring, drn=d) for d in drn]
    mat = mat[np.ix_(inds[0], inds[1])]
    if uniform:
        nind = len(inds[0]) // 2
        mat = np.block([[mat[:nind, :nind].sum(), mat[:nind, nind:].sum()],
                        [mat[nind:, :nind].sum(), mat[nind:, nind:].sum()]])
        if drn[0]:
            mat = mat.sum()
            nind = len(inds[0])
        if not grad:
            mat /= nind**2
    return mat
