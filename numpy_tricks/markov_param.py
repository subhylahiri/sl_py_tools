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
from __future__ import annotations

import typing as _ty

import numpy as np

from numpy_linalg import flattish, foldaxis

from . import _markov_helper as _mh

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
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-1,n-2.
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


def offdiag_split_inds(nst: int, drn: int = 0) -> np.ndarray:
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
        First, te upper/right triangle:
        mat_01, ..., mat_0n-1, mat_12, ..., mat_n-3n-2, mat_n-3n-1, mat_n-2n-1,
        followed by the lower/left triangle:
        mat_10, mat_20, mat_21, ..., mat_n-2n-3, mat_n-10, ... mat_n-1n-2.
    """
    if drn:
        return offdiag_inds(nst, drn)
    return np.hstack((offdiag_inds(nst, 1), offdiag_inds(nst, -1)))


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
    pos = np.r_[1:nst**2:nst+1, nst*(nst-1)]
    if drn > 0:
        return pos
    neg = np.r_[nst-1, 1:nst**2:nst+1]
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def _ind_fun(serial: bool, ring: bool, uniform: bool = False) -> IndFun:
    """which index function to use"""
    if serial:
        return serial_inds
    if ring:
        return ring_inds
    if uniform:
        return offdiag_split_inds
    return offdiag_inds


def param_inds(nst: int, serial: bool = False, ring: bool = False,
               uniform=False, drn: int = 0) -> np.ndarray:
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
    return _ind_fun(serial, ring, uniform)(nst, drn)


# =============================================================================
# Counts & types
# =============================================================================


def mat_type_val(mat: np.ndarray, axes: Axes = (-2, -1)
                 ) -> _ty.Tuple[bool, bool, int, bool]:
    """Is process (uniform) ring/serial/... from elements

    Parameters
    ----------
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)

    Returns
    -------
    serial : bool
        Is the rate vector meant for `serial_params_to_mat` or
        `gen_params_to_mat`?
    ring : bool
        Is the rate vector meant for `ring_params_to_mat` or
        `gen_params_to_mat`?
    drn: int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    uniform : bool
        Is the matrix from `*_params_to_mat` or `uni_*_params_to_mat`?
        * = gen, serial or ring.
    """
    nst = mat.shape[axes[-1]]
    mat = np.moveaxis(mat, axes, (-2, -1))
    mat = mat[(0,) * (mat.ndim - 2)].ravel()
    rng_inds = ring_inds(nst, 0)
    gen_inds = np.setdiff1d(offdiag_inds(nst, 0), rng_inds, True)
    rng_inds = np.setdiff1d(rng_inds, serial_inds(nst, 0), True)
    ring = np.allclose(mat[gen_inds], 0)
    serial = ring and np.allclose(mat[rng_inds], 0)
    ring &= not serial
    if np.allclose(mat[offdiag_inds(nst, -1)], 0):
        drn = 1
    elif np.allclose(mat[offdiag_inds(nst, 1)], 0):
        drn = -1
    else:
        drn = 0
    ifun = _ind_fun(serial, ring)
    uniform = np.allclose([np.std(mat[ifun(nst, 1)]),
                           np.std(mat[ifun(nst, -1)])], 0)
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
    serial, ring, drn, uniform = mat_type_siz(params, states)
    return {'serial': serial, 'ring': ring, 'drn': drn, 'uniform': uniform}


# =============================================================================
# Parameters to matrices
# =============================================================================


def gen_params_to_mat(params: np.ndarray, drn: IntOrSeq = 0,
                      axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    offdiag_inds, gen_mat_to_params
    """
    return _mh.params_to_mat(params, offdiag_inds, drn, axis, daxis)


def uni_gen_params_to_mat(params: np.ndarray, num_st: int, drn: IntOrSeq = 0,
                          axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
    """Uniform transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2,) or (1,)
        Vector of independent elements, in order:
        mat_01 = ... = mat_0n-1 = mat_12 = ... mat_1n-1 = ... = mat_n-2,n-1,
        mat_10 = mat_20 = mat_21 = mat_30 = ... = mat_n-10 = ... = mat_n-1,n-2.
        If `drn == 0`, you must provide 2 parameters, one for each direction.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    offdiag_split_inds, uni_gen_mat_to_params
    """
    gen_params = _mh.uni_to_any(params, num_st, axis=axis)
    if drn:
        return gen_params_to_mat(gen_params, drn, axis, daxis)
    return _mh.params_to_mat(params, offdiag_split_inds, drn, axis, daxis)


def serial_params_to_mat(params: np.ndarray, drn: IntOrSeq = 0,
                         axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    serial_inds, serial_mat_to_params
    """
    return _mh.params_to_mat(params, serial_inds, drn, axis, daxis)


def uni_serial_params_to_mat(
        params: np.ndarray, num_st: int, drn: IntOrSeq = 0,
        axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
    """Uniform serial transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2,) or (1,)
        Vector of independent elements, in order:
        mat_01 = mat_12 = ... = mat_n-2,n-1,
        mat_10 = mat_21 = ... = mat_n-1,n-2.
        If `drn == 0`, you must provide 2 parameters, one for each direction.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    serial_inds, uni_serial_mat_to_params
    """
    ser_params = _mh.uni_to_any(params, num_st, axis=axis, serial=True)
    return serial_params_to_mat(ser_params, drn, axis, daxis)


def ring_params_to_mat(params: np.ndarray, drn: IntOrSeq = 0,
                       axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    ring_inds, ring_mat_to_params
    """
    return _mh.params_to_mat(params, ring_inds, drn, axis, daxis)


def uni_ring_params_to_mat(params: np.ndarray, num_st: int, drn: IntOrSeq = 0,
                           axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
    """Ring transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (2,) or (1,)
        Vector of independent elements, in order:
        mat_01 = mat_12 = ... = mat_n-2,n-1 = mat_n-1,0,
        mat_0,n-1 = mat_10 = mat_21 = ... = mat_n-1,n-2.
        If `drn == 0`, you must provide 2 parameters, one for each direction.
    num_st : int
        Number of states.
    drn: int or Sequence[int], optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    ring_inds, ring_mat_to_params
    """
    ring_params = _mh.uni_to_any(params, num_st, axis=axis, ring=True)
    return ring_params_to_mat(ring_params, drn, axis, daxis)


def params_to_mat(params: np.ndarray, *, serial: bool = False,
                  ring: bool = False, uniform: bool = False, nst: int = 2,
                  drn: IntOrSeq = 0, axis: IntOrSeq = -1, daxis: IntOrSeq = 0
                  ) -> Array:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
        If `uniform and drn == 0`, we need 2 parameters, one for each direction
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
        Number of states. Only needed when `uniform` is `True`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.

    See Also
    --------
    param_inds, mat_to_params
    """
    opts = {'serial': serial, 'ring': ring}
    if uniform:
        params = _mh.uni_to_any(params, nst, axis=axis, **opts)
    return _mh.params_to_mat(params, _ind_fun(serial, ring, uniform),
                             drn, axis, daxis)


def matify(params_or_mat: np.ndarray, *args, **kwds) -> Array:
    """Transition matrix from independent parameters, if not already so.

    Parameters
    ----------
    params_or_mat : ndarray (np,) or (n,n)
        Either vector of independent elements (in order that depends on flags,
        see docs for `params_to_mat`) or continuous time stochastic matrix.
    Other arguments passed to `params_to_mat`

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.

    See Also
    --------
    params_to_mat, paramify
    """
    if params_or_mat.ndim >= 2:
        return params_or_mat
    return params_to_mat(params_or_mat, *args, **kwds)


# =============================================================================
# Matrices to parameters
# =============================================================================


def gen_mat_to_params(mat: ArrayType, drn: IntOrSeq = 0,
                      axes: AxesOrSeq = (-2, -1), daxis: IntOrSeq = 0
                      ) -> ArrayType:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (...,n(n-1),)
        Vector of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    offdiag_inds, gen_params_to_mat
    """
    return _mh.mat_to_params(mat, offdiag_inds, drn, axes, daxis)


def uni_gen_mat_to_params(mat: ArrayType, grad: bool = True, drn: IntOrSeq = 0,
                          axes: AxesOrSeq = (-2, -1), daxis: IntOrSeq = 0
                          ) -> ArrayType:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (2,)
        Vector of independent elements, in order (grad=False):
        mat_01 = ... = mat_0n-1 = mat_12 = ... mat_1n-1 = ... = mat_n-2,n-1,
        mat_10 = mat_20 = mat_21 = mat_30 = ... = mat_n-10 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
        mat_01 + ... + mat_0n-1 + mat_12 + ... mat_1n-1 + ... + mat_n-2,n-1,
        mat_10 + mat_20 + mat_21 + mat_30 + ... + mat_n-10 + ... + mat_n-1,n-2.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    offdiag_split_inds, uni_gen_params_to_mat
    """
    if drn:
        return _mh.to_uni(gen_mat_to_params(mat, drn, axes, daxis),
                          drn, grad, axes)
    # need to separate pos, neg
    return _mh.to_uni(_mh.mat_to_params(mat, offdiag_split_inds, drn, axes),
                      drn, grad, axes)


def serial_mat_to_params(mat: ArrayType, drn: IntOrSeq = 0,
                         axes: AxesOrSeq = (-2, -1), daxis: IntOrSeq = 0
                         ) -> ArrayType:
    """Independent parameters of serial transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (2(n-1),)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-2,n-1.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    serial_inds, serial_params_to_mat
    """
    return _mh.mat_to_params(mat, serial_inds, drn, axes, daxis)


def uni_serial_mat_to_params(mat: ArrayType, grad: bool = True,
                             drn: IntOrSeq = 0, axes: AxesOrSeq = (-2, -1),
                             daxis: IntOrSeq = 0) -> ArrayType:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (2,)
        Vector of independent elements, in order (grad=False):
            mat_01 = mat_12 = ... = mat_n-2,n-1,
            mat_10 = mat_21 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
            mat_01 + mat_12 + ... + mat_n-2,n-1,
            mat_10 + mat_21 + ... + mat_n-1,n-2.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    serial_inds, uni_serial_params_to_mat
    """
    return _mh.to_uni(serial_mat_to_params(mat, drn, axes, daxis),
                      drn, grad, axes)


def ring_mat_to_params(mat: ArrayType, drn: IntOrSeq = 0,
                       axes: AxesOrSeq = (-2, -1), daxis: IntOrSeq = 0
                       ) -> ArrayType:
    """Independent parameters of ring transition matrix.

    Parameters
    ----------
    mat : ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int] or None
        Axes to treat as (from, to) axes, by default: (-2, -1)
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (2n,)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    ring_inds, ring_params_to_mat
    """
    return _mh.mat_to_params(mat, ring_inds, drn, axes, daxis)


def uni_ring_mat_to_params(mat: ArrayType, grad: bool = True,
                           drn: IntOrSeq = 0, axes: AxesOrSeq = (-2, -1),
                           daxis: IntOrSeq = 0) -> ArrayType:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (2,)
        Vector of independent elements, in order (grad=False):
            mat_01 = mat_12 = ... = mat_n-2,n-1 = mat_n-10,
            mat_0n-1 = mat10 = mat_21 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
            mat_01 + mat_12 + ... + mat_n-2,n-1 + mat_n-10,
            mat_0n-1 + mat10 + mat_21 + ... + mat_n-1,n-2.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    ring_inds, uni_ring_params_to_mat
    """
    return _mh.to_uni(ring_mat_to_params(mat, drn, axes, daxis),
                      drn, grad, axes)


def mat_to_params(mat: ArrayType, *,
                  serial: bool = False, ring: bool = False,
                  uniform: bool = False, grad: bool = True, drn: IntOrSeq = 0,
                  axes: AxesOrSeq = (-2, -1), daxis: IntOrSeq = 0
                  ) -> ArrayType:
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
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of them
        Vector of independent elements. For the order, see docs for `*_inds`.
        Elements lie across the earlier axis of `axes`.

    See Also
    --------
    param_inds, params_to_mat
    """
    params = _mh.mat_to_params(mat, _ind_fun(serial, ring, uniform),
                               drn, axes, daxis)
    if uniform:
        return _mh.to_uni(params, drn, grad, axes)
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

    See Also
    --------
    mat_to_params, matify
    """
    if params_or_mat.ndim >= 2:
        return params_or_mat
    return mat_to_params(params_or_mat, *args, **kwds)


# =============================================================================
# Update matrix in-place from parameters
# =============================================================================


def mat_update_params(mat: ArrayType, params: np.ndarray, *, drn: IntOrSeq = 0,
                      maxes: AxesOrSeq = (-2, -1), paxis: IntOrSeq = -1,
                      mdaxis: IntOrSeq = 0, pdaxis: IntOrSeq = 0, **kwds):
    """Change independent parameters of transition matrix.

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
    maxes : Tuple[int, int] or None
        Axes of `mat` to treat as (from, to) axes, by default: (-2, -1)
    paxis : int, optional
        Axis of `params` along which each set of parameters lie, by default -1.
    mdaxis : int, optional
        Axis of `mat` to broadcast non-scalar `drn` over, by default: 0
    pdaxis : int, optional
        Axis of `params` to broadcast non-scalar `drn` over, by default: 0

    Returns
    -------
    None
        modifies `mat` in place.

    See Also
    --------
    param_inds, mat_to_params
    """
    if not isinstance(drn, int) or not isinstance(paxis, int):
        _mh.bcast_update(_mat_update, (mat, params), drn, (maxes, paxis),
                         (mdaxis, pdaxis), **kwds)
    else:
        nst = mat.shape[maxes[0]]
        params = np.asanyarray(params)
        params = np.moveaxis(params, paxis, -1)
        if kwds.get('uniform', False):
            params = _mh.uni_to_any(params, nst, **kwds)
        nmat = flattish(np.moveaxis(mat, maxes, (-2, -1)), -2)
        nmat[param_inds(nst, **kwds)] = params
        nmat = foldaxis(nmat, -1, (nst, nst))
        _mh.stochastify(nmat)
        if not np.may_share_memory(nmat, mat):
            mat[...] = np.moveaxis(nmat, (-2, -1), maxes)


def _mat_update(arrays: _ty.Tuple[np.ndarray, np.ndarray], drn: int,
                fun_axes: _ty.Tuple[Axes, int], drn_axes: _ty.Tuple[int, int],
                **kwds):
    """call back wrapper for mat_update_params in bcast_update"""
    kwds.update(zip(('maxes', 'paxis', 'mdaxis', 'pdaxis'),
                    fun_axes + drn_axes), drn=drn)
    mat_update_params(*arrays, **kwds)


# =============================================================================
# Exports
# =============================================================================
num_param = _mh.num_param
num_state = _mh.num_state
mat_type_siz = _mh.mat_type_siz
# =============================================================================
# Type hints
# =============================================================================
Array, ArrayType, Sized, Axes = _mh.array, _mh.ArrayType, _mh.Sized, _mh.Axes
IndFun, IntOrSeq, AxesOrSeq = _mh.IndFun, _mh.IntOrSeq, _mh.AxesOrSeq
