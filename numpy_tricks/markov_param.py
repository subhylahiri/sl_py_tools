# -*- coding: utf-8 -*-
"""Utilities for Markov processes
"""
import typing as _ty
import numpy as np
import numpy_linalg as la
from .markov import stochastify_c
Sized = _ty.Union[int, np.ndarray]


def offdiag_inds(nst: int, drn: int = 0) -> la.lnarray:
    """Ravel indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : la.lnarray (n(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
    """
    # when put into groups of size nst+1:
    # M[0,0], M[0,1], M[0,2], ..., M[0,n-1], M[1,0],
    # M[1,1], M[1,2], ..., M[1,n-1], M[2,0], M[2,1],
    # ...
    # M[n-2,n-2], M[n-2,n-1], M[n-1,0], ..., M[n-1,n-2],
    # unwanted elements are 1st in each group
    if drn > 0:
        return np.triu_indices(nst, 1)
    if drn < 0:
        return np.tril_indices(nst, -1)
    k_1st = la.arange(0, nst**2, nst+1)  # ravel ind of 1st element in group
    k = la.arange(nst**2)
    return la.delete(k, k_1st)


def serial_inds(nst: int, drn: int = 0) -> la.lnarray:
    """Ravel indices of independent parameters of serial transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : la.lnarray (2(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-2,n-1.
    """
    pos = np.arange(1, nst**2, nst+1)
    if drn > 0:
        return pos
    neg = np.arange(nst, nst**2, nst+1)
    if drn < 0:
        return neg
    return la.hstack((pos, neg))


def ring_inds(nst: int, drn: int = 0) -> la.lnarray:
    """Ravel indices of independent parameters of ring transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : la.lnarray (2n,)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    """
    pos = la.hstack((np.arange(1, nst**2, nst+1), [nst*(nst-1)]))
    if drn > 0:
        return pos
    neg = la.hstack(([nst-1], np.arange(nst, nst**2, nst+1)))
    if drn < 0:
        return neg
    return la.hstack((pos, neg))


def param_inds(nst: int, serial: bool = False, ring=False,
               drn: int = 0) -> la.lnarray:
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
    mat : la.lnarray (k,k), k in (n(n-1), 2(n-1), 2n, 2)
        Matrix of independent elements, each axis in order of `mat_to_params`.
    """
    if serial:
        return serial_inds(nst, drn)
    if ring:
        return ring_inds(nst, drn)
    return offdiag_inds(nst, drn)


def num_param(states: Sized, serial=False, ring=False, uniform=False,
              drn: int = 0) -> int:
    """Number of independent rates

    Parameters
    ----------
    states : int or np.ndarray (n,...)
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
    if uniform:
        return 2
    mult = 2 - bool(drn)  # double if drn == 0
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
    params : int or np.ndarray (n,)
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
    params : int or np.ndarray (np,)
        Number of rate parameters, or vector of rates.
    states : int or np.ndarray (n,...)
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
        Is the rate vector meant for `ring_params_to_mat` or
        `uni_ring_params_to_mat`?
    """
    if isinstance(params, np.ndarray):
        params = params.shape[-1]
    if isinstance(states, np.ndarray):
        states = states.shape[-1]
    uniform = params in {1, 2}
    drn = states in {states, states - 1, states * (states - 1) // 2}
    ring = uniform or (params in {2 * states, states})
    serial = uniform or (params in {2 * (states - 1), states - 1})
    return serial, ring, drn, uniform


def _params_to_mat(fun, params, nst, drn):
    """Helper function for *_params_to_mat

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    """
    mat = la.empty(nst**2)
    mat[fun(nst, drn)] = params
    mat.resize((nst, nst))
    stochastify_c(mat)
    return mat


def _uni_params(params, npar):
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    npar
        Number of parameters when `dir=+/-1.`
    """
    if len(params) == 1:
        return np.full(npar, params[0])
    return la.hstack((np.full(npar, params[0]), np.full(npar, params[1])))


def gen_params_to_mat(params: np.ndarray, drn: int = 0) -> la.lnarray:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (n(n-1),)
        Vector of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    nst = num_state(params, serial=True, drn=drn)
    return _params_to_mat(offdiag_inds, params, nst, drn)


def serial_params_to_mat(params: np.ndarray, drn: int = 0) -> la.lnarray:
    """Serial transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (2(n-1),)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-1,n-2.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    nst = num_state(params, serial=True, drn=drn)
    return _params_to_mat(serial_inds, params, nst, drn)


def uni_serial_params_to_mat(params: np.ndarray, num_st: int,
                             drn: int = 0) -> la.lnarray:
    """Uniform serial transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (2,)
        Vector of independent elements, in order:
        mat_01 = mat_12 = ... = mat_n-2,n-1,
        mat_10 = mat_21 = ... = mat_n-1,n-2.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    return serial_params_to_mat(_uni_params(params, num_st-1), drn)


def ring_params_to_mat(params: np.ndarray, drn: int = 0) -> la.lnarray:
    """Ring transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (2n,)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    nst = num_state(params.size, ring=True, drn=drn)
    return _params_to_mat(ring_inds, params, nst, drn)


def uni_ring_params_to_mat(params: np.ndarray, num_st: int,
                           drn: int = 0) -> la.lnarray:
    """Ring transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (2,)
        Vector of independent elements, in order:
        mat_01 = mat_12 = ... = mat_n-2,n-1 = mat_n-1,0,
        mat_0,n-1 = mat_10 = mat_21 = ... = mat_n-1,n-2.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    return ring_params_to_mat(_uni_params(params, num_st), drn)


def params_to_mat(params: np.ndarray,
                  serial: bool = False, ring: bool = False, drn: int = 0,
                  uniform: bool = False, nst: int = 2) -> la.lnarray:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
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
    nst : int, optional, default: 2
        Number of states. Only needed when `uniform` is True

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    if serial:
        if uniform:
            return uni_serial_params_to_mat(params, nst, drn)
        return serial_params_to_mat(params, drn)
    if ring:
        if uniform:
            return uni_ring_params_to_mat(params, nst, drn)
        return ring_params_to_mat(params, drn)
    return gen_params_to_mat(params, drn)


def _mat_to_params(fun, mat, drn):
    """Helper function for *_params_to_mat

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    """
    nst = mat.shape[0]
    param = mat.flatten()
    return param[fun(nst, drn)]


def _uni_mat(params, drn, grad):
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    npar
        Number of parameters when `dir=+/-1.`
    """
    npar = len(params)
    if drn:
        npar //= 2
        params = la.hstack([params[:npar].sum(), params[npar:].sum()])
    else:
        params = params.sum()
    if not grad:
        params /= npar
    return params


def gen_mat_to_params(mat: np.ndarray, drn: int = 0) -> la.lnarray:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    params : la.lnarray (n(n-1),)
        Vector of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-2,n-1.
    """
    return _mat_to_params(offdiag_inds, mat, drn)


def serial_mat_to_params(mat: np.ndarray, drn: int = 0) -> la.lnarray:
    """Independent parameters of serial transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    params : la.lnarray (2(n-1),)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-2,n-1.
    """
    return _mat_to_params(serial_inds, mat, drn)


def uni_serial_mat_to_params(mat: np.ndarray, grad: bool = True,
                             drn: int = 0) -> la.lnarray:
    """Independent parameters of uniform serial transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of left/right transitions.
        If False, return the mean.

    Returns
    -------
    params : la.lnarray (2,)
        Vector of independent elements, in order (grad=False):
            mat_01 = mat_12 = ... = mat_n-2,n-1,
            mat_10 = mat_21 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
            mat_01 + mat_12 + ... + mat_n-2,n-1,
            mat_10 + mat_21 + ... + mat_n-1,n-2.
    """
    return _uni_mat(serial_mat_to_params(mat, drn), drn, grad)


def ring_mat_to_params(mat: np.ndarray, drn: int = 0) -> la.lnarray:
    """Independent parameters of ring transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    params : la.lnarray (2n,)
        Vector of independent elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    """
    return _mat_to_params(ring_inds, mat, drn)


def uni_ring_mat_to_params(mat: np.ndarray, grad: bool = True,
                           drn: int = 0) -> la.lnarray:
    """Independent parameters of ring transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool, optional, default: True
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of (anti)clockwise transitions.
        If False, return the mean.

    Returns
    -------
    params : la.lnarray (2,)
        Vector of independent elements, in order (grad=False):
            mat_01 = mat_12 = ... = mat_n-2,n-1 = mat_n-10,
            mat_0n-1 = mat10 = mat_21 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
            mat_01 + mat_12 + ... + mat_n-2,n-1 + mat_n-10,
            mat_0n-1 + mat10 + mat_21 + ... + mat_n-1,n-2.
    """
    return _uni_mat(ring_mat_to_params(mat, drn), drn, grad)


def mat_to_params(mat: np.ndarray,
                  serial: bool = False, ring: bool = False, drn: int = 0,
                  uniform: bool = False, grad: bool = True) -> la.lnarray:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
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

    Returns
    -------
    params : la.lnarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements. For the order, see `*_mat_to_params`.
    """
    if serial:
        if uniform:
            return uni_serial_mat_to_params(mat, grad, drn)
        return serial_mat_to_params(mat, drn)
    if ring:
        if uniform:
            return uni_ring_mat_to_params(mat, grad, drn)
        return ring_mat_to_params(mat, drn)
    return gen_mat_to_params(mat, drn)


def mat_update_params(mat: np.ndarray, params: np.ndarray,
                      uniform: bool = False, **kwds) -> la.lnarray:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
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

    Returns
    -------
    None
        modifies `mat` in place.
    """
    nst = mat.shape[-1]
    if uniform:
        params = _uni_params(params, num_param(nst, **kwds))
    inds = param_inds(nst, **kwds)
    mat.ravel()[inds] = params


def tens_to_mat(tens: np.ndarray,
                serial: bool = False, ring: bool = False, drn: int = 0,
                uniform: bool = False, grad: bool = True) -> la.lnarray:
    """Independent parameters of 4th rank tensor.

    Parameters
    ----------
    tens : np.ndarray (n,n,n,n)
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

    Returns
    -------
    mat : la.lnarray (n**2,n**2)
        Matrix of independent elements, each axis in order `*_mat_to_params`.
    """
    nst = tens.shape[0]
    mat = tens.reshape((nst**2, nst**2))
    inds = param_inds(nst, serial=serial, ring=ring, drn=drn)
    mat = mat[np.ix_(inds, inds)]
    if uniform:
        nind = len(inds) // 2
        mat = la.block([[mat[:nind, :nind].sum(), mat[:nind, nind:].sum()],
                        [mat[nind:, :nind].sum(), mat[nind:, nind:].sum()]])
        if not grad:
            mat /= nind
    return mat
