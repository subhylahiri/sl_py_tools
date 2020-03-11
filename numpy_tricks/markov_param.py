# -*- coding: utf-8 -*-
"""Utilities for parametrising Markov processes

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
triangle are equal. When extracting parameters we average them, unless
`grad` is True when we take the sum.
"""
import typing as _ty
import numpy as np
import numpy_linalg as la
from .markov import stochastify_c
Sized = _ty.Union[int, np.ndarray]


def offdiag_inds(nst: int, drn: int = 0) -> la.lnarray:
    """Indices of independent parameters of transition matrix.

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
        mat_10, mat_21, ..., mat_n-1,n-2.
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


def _ind_fun(serial, ring):
    """which index function to use"""
    if serial:
        return serial_inds
    if ring:
        return ring_inds
    return offdiag_inds


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
        Indices of independent elements. For the order, see docs for `*_inds`.
    """
    return _ind_fun(serial, ring)(nst, drn)


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
        Is the rate vector meant for `*_params_to_mat` or
        `uni_*_params_to_mat`? * = serial or ring.
    """
    if isinstance(params, np.ndarray):
        params = params.shape[-1]
    if isinstance(states, np.ndarray):
        states = states.shape[-1]
    uniform = params in {1, 2}
    drn = states in {1, states, states - 1, states * (states - 1) // 2}
    ring = uniform or (params in {2 * states, states})
    serial = uniform or (params in {2 * (states - 1), states - 1})
    return serial, ring, drn, uniform


def mat_type_dict(params: Sized, states: Sized) -> _ty.Tuple[bool, ...]:
    """Is it a (uniform) ring

    Parameters
    ----------
    params : int or np.ndarray (np,)
        Number of rate parameters, or vector of rates.
    states : int or np.ndarray (n,...)
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
            Is the rate vector meant for `*_params_to_mat` or
            `uni_*_params_to_mat`? * = serial or ring.
    """
    serial, ring, drn, uniform = mat_type(params, states)
    return {'serial': serial, 'ring': ring, 'drn': drn, 'uniform': uniform}


def _params_to_mat(fun, params, nst, drn):
    """Helper function for *_params_to_mat

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    params : np.ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    nst : int
        Number of states.
    drn: int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    mat = la.empty(nst**2)
    mat[fun(nst, drn)] = params
    mat.resize((nst, nst))
    stochastify_c(mat)
    return mat


def _uni_params(params, nst, serial: bool = False, ring: bool = False):
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    params : np.ndarray (1,) or (2,)
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

    Returns
    -------
    params : np.ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags above.
        See docs for `*_inds` for details.
    """
    npar = num_param(nst, serial=serial, ring=ring, drn=1, uniform=False)
    return la.hstack([np.full(npar, prm) for prm in params])


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
    nst = num_state(params, drn=drn)
    return _params_to_mat(offdiag_inds, params, nst, drn)


def uni_gen_params_to_mat(params: np.ndarray, num_st: int,
                          drn: int = 0) -> la.lnarray:
    """Uniform transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (2,) or (1,)
        Vector of independent elements, in order:
        mat_01 = ... = mat_0n-1 = mat_12 = ... mat_1n-1 = ... = mat_n-2,n-1,
        mat_10 = mat_20 = mat_21 = mat_30 = ... = mat_n-10 = ... = mat_n-1,n-2.
    num_st : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    gen_params = _uni_params(params, num_st)
    if drn:
        return gen_params_to_mat(gen_params, drn)
    # pylint: disable=unbalanced-tuple-unpacking
    pos, neg = np.split(gen_params, 2)
    return gen_params_to_mat(pos, 1) + gen_params_to_mat(neg, -1)


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
    return serial_params_to_mat(_uni_params(params, num_st, serial=True), drn)


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
    return ring_params_to_mat(_uni_params(params, num_st, ring=True), drn)


def params_to_mat(params: np.ndarray,
                  serial: bool = False, ring: bool = False, drn: int = 0,
                  uniform: bool = False, nst: int = 2) -> la.lnarray:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : np.ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
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
    if uniform:
        params = _uni_params(params, nst, serial=serial, ring=ring)
    else:
        nst = num_state(params, serial=serial, ring=ring, drn=drn)
    return _params_to_mat(_ind_fun(serial, ring), params, nst, drn)


def matify(params_or_mat: np.ndarray, *args, **kwds) -> la.lnarray:
    """Transition matrix from independent parameters, if not already so.

    Parameters
    ----------
    params_or_mat : np.ndarray (np,) or (n,n)
        Either vector of independent elements (in order that depends on flags,
        see docs for `params_to_mat`) or continuous time stochastic matrix.
    other arguments passed to `params_to_mat`

    Returns
    -------
    mat : la.lnarray (n,n)
        Continuous time stochastic matrix.
    """
    if params_or_mat.ndim >= 2:
        return params_or_mat
    return params_to_mat(params_or_mat, *args, **kwds)


def _mat_to_params(fun, mat, drn):
    """Helper function for *_mat_to_params

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->inds`.
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    drn: int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    """
    nst = mat.shape[0]
    param = mat.flatten()
    return param[fun(nst, drn)]


def _uni_mat(params, drn, grad):
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    params : np.ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of <-
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    drn: int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of (anti)clockwise transitions.
        If False, return the mean.
    """
    npar = len(params) / (1 + (drn == 0))
    params = params.reshape((-1, npar)).sum(axis=-1)
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


def uni_gen_mat_to_params(mat: np.ndarray, grad: bool = True,
                          drn: int = 0) -> la.lnarray:
    """Independent parameters of uniform transition matrix.

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
        mat_01 = ... = mat_0n-1 = mat_12 = ... mat_1n-1 = ... = mat_n-2,n-1,
        mat_10 = mat_20 = mat_21 = mat_30 = ... = mat_n-10 = ... = mat_n-1,n-2.
        Or, in order (grad=True):
        mat_01 + ... + mat_0n-1 + mat_12 + ... mat_1n-1 + ... + mat_n-2,n-1,
        mat_10 + mat_20 + mat_21 + mat_30 + ... + mat_n-10 + ... + mat_n-1,n-2.
    """
    if drn:
        return _uni_mat(gen_mat_to_params(mat, drn), drn, grad)
    params_pos = gen_mat_to_params(mat, 1)
    params_neg = gen_mat_to_params(mat, -1)
    return _uni_mat(np.hstack((params_pos, params_neg)), drn, grad)


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
    params : la.lnarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of them
        Vector of independent elements. For the order, see docs for `*_inds`.
    """
    params = _mat_to_params(_ind_fun(serial, ring), mat, drn)
    if uniform:
        return _uni_mat(params, drn, grad)
    return params


def paramify(params_or_mat: np.ndarray, *args, **kwds) -> np.ndarray:
    """Independent parameters of transition matrix, if not already so.

    Parameters
    ----------
    params_or_mat : np.ndarray (np,) or (n,n)
        Either vector of independent elements (in order that depends on flags,
        see docs for `params_to_mat`) or continuous time stochastic matrix.
    other arguments passed to `mat_to_params`

    Returns
    -------
    params : np.ndarray (np,)
        Vector of independent elements (in order that depends on flags,
        see docs for `*_inds` for details).
    """
    if params_or_mat.ndim >= 2:
        return params_or_mat
    return mat_to_params(params_or_mat, *args, **kwds)


def mat_update_params(mat: np.ndarray, params: np.ndarray,
                      **kwds) -> la.lnarray:
    """Independent parameters of transition matrix.

    Parameters
    ----------
    mat : np.ndarray (n,n)
        Continuous time stochastic matrix.
    params : np.ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,) or half of them
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
        params = _uni_params(params, num_param(nst, **kwds))
    inds = param_inds(nst, **kwds)
    mat.ravel()[inds] = params
    stochastify_c(mat)


def tens_to_mat(tens: np.ndarray,
                serial: bool = False, ring: bool = False, drn: int = (0, 0),
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
    inds = [param_inds(nst, serial=serial, ring=ring, drn=d) for d in drn]
    mat = mat[np.ix_(inds[0], inds[1])]
    if uniform:
        nind = len(inds[0]) // 2
        mat = la.block([[mat[:nind, :nind].sum(), mat[:nind, nind:].sum()],
                        [mat[nind:, :nind].sum(), mat[nind:, nind:].sum()]])
        if drn[0]:
            mat = mat.sum()
            nind = len(inds[0])
        if not grad:
            mat /= nind**2
    return mat
