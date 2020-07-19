"""Generate indices for parameters of Markov processes
"""
from __future__ import annotations

import numpy as np

from ._helpers import IndFun

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
    """Ravel indices of non-zero elements of serial transition matrix.

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
    """Ravel indices of non-zero elements of ring transition matrix.

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


def cascade_inds(nst: int, drn: int = 0) -> np.ndarray:
    """Indices of transitions for the cascade model

    Parameters
    ----------
    nst : int
        Number of states, `2n`.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    K : ndarray (2(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_0n, mat_1n, ..., mat_n-1,n,
        mat_n,n+1, mat_n+1,n+2, ... mat_2n-2,2n-1,
        mat_2n-1,n-1, ..., mat_n+1,n-1, mat_n,n-1,
        mat_n-1,n-2, ..., mat_21, mat_10.
    """
    npt = nst // 2
    pos = np.r_[npt:nst*npt:nst, nst*npt + npt + 1:nst**2:nst + 1]
    if drn > 0:
        return pos
    neg = np.r_[nst**2 - npt - 1:nst*npt:-nst, nst*npt - npt - 2:0:-nst - 1]
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def ind_fun(serial: bool, ring: bool, uniform: bool = False) -> IndFun:
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
    return ind_fun(serial, ring, uniform)(nst, drn)
