"""Generate indices for parameters of Markov processes
"""
from __future__ import annotations

import typing as _ty

import numpy as np

from ._helpers import IndFun, IntOrSeq, bcast_drn_inds

# =============================================================================
# Indices of parameters
# =============================================================================


def offdiag_inds(nst: int, drn: IntOrSeq = 0) -> np.ndarray:
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
    if not isinstance(drn, int):
        return bcast_drn_inds(offdiag_inds, nst, drn)
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


def offdiag_split_inds(nst: int, drn: IntOrSeq = 0) -> np.ndarray:
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
    if not isinstance(drn, int):
        return bcast_drn_inds(offdiag_split_inds, nst, drn)
    if drn:
        return offdiag_inds(nst, drn)
    return np.hstack((offdiag_inds(nst, 1), offdiag_inds(nst, -1)))


def ring_inds(nst: int, drn: IntOrSeq = 0) -> np.ndarray:
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
    if not isinstance(drn, int):
        return bcast_drn_inds(ring_inds, nst, drn)
    pos = np.r_[1:nst**2:nst+1, nst*(nst-1)]
    if drn > 0:
        return pos
    neg = np.r_[nst-1, 1:nst**2:nst+1]
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def serial_inds(nst: int, drn: IntOrSeq = 0) -> np.ndarray:
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
    if not isinstance(drn, int):
        return bcast_drn_inds(serial_inds, nst, drn)
    pos = np.arange(1, nst**2, nst+1)
    if drn > 0:
        return pos
    neg = np.arange(nst, nst**2, nst+1)
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def cascade_inds(nst: int, drn: IntOrSeq = 0) -> np.ndarray:
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
    if not isinstance(drn, int):
        return bcast_drn_inds(cascade_inds, nst, drn)
    npt = nst // 2
    pos = np.r_[npt:nst*npt:nst, nst*npt + npt + 1:nst**2:nst + 1]
    if drn > 0:
        return pos
    neg = np.r_[nst**2 - npt - 1:nst*npt:-nst, nst*npt - npt - 2:0:-nst - 1]
    if drn < 0:
        return neg
    return np.hstack((pos, neg))


def ind_fun(serial: bool, ring: bool, uniform: bool = False) -> IndFun:
    """Which index function to use

    Parameters
    ----------
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?

    Returns
    -------
    ifun : Callable[[int, int] -> ndarray[int]]
        Function that computes ravelled indices of nonzero elements from
        number of states and direction.
    """
    if serial:
        return serial_inds
    if ring:
        return ring_inds
    if uniform:
        return offdiag_split_inds
    return offdiag_inds


def param_inds(nst: int, serial: bool = False, ring: bool = False,
               uniform: bool = False, drn: IntOrSeq = 0) -> np.ndarray:
    """Ravel indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
    inds : ndarray (k,), k in (n(n-1), 2(n-1), 2n, 2)
        Indices of independent elements. For the order, see docs for `*_inds`.
    """
    return ind_fun(serial, ring, uniform)(nst, drn)


def _unravel_ind_fun(func: IndFun) -> SubFun:
    """Convert a function that returns ravelled indices to one that returns
    unravelled indices.

    Parameters
    ----------
    func : Callable[[int, int], np.ndarray]
        Function that returns ravelled indices

    Returns
    -------
    new_func : Callable[[int, int], Tuple[np.ndarray, np.ndarray]]
        Function that returns unravelled indices
    """
    def new_func(nst: int, drn: IntOrSeq = 0) -> Subs:
        """Row and column indices of transitions

        Parameters
        ----------
        nst : int
            Number of states, `M`.
        drn: int|Sequence[int], optional, default: 0
            If `drn`, only include transitions in direction `i -> i+sgn(drn)`.
            If it is a sequence of length `P`, return the subscripts for a
            `(P,M,M)` array of matrices

        Returns
        -------
        mats : ndarray
            Which transition matrix, in a `(P,M,M)` array of matrices?
            Not returned if `drn` is an `int`.
        rows : ndarray
            Vector of row indices of off-diagonal elements.
        cols : ndarray
            Vector of column indices of off-diagonal elements.
        For the order, see docs for `*_inds`.
        """
        if isinstance(drn, int):
            inds = func(nst, drn)
            return np.unravel_index(inds, (nst, nst))
        mats, rows, cols = [], [], []
        for mat, dirn in enumerate(drn):
            row_col = new_func(nst, dirn)
            mats.append(np.full_like(row_col[0], mat))
            rows.append(row_col[0])
            cols.append(row_col[1])
        return np.concatenate(mats), np.concatenate(rows), np.concatenate(cols)
    return new_func


def sub_fun(serial: bool, ring: bool, uniform: bool = False) -> SubFun:
    """which index function to use

    Parameters
    ----------
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?

    Returns
    -------
    sfun : Callable[[int, int] -> Tuple[ndarray[int], ndarray[int]]]
        Function that computes unravelled indices of nonzero elements from
        number of states and direction.
    """
    return _unravel_ind_fun(ind_fun(serial, ring, uniform))


def param_subs(nst: int, serial: bool = False, ring: bool = False,
               uniform: bool = False, drn: IntOrSeq = 0) -> Subs:
    """Row and column indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    serial : bool, optional, default: False
        Is the rate vector meant for a model with the serial topology?
    ring : bool, optional, default: False
        Is the rate vector meant for a model with the ring topology?
    uniform : bool, optional, default: False
        Do the nonzero transition rates (in one direction) have the same value?
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.

    Returns
    -------
        rows : ndarray
            Vector of row indices of off-diagonal elements.
        cols : ndarray
            Vector of column indices of off-diagonal elements.
        For the order, see docs for `*_inds`.
    """
    return sub_fun(serial, ring, uniform)(nst, drn)


# =============================================================================
# Aliases
# =============================================================================
offdiag_subs = _unravel_ind_fun(offdiag_inds)
offdiag_split_subs = _unravel_ind_fun(offdiag_split_inds)
ring_subs = _unravel_ind_fun(ring_inds)
serial_subs = _unravel_ind_fun(serial_inds)
cascade_subs = _unravel_ind_fun(cascade_inds)
Subs = _ty.Tuple[np.ndarray, ...]
SubFun = _ty.Callable[[int, int], Subs]
