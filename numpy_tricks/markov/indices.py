"""Generate indices for parameters of Markov processes
"""
from __future__ import annotations

import numpy as np

from ._helpers import IndFun, IntOrSeq, bcast_inds, Subs, SubFun, stack_inds

# =============================================================================
# Indices of parameters
# =============================================================================


def offdiag_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    K : ndarray (n(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_02, ..., mat_0n-1, mat10, mat_12, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return bcast_inds(offdiag_inds, nst, drn, ravel)
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
    # k_1st = np.arange(0, nst**2, nst+1)  # ravel ind of 1st element in group
    # k = np.arange(nst**2)
    return np.delete(np.arange(nst**2), np.s_[::nst+1])


def offdiag_split_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True
                       ) -> np.ndarray:
    """Indices of independent parameters of transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

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
        return bcast_inds(offdiag_split_inds, nst, drn, ravel)
    if drn:
        return offdiag_inds(nst, drn)
    return stack_inds(offdiag_inds, nst)


def ring_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Ravel indices of non-zero elements of ring transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    K : ndarray (2n,)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1, mat_n-1,0,
        mat_0,n-1, mat_10, mat_21, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return bcast_inds(ring_inds, nst, drn, ravel)
    if drn > 0:
        return np.r_[1:nst**2:nst+1, nst*(nst-1)]
    if drn < 0:
        return np.r_[nst-1, 1:nst**2:nst+1]
    return stack_inds(ring_inds, nst)


def serial_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Ravel indices of non-zero elements of serial transition matrix.

    Parameters
    ----------
    nst : int
        Number of states.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    K : ndarray (2(n-1),)
        Vector of ravel indices of off-diagonal elements, in order:
        mat_01, mat_12, ..., mat_n-2,n-1,
        mat_10, mat_21, ..., mat_n-1,n-2.
    """
    if not isinstance(drn, int):
        return bcast_inds(serial_inds, nst, drn, ravel)
    if drn > 0:
        return np.arange(1, nst**2, nst+1)
    if drn < 0:
        return np.arange(nst, nst**2, nst+1)
    return stack_inds(serial_inds, nst)


def cascade_inds(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
    """Indices of transitions for the cascade model

    Parameters
    ----------
    nst : int
        Number of states, `2n`.
    drn: int, optional, default: 0
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

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
        return bcast_inds(cascade_inds, nst, drn, ravel)
    npt = nst // 2
    if drn > 0:
        return np.r_[npt:nst*npt:nst, nst*npt + npt + 1:nst**2:nst+1]
    if drn < 0:
        return np.r_[nst**2 - npt - 1:nst*npt:-nst, nst*npt - npt - 2:0:-nst-1]
    return stack_inds(cascade_inds, nst)


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
    def new_func(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
        """Row and column indices of transitions

        Parameters
        ----------
        nst : int
            Number of states, `M`.
        drn: int|Sequence[int], optional, default: 0
            If `drn`, only include transitions in direction `i -> i+sgn(drn)`.
            If it is a sequence of length `P`, return the subscripts for a
            `(P,M,M)` array of matrices
        ravel : bool, optional
            Return a ravelled array, or use first axis for different matrices
            if `drn` is a sequence. By default `True`.

        Returns
        -------
        [mats : ndarray (PQ,)
            Which transition matrix, in a `(P,M,M)` array of matrices?
            Not returned if `drn` is an `int`.]
        rows : ndarray (PQ,)
            Vector of row indices of off-diagonal elements.
        cols : ndarray (PQ,)
            Vector of column indices of off-diagonal elements.
        For the order of elements, see docs for `*_inds`.
        """
        shape = (nst, nst) if isinstance(drn, int) else (len(drn), nst, nst)
        inds = func(nst, drn, ravel)
        return np.unravel_index(inds, shape)
    return new_func


def _ravel_sub_fun(func: SubFun) -> IndFun:
    """Convert a function that returns unravelled indices to one that returns
    ravelled indices.

    Parameters
    ----------
    func : Callable[[int, int], Tuple[np.ndarray, np.ndarray]]
        Function that returns unravelled indices

    Returns
    -------
    new_fun : Callable[[int, int], np.ndarray]
        Function that returns ravelled indices
    """
    def new_fun(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> np.ndarray:
        """Row and column indices of transitions

        Parameters
        ----------
        nst : int
            Number of states, `M`.
        drn: int|Sequence[int], optional, default: 0
            If `drn`, only include transitions in direction `i -> i+sgn(drn)`.
            If it is a sequence of length `P`, return the subscripts for a
            `(P,M,M)` array of matrices
        ravel : bool, optional
            Return a ravelled array, or use first axis for different matrices
            if `drn` is a sequence. By default `True`.

        Returns
        -------
        cols : ndarray (PQ,)
            Vector of ravelled indices of off-diagonal elements.
        For the order of elements, see docs for `*_inds`.
        """
        shape = (nst, nst) if isinstance(drn, int) else (len(drn), nst, nst)
        subs = func(nst, drn, ravel)
        return np.ravel_multi_index(subs, shape)
    return new_fun


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
    if serial:
        return serial_subs
    if ring:
        return ring_subs
    if uniform:
        return offdiag_split_subs
    return offdiag_subs


def param_inds(nst: int, *, serial: bool = False, ring: bool = False,
               uniform: bool = False, drn: IntOrSeq = 0, ravel: bool = True
               ) -> np.ndarray:
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
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
    inds : ndarray (k,), k in (n(n-1), 2(n-1), 2n, 2)
        Indices of independent elements. For the order, see docs for `*_inds`.
    """
    return ind_fun(serial, ring, uniform)(nst, drn, ravel)


def param_subs(nst: int, *, serial: bool = False, ring: bool = False,
               uniform: bool = False, drn: IntOrSeq = 0, ravel: bool = True
               ) -> Subs:
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
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices if
        `drn` is a sequence. By default `True`.

    Returns
    -------
        rows : ndarray
            Vector of row indices of off-diagonal elements.
        cols : ndarray
            Vector of column indices of off-diagonal elements.
        For the order, see docs for `*_inds`.
    """
    return sub_fun(serial, ring, uniform)(nst, drn, ravel)


# =============================================================================
# Aliases
# =============================================================================
offdiag_subs = _unravel_ind_fun(offdiag_inds)
offdiag_split_subs = _unravel_ind_fun(offdiag_split_inds)
ring_subs = _unravel_ind_fun(ring_inds)
serial_subs = _unravel_ind_fun(serial_inds)
cascade_subs = _unravel_ind_fun(cascade_inds)
