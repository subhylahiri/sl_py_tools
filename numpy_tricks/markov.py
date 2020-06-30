# -*- coding: utf-8 -*-
"""Utilities for Markov processes
"""
from typing import Optional, Tuple
import numpy as np
import numpy_linalg as la

from . import logic as lgc


def stochastify_c(mat: la.lnarray):  # make cts time stochastic
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
    mat -= np.apply_along_axis(np.diagflat, -1, mat.sum(axis=-1))


def stochastify_d(mat: la.lnarray):  # make dscr time stochastic
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


def isstochastic_c(mat: la.lnarray, thresh: float = 1e-5) -> bool:
    """Are row sums zero?
    """
    nonneg = mat.flattish(-2) >= 0
    nonneg[..., ::mat.shape[-1]+1] = True
    return nonneg.all() and (np.fabs(mat.sum(axis=-1)) < thresh).all()


def isstochastic_d(mat: la.lnarray, thresh: float = 1e-5) -> bool:
    """Are row sums one?
    """
    return (np.fabs(mat.sum(axis=-1) - 1) < thresh).all() and (mat >= 0).all()


def rand_trans(nst: int, npl: int = 1, sparsity: float = 1.) -> la.lnarray:
    """
    Make a random transition matrix (continuous time).

    Parameters
    ----------
    n : int
        total number of states
    npl : int
        number of matrices
    sparsity : float
        sparsity

    Returns
    -------
    mat : la.lnarray
        transition matrix
    """
    mat = la.random.rand(npl, nst, nst)
    ind = la.random.rand(npl, nst, nst)
    mat[ind > sparsity] = 0.
    stochastify_c(mat)
    return mat


def calc_peq(rates: np.ndarray,
             luf: bool = False) -> Tuple[la.lnarray, Tuple[la.lnarray, ...]]:
    """Calculate steady state distribution.

    Parameters
    ----------
    rates : np.ndarray (n,n) or tuple(np.ndarray) ((n,n), (n,))
        Continuous time stochastic matrix or LU factors of inverse fundamental.
    luf : bool, optional
        Return LU factorisation of inverse fundamental as well? default: True

    Returns
    -------
    peq : la.lnarray (n,)
        Steady-state distribution.
    (z_lu, ipv) : tuple(la.lnarray) ((n,n),(n,))
        LU factors of inverse fundamental matrix.
    """
    if isinstance(rates, tuple):
        z_lu, ipv = rates
        evc = la.ones(z_lu[0].shape[0])
        peq = la.gufuncs.rlu_solve(evc.r, z_lu, ipv).ur
    else:
        evc = la.ones(rates.shape[0])
        fund_inv = la.ones_like(rates) - rates
        peq, z_lu, ipv = la.gufuncs.rsolve_lu(evc.r, fund_inv)
        peq = peq.ur
    # check for singular matrix
    if not lgc.allfinite(z_lu) or lgc.tri_low_rank(z_lu):
        peq = la.full_like(evc, np.nan)
    if luf:
        return peq, (z_lu, ipv)
    return peq


def calc_peq_d(jump: np.ndarray,
               luf: bool = False) -> Tuple[la.lnarray, Tuple[la.lnarray, ...]]:
    """Calculate steady state distribution.

    Parameters
    ----------
    jump : np.ndarray (n,n) or tuple(np.ndarray) ((n,n), (n,))
        Discrete time stochastic matrix or LU factors of inverse fundamental.

    Returns
    -------
    peq : la.lnarray (n,)
        Steady-state distribution.
    (z_lu, ipv) : tuple(la.lnarray) ((n,n),(n,))
        LU factors of inverse fundamental matrix.
    """
    if isinstance(jump, tuple):
        return calc_peq(jump, luf)
    return calc_peq(jump - la.eye(jump.shape[-1]), luf)


def adjoint(tensor: la.lnarray, measure: la.lnarray) -> la.lnarray:
    """Adjoint with respect to L2 inner product with measure

    Parameters
    ----------
    tensor : la.lnarray (...,n,n) or (...,1,n) or (...,n,1)
        The matrix/row/column vector to be adjointed.
    measure : la.lnarray (...,n)
        The measure for the inner-product wrt which we adjoint

    Parameters
    ----------
    tensor : la.lnarray (...,n,n) or (...,n,1) or (...,1,n)
        The adjoint matrix/column/row vector.
    """
    adj = tensor.t
    if adj.shape[-1] == 1:  # row -> col
        adj /= measure.c
    elif adj.shape[-2] == 1:  # col -> row
        adj *= measure.r
    else:  # mat -> mat
        adj *= measure.r / measure.c
    return adj


def mean_dwell(rates: np.ndarray, peq: Optional[np.ndarray] = None) -> float:
    """Mean time spent in any state.

    Parameters
    ----------
    rates : la.lnarray (n,n)
        Continuous time stochastic matrix.
    peq : la.lnarray (n,), optional
        Steady-state distribution, default: calculate frm `rates`.
    """
    if peq is None:
        peq = calc_peq(rates)[0]
    dwell = -1. / np.diagonal(rates)
    return 1. / (peq / dwell).sum()


def sim_markov_d(jump: la.lnarray, peq: Optional[np.ndarray] = None,
                 num_jump: int = 10) -> Tuple[la.lnarray, ...]:
    """Simulate Markov process trajectory.

    Parameters
    ----------
    jump : la.lnarray (n,n)
        Discrete time stochastic matrix.
    peq : la.lnarray (n,), optional
        Initial-state distribution, default: use steady-state.
    num_jump : int, optional, default: 10
        Stop after this many jumps.

    Returns
    -------
    states : la.lnarray (w,)
        Vector of states visited.
    """
    jump = la.asanyarray(jump)
    if peq is None:
        peq = calc_peq_d(jump)[0]

    state_inds = la.arange(len(peq))
    states_from = la.array([la.random.choice(state_inds,
                                             size=num_jump-1,
                                             p=p) for p in jump])
    states = la.empty(num_jump)
    states[0] = la.random.choice(state_inds, p=peq)
    for num in range(num_jump-1):
        states[num+1] = states_from[states[num], num]
    return states


def sim_markov(rates: la.lnarray, peq: Optional[np.ndarray] = None,
               num_jump: Optional[int] = None,
               max_time: Optional[float] = None) -> Tuple[la.lnarray, ...]:
    """Simulate Markov process trajectory.

    Parameters
    ----------
    rates : la.lnarray (n,n)
        Continuous time stochastic matrix.
    peq : la.lnarray (n,), optional
        Initial-state distribution, default: use steady-state.
    num_jump : int, optional, default: None
        Stop after this many jumps.
    max_time : float, optional, default: None
        Stop after this much time.

    Returns
    -------
    states : la.lnarray (w,)
        Vector of states visited.
    dwells : la.lnarray (w,)
        Time spent in each state.
    """
    rates = la.asanyarray(rates)
    if peq is None:
        peq = calc_peq(rates)[0]
    num_states = len(peq)
    dwell = -1. / np.diagonal(rates)
    jump = rates * dwell.c
    jump[np.diag_indices(num_states)] = 0.

    est_num = num_jump
    if num_jump is None:
        if max_time is None:
            raise ValueError("Must specify either num_jump or max_time")
        est_num = int(5 * max_time / mean_dwell(rates, peq))
    if max_time is None:
        max_time = np.inf
    est_num = max(est_num, 1)

    dwells_from = - dwell.c * np.log(la.random.rand(est_num))
    states = sim_markov_d(jump, peq, est_num)
    dwells = dwells_from[states, la.arange(est_num)]

    states, dwells = states[slice(num_jump)], dwells[slice(num_jump)]
    cum_dwell = np.cumsum(dwells)
    mask = cum_dwell < max_time
    if not mask[-1]:
        ind = np.nonzero(~mask)[0][0]
        mask[ind] = True
        dwells[ind] -= cum_dwell[ind] - max_time
    states, dwells = states[mask], dwells[mask]

    return states, dwells
