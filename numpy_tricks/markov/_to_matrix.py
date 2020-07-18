"""Utilities to convert parameters to Markov matrices
"""
from __future__ import annotations

import numpy as np

from . import _helpers as _mh
from . import indices as _in
from ._helpers import IntOrSeq
from ._helpers import array as Array
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
    return _mh.params_to_mat(params, _in.offdiag_inds, drn, axis, daxis)


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
    return _mh.params_to_mat(params, _in.offdiag_split_inds, drn, axis, daxis)


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
    return _mh.params_to_mat(params, _in.serial_inds, drn, axis, daxis)


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
    return _mh.params_to_mat(params, _in.ring_inds, drn, axis, daxis)


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
    return _mh.params_to_mat(params, _in.ind_fun(serial, ring, uniform),
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
