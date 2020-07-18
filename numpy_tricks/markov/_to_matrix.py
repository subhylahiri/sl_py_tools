"""Utilities to convert parameters to Markov matrices
"""
from __future__ import annotations

import numpy as np

import numpy_linalg as la

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


def cascade_params_to_mat(params: np.ndarray, drn: IntOrSeq = 0,
                          axis: IntOrSeq = -1, daxis: IntOrSeq = 0) -> Array:
    """Cascade transition matrix from transition rates.

    Parameters
    ----------
    params : ndarray (2n-2,)
        Vector of elements, in order:
        mat_0n, mat_1n, ..., mat_n-1,n,
        mat_n,n+1, mat_n+1,n+2, ..., mat_2n-2,2n-1,
        mat_10, mat_21, ..., mat_n-1,n-2,
        mat_n,n-1, mat_n+1,n-1, ..., mat_2n-1,n-1.
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
    cascade_inds, cascade_mat_to_params
    """
    return _mh.params_to_mat(params, _in.cascade_inds, drn, axis, daxis)


def std_cascade_params_to_mat(params: np.ndarray, num_st: int,
                              drn: IntOrSeq = 0, axis: IntOrSeq = -1,
                              daxis: IntOrSeq = 0) -> Array:
    """Cascade transition matrix from transition rates.

    Parameters
    ----------
    params : ndarray (2,)
        Vector of elements, `x` so that:
        [mat_0n, mat_1n, ..., mat_n-1,n] = [x**n-1/(1-x), x**n-2, ..., 1]
        [mat_n,n+1, ..., mat_2n-2,2n-1] = [x/(1-x), ..., x**n-2/(1-x)],
        [mat_10, ..., mat_n-1,n-2] = [x**n-2/(1-x), ..., x/(1-x)],
        [mat_n,n-1, ..., mat_2n-1,n-1] = [1, ..., x**n-2, x**n-1/(1-x)].
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
    cascade_inds, cascade_params_to_mat
    """
    npt = num_st // 2
    # (...,2,1)
    params = la.asanyarray(params).moveaxis(axis, -1)[..., None]
    expn = np.abs(np.arange(1 - npt, npt))
    params = params**expn
    if drn >= 0:
        params[..., :1, :1] /= (1 - params[..., :1, npt-1:npt])
        params[..., :1, npt:] /= (1 - params[..., :1, npt-1:npt])
    if drn <= 0:
        params[..., -1:, -1:] /= (1 - params[..., -1:, npt-1:npt])
        params[..., -1:, :npt-1] /= (1 - params[..., -1:, npt-1:npt])
    params = params.reshape(params.shape[:-2] + (-1,)).moveaxis(-1, axis)
    return cascade_params_to_mat(params, drn=drn, axis=axis, daxis=daxis)


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
