"""Utilities to convert parameters to Markov matrices
"""
from __future__ import annotations

import numpy as np

import numpy_linalg as la

import sl_py_tools.numpy_tricks.markov._helpers as _mh
import sl_py_tools.numpy_tricks.markov.indices as _in
from ._helpers import IntOrSeq, Array
# =============================================================================
# Parameters to matrices
# =============================================================================


def gen_params_to_mat(params: Array, drn: IntOrSeq = 0, axis: IntOrSeq = -1,
                      daxis: IntOrSeq = 0, **kwds) -> Array:
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
    offdiag_subs, gen_mat_to_params
    """
    return _mh.params_to_mat(params, _in.offdiag_subs, drn, axis, daxis, **kwds)


def uni_gen_params_to_mat(params: Array, num_st: int, drn: IntOrSeq = 0,
                          axis: IntOrSeq = -1, daxis: IntOrSeq = 0, **kwds
                          ) -> Array:
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
    offdiag_split_subs, uni_gen_mat_to_params
    """
    params = _mh.uni_to_any(params, num_st, axis=axis, **kwds)
    return _mh.params_to_mat(params, _in.offdiag_split_subs, drn, axis, daxis, **kwds)


def ring_params_to_mat(params: Array, drn: IntOrSeq = 0, axis: IntOrSeq = -1,
                       daxis: IntOrSeq = 0, **kwds) -> Array:
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
    ring_subs, ring_mat_to_params
    """
    kwds['ring'] = True
    return _mh.params_to_mat(params, _in.ring_subs, drn, axis, daxis, **kwds)


def uni_ring_params_to_mat(params: Array, num_st: int, drn: IntOrSeq = 0,
                           axis: IntOrSeq = -1, daxis: IntOrSeq = 0, **kwds
                           ) -> Array:
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
    ring_subs, uni_ring_mat_to_params
    """
    kwds['ring'] = True
    ring_params = _mh.uni_to_any(params, num_st, axis=axis, **kwds)
    return ring_params_to_mat(ring_params, drn, axis, daxis, **kwds)


def serial_params_to_mat(params: Array, drn: IntOrSeq = 0,
                         axis: IntOrSeq = -1, daxis: IntOrSeq = 0, **kwds
                         ) -> Array:
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
    serial_subs, serial_mat_to_params
    """
    kwds['serial'] = True
    return _mh.params_to_mat(params, _in.serial_subs, drn, axis, daxis, **kwds)


def uni_serial_params_to_mat(params: Array, num_st: int, drn: IntOrSeq = 0,
                             axis: IntOrSeq = -1, daxis: IntOrSeq = 0, **kwds
                             ) -> Array:
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
    serial_subs, uni_serial_mat_to_params
    """
    kwds['serial'] = True
    ser_params = _mh.uni_to_any(params, num_st, axis=axis, **kwds)
    return serial_params_to_mat(ser_params, drn, axis, daxis, **kwds)


def cascade_params_to_mat(params: Array, drn: IntOrSeq = 0,
                          axis: IntOrSeq = -1, daxis: IntOrSeq = 0, **kwds
                          ) -> Array:
    """Transition matrix with cascade topology from non-zero transition rates.

    Parameters
    ----------
    params : ndarray (2n-2,)
        Vector of elements, in order:
        mat_0n, mat_1n, ..., madrnmat_n,n+1, mat_n+1,n+2, ..., mat_2n-2,2n-1,
        mat_2n-1,n-1, ..., mat_n+1,n-1, mat_n,n-1,
        mat_n-1,n-2, ..., mat_21, mat_10.
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
    cascade_subs, cascade_mat_to_params
    """
    # cascade topology has same number of transitions as
    kwds['serial'] = True
    return _mh.params_to_mat(params, _in.cascade_subs, drn, axis, daxis,
                             **kwds)


def std_cascade_params_to_mat(params: Array, num_st: int,
                              drn: IntOrSeq = 0, axis: IntOrSeq = -1,
                              daxis: IntOrSeq = 0, **kwds) -> Array:
    """Cascade transition matrix with standard transition rates.

    Parameters
    ----------
    params : ndarray (2,)
        Vector of elements, `x` so that:
        [mat_0n, mat_1n, ..., mat_n-1,n] = [x**n-1/(1-x), x**n-2, ..., 1],
        [mat_n,n+1, ..., mat_2n-2,2n-1] = [x/(1-x), ..., x**n-2/(1-x)],
        mat_2n-1,n-1, ..., mat_n+1,n-1, mat_n,n-1 = [x**n-1/(1-x), ..., 1],
        mat_n-1,n-2, ..., mat_21, mat_10 = [x/(1-x), ..., x**n-2/(1-x)].
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
    cascade_subs, cascade_params_to_mat
    """
    if not isinstance(axis, int):
        return _mh.bcast_axes(std_cascade_params_to_mat, params, num_st,
                              drn=drn, drn_axis=daxis, fun_axis=axis,
                              to_mat=True, **kwds)
    npt = num_st // 2
    # (...,2,1)
    params = la.asarray(params).moveaxis(axis, -1)[..., None]
    # (n-1,)
    expn = np.abs(np.arange(1 - npt, npt))
    denom = np.r_[0, npt:2*npt-1]
    # (...,2,n-1)
    full = params**expn
    full[..., denom] /= (1 - params)
    # (...,2(n-1)) -> (...,2(n-1),...)
    full = full.ravelaxes(-2).moveaxis(-1, axis)
    return cascade_params_to_mat(full, drn=drn, axis=axis, daxis=daxis, **kwds)


def params_to_mat(params: Array, *, serial: bool = False,
                  ring: bool = False, uniform: bool = False, nst: int = 2,
                  drn: IntOrSeq = 0, axis: IntOrSeq = -1, daxis: IntOrSeq = 0,
                  **kwds) -> Array:
    """Transition matrix from independent parameters.

    Parameters
    ----------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_subs` for details.
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
    param_subs, mat_to_params
    """
    kwds.update({'serial': serial, 'ring': ring})
    if uniform:
        params = _mh.uni_to_any(params, nst, axis=axis, **kwds)
    return _mh.params_to_mat(params, _in.sub_fun(serial, ring, uniform),
                             drn, axis, daxis, **kwds)


def matify(params_or_mat: Array, *args, **kwds) -> Array:
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
