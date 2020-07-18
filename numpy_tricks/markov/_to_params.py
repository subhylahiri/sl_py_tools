"""Utilities to convert Markov matrices to parameters
"""
from __future__ import annotations

from . import _helpers as _mh
from . import indices as _in
from ._helpers import ArrayType, AxesOrSeq, IntOrSeq

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
    return _mh.mat_to_params(mat, _in.offdiag_inds, drn, axes, daxis)


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
    return _mh.to_uni(_mh.mat_to_params(mat, _in.offdiag_split_inds, drn, axes),
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
    return _mh.mat_to_params(mat, _in.serial_inds, drn, axes, daxis)


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
    return _mh.mat_to_params(mat, _in.ring_inds, drn, axes, daxis)


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
    params = _mh.mat_to_params(mat, _in.ind_fun(serial, ring, uniform),
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
