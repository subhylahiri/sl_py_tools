# -*- coding: utf-8 -*-
"""Utilities for module markov_param
"""
from __future__ import annotations

import functools
import typing as _ty
from collections.abc import Iterable, Sequence

import numpy as np

import sl_py_tools.containers as _cn

# =============================================================================
# Utilities
# =============================================================================


def diff_like(fun: _ty.Callable[[Array, Array], Array],
              arr: Array, step: int = 1, axis: int = -1) -> Array:
    """Perform an operation on adjacent elements in an array

    Parameters
    ----------
    fun : callable
        Function to perform on elements, as in `fun(arr[i + step], arr[i])`.
    arr : array (...,n,...)
        array to perform operation on
    step : int, optional
        Perform operation on elemnts `step` apart, by default: 1.
    axis : int, optional
        Elements are separated by `step` along this axis, by default: -1.

    Returns
    -------
    out_arr : array (...,n-step,...)
        Output of `fun` for each pair of elements.
    """
    arr = np.moveaxis(arr, axis, -1)
    if step > 0:
        out_arr = fun(arr[..., step:], arr[..., :-step])
    elif step < 0:
        out_arr = fun(arr[..., :step], arr[..., -step:])
    else:
        out_arr = fun(arr, arr)
    return np.moveaxis(out_arr, -1, axis)


# =============================================================================
# Fixing non-parameter parts
# =============================================================================


def stochastify_c(mat: np.ndarray):  # make cts time stochastic
    """Make a matrix the generator of a continuous time Markov process.

    Changes diagonal to make row sums zero.
    **Modifies** in place, **does not** return.

    Parameters
    ----------
    mat : la.lnarray (...,n,n)
        square matrix with non-negative off-diagonal elements.
        **Modified** in place.
    """
    mat -= mat.sum(axis=-1, keepdims=True) * np.identity(mat.shape[-1])


def unstochastify_c(mat: np.ndarray):  # make cts time stochastic
    """Undo the effect of `stochastify_c` or `stochastify_pd`.

    Makes diagonal zero.
    **Modifies** in place, **does not** return.

    Parameters
    ----------
    mat : la.lnarray (...,n,n)
        square matrix with non-negative off-diagonal elements.
        **Modified** in place.
    """
    mat[(...,) + np.diag_indices(mat.shape[-1])] = 0.


def stochastify_pd(mat: np.ndarray):  # make dscr time stochastic
    """
    Make a matrix the generator of a discrete time Markov process.
    Shifts diagonals to make row sums one.

    Parameters
    ----------
    mat : la.lnarray (...,n,n)
        square matrix with non-negative off-diagonals and row sums below 1.
        **Modified** in place
    """
    mat += (1 - mat.sum(axis=-1, keepdims=True)) * np.identity(mat.shape[-1])


def stochastify_d(mat: np.ndarray):  # make dscr time stochastic
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


stochastify = stochastify_c
# =============================================================================
# Counts & types
# =============================================================================


def unpack_nest(nest: IntOrSeq) -> int:
    """Get one element of (nested) sequence"""
    while isinstance(nest, Sequence):
        nest = nest[0]
    return nest


def _get_size(arr: np.ndarray, kwds: dict, is_params: bool) -> int:
    """get size by pulling axes argument from keywords"""
    args = ('num_param', 'npar') if is_params else ('num_st', 'nst')
    siz = kwds.get(args[0], kwds.get(args[1], None))
    if siz is not None:
        return siz
    args = ('paxis', 'axis', -1) if is_params else ('maxes', 'axes', (-2, -1))
    axis = unpack_nest(kwds.get(args[0], kwds.get(*args[1:])))
    return arr.shape[axis]


def _drn_mult(drn: IntOrSeq) -> int:
    """Get factor of two if drn == 0"""
    return 1 if unpack_nest(drn) else 2


def num_param(states: Sized, *, serial: bool = False, ring: bool = False,
              uniform: bool = False, drn: IntOrSeq = 0, **kwds) -> int:
    """Number of independent rates per matrix

    Parameters
    ----------
    states : int or ndarray (n,...)
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
        states = _get_size(states, kwds, False)
    mult = _drn_mult(drn)
    if uniform:
        return mult
    if serial:
        return mult * (states - 1)
    if ring:
        return mult * states
    return mult * states * (states - 1) // 2


def num_state(params: Sized, *, serial: bool = False, ring: bool = False,
              uniform: bool = False, drn: IntOrSeq = 0, **kwds) -> int:
    """Number of states from rate vector

    Parameters
    ----------
    params : int or ndarray (n,)
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
    if uniform:
        raise ValueError("num_states is ambiguous when uniform")
    if isinstance(params, np.ndarray):
        params = _get_size(params, kwds, True)
    params *= 2 // _drn_mult(drn)
    if serial:
        return params // 2 + 1
    if ring:
        return params // 2
    return np.rint(0.5 + np.sqrt(0.25 + params)).astype(int)


def mat_type_siz(params: Sized, states: Sized, **kwds) -> _ty.Tuple[bool, ...]:
    """Is process (uniform) ring/serial/... inferred from array shapes

    If `uniform`, we cannot distinguish `general`, `seial` and `ring` without
    looking at matrix elements.

    Parameters
    ----------
    params : int or ndarray (np,)
        Number of rate parameters, or vector of rates.
    states : int or ndarray (n,...)
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
        Is the rate vector for `*_params_to_mat` or `uni_*_params_to_mat`?
        * = general, serial or ring.
    """
    if isinstance(params, np.ndarray):
        params = _get_size(params, kwds, True)
    if isinstance(states, np.ndarray):
        states = _get_size(states, kwds, False)
    uniform = params in {1, 2}
    drn = params in {1, states, states - 1, states * (states - 1) // 2}
    ring = uniform or (params in {2 * states, states})
    serial = uniform or (params in {2 * (states - 1), states - 1})
    return serial, ring, drn, uniform


# =============================================================================
# Helpers for broadcasting drn/axis
# =============================================================================


def _posify(ndim: int, axes: AxesOrSeq) -> AxesOrSeq:
    """normalise axes"""
    if isinstance(axes, int):
        return axes % ndim
    return [_posify(ndim, axs) for axs in axes]


def _negify(ndim: int, axes: AxesOrSeq) -> AxesOrSeq:
    """normalise axes"""
    if isinstance(axes, int):
        return (axes % ndim) - ndim
    return [_negify(ndim, axs) for axs in axes]


def _sort_axes(ndim: int, fun_axes: AxesOrSeq, drn_axes: IntOrSeq,
               to_mat: bool) -> _ty.Tuple[AxesOrSeq, IntOrSeq]:
    """order axes so that fun_axes is increasing"""
    fun_axes, drn_axes = _negify(ndim, fun_axes), _negify(ndim, drn_axes)
    drn_axes = _cn.tuplify(drn_axes, len(fun_axes))
    faxes, daxes = np.array(fun_axes), np.array(drn_axes)
    inds = np.argsort(faxes) if to_mat else np.argsort(faxes[:, -1])
    return faxes[inds].tolist(), daxes[inds].tolist()


def bcast_axes(fun: _ty.Callable[..., Array], arr: Array, *args,
               drn: IntOrSeq = 0, drn_axis: _ty.Sequence[int] = (0,),
               fun_axis: _ty.Sequence[Axies] = (-1,), **kwds) -> Array:
    """broadcast over axes

    Parameters
    ----------
    fun : _ty.Callable[..., ArrayType]
        Function to perform conversion over one set of axes.
    arr : ArrayType
        The array of parameters/matrices
    *args
        Additional positional arguments for `fun` after `arr`.
    drn : int|Sequence[int], optional
        If nonzero only include transitions `i -> i+sgn(drn)`, by default `0`.
    drn_axis : Sequence[int], optional
        If `drn` is a sequence, the axis to iterate over for each element,
        by default `(0,)`.
    fun_axis : int|Tuple[int,int]|Sequence[...], optional
        The axis along which each set of parameters lie, or axes to treat as
        (from, to) axes of matrix, by default `(-1,)`.

    Returns
    -------
    new_arr : ArrayType
        The result of applying `fun` to `arr` along each set of axes.
    """
    to_mat = kwds.get('to_mat', False)
    fkey = 'axis' if to_mat else 'axes'
    outarr = np.asanyarray(arr)
    fun_axis, drn_axis = _sort_axes(outarr.ndim, fun_axis, drn_axis, to_mat)
    for daxis, faxis in zip(drn_axis, fun_axis):
        kwds[fkey] = faxis
        outarr = fun(outarr, *args, drn=drn, daxis=daxis, **kwds)
    return outarr


def bcast_inds(ifun: IndFun, nst: int, drn: _ty.Sequence[int],
               ravel: bool = True) -> np.ndarray:
    """Indices for an array of transition matrices, with different directions.

    Parameters
    ----------
    ifun : IndFun
        Function that computes ravelled indices for a single transition matrix
    nst : int
        Number of states, `M`.
    drn: Sequence[int] (P,), optional
        If nonzero only include transitions in direction `i -> i+sgn(drn)`.
        Return the subscripts for a `(P,M,M)` array of matrices. By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices.
        By default `True`.

    Returns
    -------
    inds : np.ndarray (PQ,) or (P,Q)
        Ravelled indices for a (P, M, M) array of transition matrices.
    """
    inds = [ifun(nst, dirn) + k * nst**2 for k, dirn in enumerate(drn)]
    return np.concatenate(inds) if ravel else np.stack(inds)


def bcast_subs(sfun: SubFun, nst: int, drn: _ty.Sequence[int],
               ravel: bool = True) -> Subs:
    """Indices for an array of transition matrices, with different directions.

    Parameters
    ----------
    sfun : SubFun
        Function to compute unravelled indices for a single transition matrix.
    nst : int
        Number of states, `M`.
    drn: Sequence[int] (P,), optional
        If nonzero only include transitions in direction `i -> i+sgn(drn)`.
        Return the subscripts for a `(P,M,M)` array of matrices. By default 0.
    ravel : bool, optional
        Return a ravelled array, or use first axis for different matrices.
        By default `True`.

    Returns
    -------
    mats : ndarray (PQ,)
        Which transition matrix, in a `(P,M,M)` array of matrices?
    rows : ndarray (PQ,)
        Vector of row indices of off-diagonal elements.
    cols : ndarray (PQ,)
        Vector of column indices of off-diagonal elements.
    """
    subs = [sfun(nst, dirn) for dirn in drn]
    rows, cols = zip(*subs)
    mats = [np.full_like(row, k) for k, row in enumerate(rows)]
    combo = np.concatenate if ravel else np.stack
    return combo(mats), combo(rows), combo(cols)


def stack_inds(ifun: IndFun, nst) -> np.ndarray:
    """Stack ravel indices for each direction

    Parameters
    ----------
    ifun : IndFun
        Function to compute ravelled indices for a single direction.
    nst : int
        Number of states, `M`.

    Returns
    -------
    rows : ndarray (PQ,)
        Vector of row indices of off-diagonal elements.
    cols : ndarray (PQ,)
        Vector of column indices of off-diagonal elements.
    """
    return np.concatenate([ifun(nst, dirn) for dirn in (1, -1)])


def stack_subs(sfun: SubFun, nst) -> Subs:
    """Stack rows and columns for each direction

    Parameters
    ----------
    sfun : SubFun
        Function to compute unravelled indices for a single direction.
    nst : int
        Number of states, `M`.

    Returns
    -------
    rows : ndarray (PQ,)
        Vector of row indices of off-diagonal elements.
    cols : ndarray (PQ,)
        Vector of column indices of off-diagonal elements.
    """
    subs = [sfun(nst, dirn) for dirn in (1, -1)]
    rows, cols = zip(*subs)
    return np.concatenate(rows), np.concatenate(cols)


def sub_fun_bcast(fun: SubFun):
    """Decorate an unravelled-multi-index function for multiple directions

    Parameters
    ----------
    fun : Callable[(nst, drn)->(rows, cols)]
        Function to make multi-indices when `drn == +/-1`.

    Returns
    -------
    fun: Callable[(nst, drn, ravel)->([mats, ]rows, cols)]
        Function to make multi-indices when `drn == 0` or is a sequence.
    """
    @functools.wraps(fun, assigned=functools.WRAPPER_ASSIGNMENTS[:-1])
    def new_func(nst: int, drn: IntOrSeq = 0, ravel: bool = True) -> Subs:
        """Row and column indices of transitions

        Parameters
        ----------
        nst : int
            Number of states, `M`.
        drn: int|Sequence[int], optional
            If nonzero only include transitions in direction `i -> i+sgn(drn)`.
            If it is a sequence of length `P`, return the subscripts for a
            `(P,M,M)` array of matrices. By default 0.
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
        For the order of elements, see docs for `*_subs`.
        """
        if isinstance(drn, Iterable):
            return bcast_subs(new_func, nst, drn, ravel)
        if drn == 0:
            return stack_subs(fun, nst)
        return fun(nst, drn)

    return new_func


# =============================================================================
# Parameters to matrices
# =============================================================================


def _to_std(arr: Array, fax: Axies, dax: int, drnseq: bool) -> Array:
    """put axes into standard position

    drnseq : bool
        Is the drn argument a sequence?
    """
    fax, dax = _cn.tuplify(fax), _cn.tuplify(dax)
    nax = tuple(range(-len(fax) - drnseq, 0))
    oax = (dax * drnseq) + fax
    return np.moveaxis(np.asanyarray(arr), oax, nax)


def _from_std(arr: Array, fax: Axies, dax: int, drnseq: bool) -> Array:
    """put axes back from standard position

    drnseq : bool
        Is the drn argument a sequence?
    """
    to_mat = isinstance(fax, int)
    ndim = arr.ndim + (-1 if to_mat else 1)
    fax, dax = _posify(ndim, fax), _posify(ndim, dax)
    dax += to_mat and dax > fax
    oax = list(range(-1 - to_mat - drnseq, 0))
    nax = [dax] if drnseq else []
    nax += [fax, fax+1] if to_mat else [min(fax)]
    return np.moveaxis(arr, oax, nax)


def _par_axis(ndim: int, axes: Axes) -> int:
    """Which matrix axis to use for parameters"""
    return min(_posify(ndim, axes))


def params_to_mat(params: Array, fun: SubFun, drn: IntOrSeq,
                  axis: IntOrSeq, daxis: IntOrSeq, **kwds) -> Array:
    """Helper function for *_params_to_mat

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn)->subs`.
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    nst : int
        Number of states.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    daxis : int, optional
        Axis to broadcast non-scalar `drn` over, by default: 0
    Other key word parameters for `num_state`.

    Returns
    -------
    mat : array (n,n)
        Continuous time stochastic matrix.
        The extra axis in (from,to) is inserted after `axis`.
    """
    kwds.update(drn=drn, fun_axis=axis, drn_axis=daxis, to_mat=True)
    if isinstance(axis, Sequence):
        return bcast_axes(params_to_mat, params, fun, **kwds)
    stochastifier = kwds.pop('stochastifier', stochastify)
    drnseq = not isinstance(drn, int)
    params = _to_std(params, axis, daxis, drnseq)
    nst = num_state(params, **kwds)
    mat = np.zeros(params.shape[:-1] + (nst, nst)).view(type(params))
    mat[(...,) + fun(nst, drn, False)] = params
    stochastifier(mat)
    return _from_std(mat, axis, daxis, drnseq)


def uni_to_any(params: np.ndarray, nst: int, axis: IntOrSeq, **kwds
               ) -> np.ndarray:
    """Helper for uni_*_params_to_mat

    Parameters
    ----------
    params : ndarray (1,) or (2,)
        Vector of independent elements, in order that depends on flags below.
        If `drn == 0`, you must provide 2 parameters, one for each direction.
        See docs for `*_inds` for details.
    nst : int
        Number of states.
    axis : int, optional
        Axis along which each set of parameters lie, by default -1.
    Other key word parameters for `num_param`.

    Returns
    -------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or (2,)
        Vector of independent elements, in order that depends on flags above.
        See docs for `*_inds` for details.
    """
    if isinstance(axis, Sequence):
        kwds.update(fun_axis=axis, to_mat=True)
        return bcast_axes(uni_to_any, params, nst, **kwds)
    params = np.asanyarray(params)
    axis = _posify(params.ndim, axis)
    params = np.moveaxis(params, axis, -1)
    # if drn == 0, then params.shape[axis] == 2, so each needs expanding by
    # half of the real num_param
    kwds.update(drn=1, uniform=False)
    npar = num_param(nst, **kwds)
    full = np.broadcast_to(params[..., None], params.shape + (npar,))
    shape = full.shape[:-2] + (-1,)
    return np.moveaxis(full.reshape(shape), -1, axis)


# =============================================================================
# Matrices to parameters
# =============================================================================


def mat_to_params(mat: Array, fun: IndFun, drn: IntOrSeq, axes: AxesOrSeq,
                  daxis: IntOrSeq, **kwds) -> Array:
    """Helper function for *_mat_to_params

    Parameters
    ----------
    fun : callable
        Function that takes `(nst,drn,ravel)->inds`.
    mat : ndarray (...,n,n)
        Continuous time stochastic matrix.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    axes : Tuple[int, int]
        Axes to treat as (from, to) axes.
    daxis : int
        Axis to broadcast non-scalar `drn` over.
    """
    kwds.update(drn=drn, fun_axis=axes, drn_axis=daxis)
    if isinstance(axes[0], Sequence):
        return bcast_axes(mat_to_params, mat, fun, **kwds)
    drnseq = not isinstance(drn, int)
    mat = _to_std(mat, axes, daxis, drnseq)
    params = mat[(...,) + fun(mat.shape[-1], drn, False)]
    return _from_std(params, axes, daxis, drnseq)


def to_uni(params: Array, drn: IntOrSeq, grad: bool, axes: AxesOrSeq,
           **kwds) -> Array:
    """Helper for uni_*_mat_to_params

    Parameters
    ----------
    params : ndarray (n(n-1),) or (2(n-1),) or (2n,) or half of <-
        Vector of independent elements, in order that depends on flags below.
        See docs for `*_inds` for details.
    drn : int
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`.
    grad : bool
        Is the output for a gradient (True) or a transition matrix (False).
        If True, return sum of values in each direction, else, return mean.
    axes : Tuple[int, int] or None
        Original axes to treat as (from, to) axes.
    """
    if isinstance(axes[0], Sequence):
        kwds.update(drn=drn, fun_axis=axes, grad=grad)
        return bcast_axes(to_uni, params, **kwds)
    # Ensure the same oaxis here as in mat_to_params
    oaxis = _par_axis(params.ndim + 1, axes)
    npar = params.shape[oaxis] // _drn_mult(drn)
    new_shape = params.shape[:oaxis] + (-1, npar) + params.shape[oaxis+1:]
    params = params.reshape(new_shape).sum(axis=oaxis+1)
    if not grad:
        params /= npar
    return params


# =============================================================================
# Type hints
# =============================================================================
Array = _ty.TypeVar('Array', bound=np.ndarray)
Sized = _ty.Union[int, np.ndarray]
Axes = _ty.Tuple[int, int]
Subs = _ty.Tuple[np.ndarray, ...]
SubFun = _ty.Callable[[int, int], _ty.Tuple[np.ndarray, np.ndarray]]
IndFun = _ty.Callable[[int, int], np.ndarray]
SubsFun = _ty.Callable[[int, int, bool], Subs]
IndsFun = _ty.Callable[[int, int, bool], np.ndarray]
Axies = _ty.Union[int, Axes]
AxType = _ty.TypeVar('AxType', int, Axes, Axies)
OrSeqOf = _ty.Union[AxType, _ty.Sequence[AxType]]
IntOrSeq = OrSeqOf[int]
AxesOrSeq = OrSeqOf[Axes]
