# -*- coding: utf-8 -*-
# =============================================================================
# Created on Sat Sep  1 20:35:47 2018
#
# @author: subhy
# =============================================================================
"""
Generalized ufunc versions of `numpy.linalg` routines:

matmul
    Matrix multiplication.
rmatmul
    Reversed matrix multiplication (only useful to implement binary operators).
solve
    Solve systems of linear equations.
rsolve
    Solve reversed systems of linear equations.
solve_lu, rsolve_lu
    Also return LU factors for use next time.
lu_solve, rlu_solve
    Use LU factors from previous time.
lstsq
    Least square solution of systems of linear equations.
rlstsq
    Least square solution of reversed systems of linear equations.
lstsq_qrm, lstsq_qrn, rlstsq_qrm, rlstsq_qrn
    Also return QR factors for use next time.
qr_lstsq, rqr_lstsq
    Use QR factors from previous time.
norm
    Vector 2-norm.
lu_*
    LU factorization in various forms - {m,n,rawm,rawn}.
        m: #(rows) < #(columns).
        n: #(rows) > #(columns).
        raw: return `AF`, which contains `L` and `U`, and `ipiv`.
pivot, rpivot
    Perform row pivots with the output of lu_*.
qr_*
    QR factorization in various forms - {m,n,rm,rn,rawm,rawn}.
        m: #(rows) < #(columns).
        n: #(rows) > #(columns).
        r: only return `R`.
        raw: return `H` and `tau`, from which `Q` and `R` can be computed.
rtrue_tivide
    Reversed division (only useful to implement binary operators).

They raise ValueError instead of LinAlgError

`numpy.linalg` broadcasting rules apply. 1D arrays are not dealt with here (but
see `lnarray` class).
"""
import functools as _ft
import itertools as _it
import numpy as _np
import numpy.lib.mixins as _mix
from ._gufuncs_cloop import norm, rtrue_divide  # matmul, rmatmul
from ._gufuncs_blas import matmul, rmatmul  # norm
# from ._gufuncs_lapack import *
from ._gufuncs_lu_solve import (lu_m, lu_n, lu_rawm, lu_rawn, solve, rsolve,
                                solve_lu, rsolve_lu, lu_solve, rlu_solve,
                                pivot, rpivot)
from ._gufuncs_qr_lstsq import (qr_m, qr_n, qr_rm, qr_rn, qr_rawm, qr_rawn,
                                lstsq, rlstsq, lstsq_qrm, lstsq_qrn,
                                rlstsq_qrm, rlstsq_qrn, qr_lstsq, rqr_lstsq)
# fool pyflakes
assert norm
assert pivot
assert rpivot
assert any((lu_m, lu_n, lu_rawm, lu_rawn))
assert any((qr_m, qr_n, qr_rm, qr_rn, qr_rawm, qr_rawn))
# =============================================================================
# Mixin for linear algebra operators
# =============================================================================


class MatmulOperatorsMixin():
    """Mixin for defining __matmul__ special methods via gufuncs
    """
    __matmul__, __rmatmul__, __imatmul__ = _mix._numeric_methods(matmul,
                                                                 'matmul')


class LNArrayOperatorsMixin(_mix.NDArrayOperatorsMixin, MatmulOperatorsMixin):
    """Mixin for defining operator special methods via __array_ufunc__
    """
    pass


# =============================================================================
# Categories of binary operators
# =============================================================================

# maps Tuple[bool] (left, right) -> gufunc
# if left, 1st argument of gufunc is 'inverted'
# if right, 2nd argument of gufunc is 'inverted'
solve_family = ((matmul, rsolve), (solve, rmatmul))
solve_lu_family = ((matmul, rsolve_lu), (solve_lu, rmatmul))
lu_solve_family = ((matmul, rlu_solve), (lu_solve, rmatmul))
lstsq_family = ((matmul, rlstsq), (lstsq, None))
lstsq_qrm_family = ((matmul, rlstsq_qrm), (lstsq_qrm, None))
lstsq_qrn_family = ((matmul, rlstsq_qrn), (lstsq_qrn, None))
qr_lstsq_family = ((matmul, rqr_lstsq), (qr_lstsq, None))

_families = [
        solve_family,
        solve_lu_family,
        lu_solve_family,
        lstsq_family,
        lstsq_qrm_family,
        lstsq_qrn_family,
        qr_lstsq_family,
]

# maps gufunc -> (left, right) Tuple[bool]
# if left, 1st argument of gufunc is 'inverted'
# if right, 2nd argument of gufunc is 'inverted'
inverse_arguments = {}
_bools = (False, True)
for _family, _left_arg, _right_arg in _it.product(_families, _bools, _bools):
    _func = _family[_left_arg][_right_arg]
    if _func is not None and _func not in inverse_arguments.keys():
        inverse_arguments[_func] = (_left_arg, _right_arg)


# backwards maps Tuple[bool] (left, right) -> ufunc
# if left, *2nd* argument of ufunc is a *numerator*
# if right, *1st* argument of ufunc is a *numerator*
truediv_family = ((None, _np.true_divide), (rtrue_divide, _np.multiply))
inverse_scalar_arguments = {}
# backwards maps ufunc -> (left, right) Tuple[bool]
for _left_arg, _right_arg in _it.product(_bools, _bools):
    _func = truediv_family[_left_arg][_right_arg]
    if _func is not None and _func not in inverse_scalar_arguments.keys():
        inverse_scalar_arguments[_func] = (_left_arg, _right_arg)

# =============================================================================
# Shape for linear algebra
# =============================================================================


def return_shape_mat(x, y):
    """Shape of result of broadcasted matrix multiplication
    """
    if x.ndim == 0 or y.ndim == 0:
        raise ValueError('Scalar operations not supported. Use mul.')
    if y.ndim == 1:
        return x.shape[:-1]
    if x.ndim == 1:
        return y.shape[:-2] + y.shape[-1:]
    if x.shape[-1] != y.shape[-2]:
        msg = 'Inner matrix dimensions mismatch: {0} and {1}.'
        raise ValueError(msg.format(x.shape, y.shape))
    return _np.broadcast(x[..., :1], y[..., :1, :]).shape


def vec2mat(x, y, case=(False, False)):
    """Convert vectors to single column/row matrices for linear algebra gufuncs

    Only does anything when `x.ndim==1` or `y.ndim==1`.

    Parameters
    ----------
    x,y : ndarray
        Left/right-hand-side of a linear algebra binary operation.
    case : Tuple[bool or None, bool or None]
        convert to row or column? `case = (x_ax, y_ax)`.
            ax = False: x/y -> row/column
            ax = True: x/y -> column/row
            ax = None: do not change
        for `case=(True, True)` we also reverse `needs_squeeze`

    Returns
    -------
    x,y : ndarray
        The inputs, but with singleton axes added so that linear algebra
        gufuncs behave correctly.
    needs_squeeze : list(bool)
        Tells us if axes [-2, -1] need to be removed from the gufunc output.
    """
    needs_squeeze = [False, False]
    if x.ndim == 1 and case[0] is not None:
        x = _np.expand_dims(x, case[0] + 0)
        needs_squeeze[0] = True
    if y.ndim == 1 and case[1] is not None:
        y = _np.expand_dims(y, 1 - case[1])
        needs_squeeze[1] = True
    if all(case):
        needs_squeeze = [x for x in reversed(needs_squeeze)]
    return x, y, needs_squeeze


def mat2vec(z, squeeze):
    """Convert column/row-matrices back to vectors from linear algebra gufuncs

    Parameters
    ----------
    z : ndarray
        Output of a gufunc performing a linear algebra binary operation.
    needs_squeeze : list(bool)
        Tells us if axes [-2, -1] need to be removed from the gufunc output.

    Returns
    -------
    z : ndarray
        The input, stripped of any sigleton axes added by `vec2mat`.
    """
    axs = (-2,) * squeeze[0] + (-1,) * squeeze[1]
    z = z.squeeze(axis=axs)
    return z[()] if z.ndim == 0 else z


# these help adjust docstrings in wrapped functions
_vec_doc = """
Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.
"""
_bin_doc = "It is intended for use in binary operators.\n"


def vec_wrap(gufunc, case=()):
    """Wrap a gufunc with special handling for vectors

    Parameters
    ----------
    case : int
        convert to row or column? `case = (x_ax, y_ax)`.
        ax = 0: x/y -> row/column
        ax = 1: x/y -> column/row
        ax = 2: do not change
    """
    if not case:
        case = inverse_arguments[gufunc]

    @_ft.wraps(gufunc)
    def wrapper(x, y, *args, **kwargs):
        x, y, squeeze = vec2mat(_np.asanyarray(x), _np.asanyarray(y), case)
        with _np.errstate(invalid='raise'):
            try:
                z = gufunc(x, y, *args, **kwargs)
            except FloatingPointError:
                raise _np.linalg.LinAlgError("Failure in linalg routine "
                                             + gufunc.__name__)
        return mat2vec(z, squeeze)

    wrapper.__doc__ = wrapper.__doc__.replace("\nParameters",
                                              "\n" + _vec_doc + "\nParameters")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,M,N)",
                                              "(...,M,N) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,N,P)",
                                              "(...,N,P) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace(
                            "(...,M,P)", "(...,M,P), (...,M), (...,P) or ()")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,M,NRHS)",
                                              "(...,M,NRHS) or (M,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,NRHS,M)",
                                              "(...,NRHS,M) or (M,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,N,NRHS)",
                                              "(...,N,NRHS) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,NRHS,N)",
                                              "(...,NRHS,N) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace(_bin_doc, "")
    return wrapper
