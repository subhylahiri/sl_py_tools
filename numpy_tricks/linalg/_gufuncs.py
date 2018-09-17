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
solve
    Solve systems of linear equations.
lstsq_m
    Least square solution of under-determined systems of linear equations.
lstsq_n
    Least square solution of over-determined systems of linear equations.
qr
    QR factorization.
qr_tall
    QR factorization for tall-skinny matrices.

They raise ValueErrors instead of LinAlgError

Wrappers for gufuncs:
rsolve
    Solve reversed systems of linear equations.
lstsq
    Least square solution of systems of linear equations.
rlstsq
    Least square solution of reversed systems of linear equations.

`numpy.linalg` broadcasting rules apply. 1D arrays are not dealt with here (but
see `lnarray` class).
"""

import numpy as np
from ._gufuncs_nb import qr, qr_tall

try:
    from numpy.core._umath_tests import matrix_multiply as matmul
except ImportError:
    from ._gufuncs_nb import matmul

try:
    from numpy.linalg._umath_linalg import solve
except ImportError:
    from ._gufuncs_nb import solve

try:
    from numpy.linalg._umath_linalg import lstsq_m, lstsq_n
except ImportError:
    from ._gufuncs_nb import lstsq_m, lstsq_n


def ufunc_quack(*args, **kwds):
    """Fake ufunc method"""
    return NotImplemented


def ufunc_duck(parent_ufunc: np.ufunc):
    """Decorator to give function fake ufunc properties
    """
    def ufunc_egg(func):
        """Give function fake ufunc properties
        """
        func.nin = parent_ufunc.nin
        func.nout = parent_ufunc.nout
        func.nargs = parent_ufunc.nargs
        func.ntypes = parent_ufunc.ntypes
        func.types = parent_ufunc.types
        func.identity = parent_ufunc.identity
        func.signature = parent_ufunc.signature
        func.reduce = ufunc_quack
        func.accumulate = ufunc_quack
        func.reduceat = ufunc_quack
        func.outer = ufunc_quack
        func.at = ufunc_quack
        return func
    return ufunc_egg


@ufunc_duck(solve)
def rsolve(a: np.ndarray, b: np.ndarray, out=None, **kwds) -> np.ndarray:
    """Solve reversed systems of linear equations.

    Solves for `c` in :math:`a = c.b`.
    It is a `numba` generated `gufunc` reverse version of `numpy.linalg.solve`,
    so it broadcasts according to standard `numpy.linalg` rules and deals with
    subclasses and other classes properly.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., N, N)
        Matrix, or stack of matrices to matrix multiply from the right.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., M, N)
        Matrix product of `a` and inverse of `b`.

    Raises
    ------
    LinAlgError
        If `b` is singular or not square.

    See Also
    --------
    numpy.linalg.solve
    """
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, np.ndarray):
        newout = (out.swapaxes(-2, -1)[...],)
    else:
        newout = (out,)
    return solve(b.swapaxes(-2, -1), a.swapaxes(-2, -1),
                 out=newout, **kwds).swapaxes(-2, -1)


@ufunc_duck(lstsq_m)
def lstsq(a: np.ndarray, b: np.ndarray, rcond=None,
          *optargs, **kwds) -> np.ndarray:
    """Least square solution of systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `gufunc` based version of `numpy.linalg.lstsq`, so it broadcasts
    according to standard `numpy.linalg` rules and deals with subclasses and
    other classes properly.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., M, P)
        Matrix, or stack of matrices on the right-hand-side.
    rcond : float or np.ndarray(...)
        minimum recprocal condition number for non-zero singular values.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., N, P)
        Matrix product of pseudoinverse of `a` and `b`.
    resid : np.ndarray (...) or (..., P)
        Residuals
    rank : np.ndarray (..., dtype=int)
        Rank of `a`
    s : np.ndarray (..., M) or (..., N)
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    numpy.linalg.lstsq
    """
    m, n = a.shape[-2:]
    if rcond is None:
        rcond = np.finfo(np.result_type(a, b)).eps * max(n, m)
    ufunc = lstsq_n
    if m <= n:
        ufunc = lstsq_m
    return ufunc(a, b, rcond, *optargs, **kwds)


@ufunc_duck(lstsq_m)
def rlstsq(a: np.ndarray, b: np.ndarray, rcond=None,
           out=(None, None, None, None), **kwds) -> np.ndarray:
    """Least square solution of reversed systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `gufunc` based reverse version of `numpy.linalg.lstsq`, so it
    broadcasts according to standard `numpy.linalg` rules and deals with
    subclasses and other classes properly.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., P, N)
        Matrix, or stack of matrices on the right-hand-side.
    rcond : float or np.ndarray(...)
        minimum recprocal condition number for non-zero singular values.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., M, P)
        Matrix product of `a` and pseudoinverse of `b`.
    resid : np.ndarray (...) or (..., P)
        Residuals
    rank : np.ndarray (..., dtype=int)
        Rank of `b`
    s : np.ndarray (..., P) or (..., N)
        Singular values of `b`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    numpy.linalg.lstsq
    """

    if isinstance(out[0], np.ndarray):
        newout = (out[0].swapaxes(-2, -1)[...],)
    else:
        newout = (out[0],)
    newout += out[1:]
    results = lstsq(b.swapaxes(-2, -1), a.swapaxes(-2, -1), rcond,
                    out=newout, **kwds)
    return (results[0].swapaxes(-2, -1),) + results[1:]


@ufunc_duck(lstsq_m)
def lstsq_bin(a: np.ndarray, b: np.ndarray, out=(None,), **kwds) -> np.ndarray:
    """Least square solution of systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `gufunc` based version of `numpy.linalg.lstsq`, so it broadcasts
    according to standard `numpy.linalg` rules and deals with subclasses and
    other classes properly.
    It is meant for use in binary operators, so it uses the default `rcond` and
    discards all inputs except the first.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., M, P)
        Matrix, or stack of matrices on the right-hand-side.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., N, P)
        Matrix product of pseudoinverse of `a` and `b`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    numpy.linalg.lstsq
    """
    newout = out + (None,) * 3
    return lstsq(a, b, rcond=None, out=newout, **kwds)[0]


lstsq_bin.nin = 2
lstsq_bin.nout = 1
lstsq_bin.nargs = 3
lstsq_bin.signature = '(m,n),(m,nrhs)->(n,nrhs)'
lstsq_bin.types = ['ff->f', 'dd->d', 'FF->F', 'DD->D']


@ufunc_duck(lstsq_bin)
def rlstsq_bin(a: np.ndarray, b: np.ndarray, out=(None,), **kws) -> np.ndarray:
    """Least square solution of reversed systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `gufunc` based reverse version of `numpy.linalg.lstsq`, so it
    broadcasts according to standard `numpy.linalg` rules and deals with
    subclasses and other classes properly.
    It is meant for use in binary operators, so it uses the default `rcond` and
    discards all inputs except the first.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., P, N)
        Matrix, or stack of matrices on the right-hand-side.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., M, P)
        Matrix product of `a` and pseudoinverse of `b`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    numpy.linalg.lstsq
    """
    newout = out + (None,) * 3
    return rlstsq(a, b, rcond=None, out=newout, **kws)[0]


@ufunc_duck(matmul)
def rmatmul(a: np.ndarray, b: np.ndarray, *args, **kwds) -> np.ndarray:
    """Reversed matrix multiplication.

    Helper function for `invarray` binary operators.
    """
    return matmul(b, a, *args, **kwds)


@ufunc_duck(np.true_divide)
def rtrue_divide(a: np.ndarray, b: np.ndarray, *args, **kwds) -> np.ndarray:
    """Reversed true division.

    Helper function for `(p)invarray` binary operators.
    """
    return np.true_divide(b, a, *args, **kwds)
