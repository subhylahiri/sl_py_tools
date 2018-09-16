# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:35:26 2018

@author: Subhy
"""

import numpy as np
import numpy.linalg as la
from numba import guvectorize, int32, int64, float32, float64

mat_sig = [(int32[:, :], int32[:, :], int32[:, :]),
           (int64[:, :], int64[:, :], int64[:, :]),
           (float32[:, :], float32[:, :], float32[:, :]),
           (float64[:, :], float64[:, :], float64[:, :])]

lsq_sig = [(float32[:, :], float32[:, :], float32[:],
            float32[:, :], float32[:], int32[:], float32[:]),
           (float64[:, :], float64[:, :], float64[:],
            float64[:, :], float64[:], int64[:], float64[:])]

# =============================================================================
# stuff
# =============================================================================


@guvectorize(mat_sig, '(m,n),(n,p)->(m,p)', cache=True)
def matmul(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Matrix multiplication.

    Calculates :math:`c = a.b`.
    It is a `numba` generated `gufunc` version of `numpy.matmul`.
    It is ~30% slower, but deals with subclasses and other classes properly.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., N, P)
        Matrix, or stack of matrices to matrix multiply from the right.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., M, P)
        Matrix product of `a` and `b`.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as the second-to-last
        dimension of `b`.

    See Also
    --------
    numpy.matmul
    """
    c[:, :] = np.matmul(a, b)


@guvectorize(mat_sig, '(m,m),(m,n)->(m,n)', cache=True)
def solve(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Solve systems of linear equations.

    Solves for `c` in :math:`a.c = b`.
    It is a `numba` generated `gufunc` version of `nnumpy.linalg.solve`.
    It is ~30% slower, but deals with subclasses and other classes properly.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, M)
        Matrix, or stack of matrices to matrix multiply from the left.
    b : np.ndarray (..., M, N)
        Matrix, or stack of matrices to matrix multiply from the right.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    c : np.ndarray (..., M, N)
        Matrix product of inverse of `a` and `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

    See Also
    --------
    numpy.linalg.solve
    """
    c[:, :] = la.solve(a, b)


@guvectorize(mat_sig, '(m,n),(n,n)->(m,n)', cache=True)
def rsolve(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Solve systems of linear equations.

    Solves for `c` in :math:`a = c.b`.
    It is a `numba` generated `gufunc` reverse version of `numpy.linalg.solve`.
    It is ~30% slower, but deals with subclasses and other classes properly.
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
    c[:, :] = la.solve(b.T, a.T).T


@guvectorize(lsq_sig, '(m,n),(m,p),()->(n,p),(),(),(m)', cache=True)
def lstsq_m(a: np.ndarray, b: np.ndarray, rcond: np.ndarray,
            c: np.ndarray, resid: np.ndarray, rank: np.ndarray, s: np.ndarray):
    """Least square solution of under-determined systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `numba` generated `gufunc` version of `numpy.linalg.lstsq` with
    `rcond` non-optional and requiring :math:`M \leq N`.
    It is ~30% slower, but it broadcasts according to standard `numpy.linalg`
    rules and deals with subclasses and other classes properly.
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
    resid : np.ndarray (...)
        Residuals (always zero)
    rank : np.ndarray (..., dtype=int)
        Rank of `a`
    s : np.ndarray (..., M)
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    numpy.linalg.lstsq
    """
    cc, _, rnk, sv = la.lstsq(a, b, rcond[0])[:2]
    c[:, :] = cc
    resid[:] = 0
    rank[0] = rnk
    s[:] = sv


@guvectorize(lsq_sig, '(m,n),(m,p),()->(n,p),(p),(),(m)', cache=True)
def lstsq_n(a: np.ndarray, b: np.ndarray, rcond: np.ndarray,
            c: np.ndarray, resid: np.ndarray, rank: np.ndarray, s: np.ndarray):
    """Least square solution of over-determined systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `numba` generated `gufunc` version of `numpy.linalg.lstsq` with
    `rcond` non-optional and requiring :math:`M > N`.
    It is ~30% slower, but it broadcasts according to standard `numpy.linalg`
    rules and deals with subclasses and other classes properly.
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
    resid : np.ndarray (..., P)
        Residuals
    rank : np.ndarray (..., dtype=int)
        Rank of `a`
    s : np.ndarray (..., N)
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    numpy.linalg.lstsq
    """
    cc, rr, rnk, sv = la.lstsq(a, b, rcond[0])[:2]
    c[:, :] = cc
    resid[:] = rr
    rank[0] = rnk
    s[:] = sv


@guvectorize(mat_sig, '(m,n),(m,p)->(n,p)', cache=True)
def lstsq(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Least square solution of systems of linear equations.

    Least square solution for `c` in :math:`a.c = b`.
    It is a `numba` generated `gufunc` version of `numpy.linalg.lstsq` with
    `rcond = None` and other outputs discarded.
    It is ~30% slower, but it broadcasts according to standard `numpy.linalg`
    rules and deals with subclasses and other classes properly.
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
    c[:, :] = la.lstsq(a, b, None)[0]


@guvectorize(mat_sig, '(m,n),(p,n)->(m,p)', cache=True)
def rlstsq(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Least square solution of systems of linear equations.

    Least square solution for `c` in :math:`a = c.b`.
    It is a `numba` generated `gufunc` version of reverse `numpy.linalg.lstsq`,
    with `rcond = None` and other outputs discarded.
    It is ~30% slower, but it broadcasts according to standard `numpy.linalg`
    rules and deals with subclasses and other classes properly.
    It does not work with 1D arrays (but see `lnarray` class).

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices on the left-hand-side.
    b : np.ndarray (..., P, N)
        Matrix, or stack of matrices to matrix multiply from the right.
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
#    m = min(a.shape[-2:])
    c[:, :] = la.lstsq(b.T, a.T, None)[0].T


@guvectorize(mat_sig, '(m,n)->(m,m),(m,n)', cache=True)
def qr(a: np.ndarray, q: np.ndarray, r: np.ndarray):
    """QR factorization.

    Factor the matrix `a` as `q.r`, where `q` is orthonormal and `r` is
    upper-triangular.
    It is a `gufunc` version of `numpy.linalg.qr` in 'complete' mode.
    It is ~30% slower, but it broadcasts according to standard `numpy.linalg`
    rules and deals with subclasses and other classes properly.

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices, to be factored.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    q : np.ndarray (..., M, M)
        Orthogonal matrix, or stack of orthogonal matrices.
    c : np.ndarray (..., M, N)
        Upper triangular matrix.

    Raises
    ------
    LinAlgError
        If factoring fails.

    See Also
    --------
    numpy.linalg.qr
    """
    qq, rr = la.qr(a, mode='complete')
    q[:, :] = qq
    r[:, :] = rr


@guvectorize(mat_sig, '(m,n)->(m,n),(n,n)', cache=True)
def qr_tall(a: np.ndarray, q: np.ndarray, r: np.ndarray):
    """QR factorization for tall-skinny matrices.

    Factor the matrix `a` as `qr`, where `q` is orthonormal and `r` is
    upper-triangular.
    It is a `gufunc` version of `numpy.linalg.qr` in 'reduced'. mode,
    restricted to tall matrices, with :math:`M > N`.
    It is ~30% slower, but it broadcasts according to standard `numpy.linalg`
    rules and deals with subclasses and other classes properly.

    Parameters
    ----------
    a : np.ndarray (..., M, N)
        Matrix, or stack of matrices, with :math:`M > N`, to be factored.
    others
        The standard `numpy.gufunc` optional parameters

    Returns
    -------
    q : np.ndarray (..., M, N)
        Matrix, or stack of tall matrices, with orthonormal columns.
    c : np.ndarray (..., N, N)
        Upper triangular square matrix.

    Raises
    ------
    LinAlgError
        If factoring fails.

    See Also
    --------
    numpy.linalg.qr
    """
    qq, rr = la.qr(a, mode='reduced')
    q[:, :] = qq
    r[:, :] = rr
