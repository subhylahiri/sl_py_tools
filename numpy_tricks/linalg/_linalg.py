# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:10:02 2017

@author: Subhy

Linear algebra routines.

Functions
---------
transpose
    Transpose last two indices.
dagger
    Transpose-conjugate last two indices.
col
    Treat multi-dim array as a stack of column vectors.
row
    Treat multi-dim array as a stack of row vectors.
scal
    Treat multi-dim array as a stack of scalars.
matmul
    Matrix multiplication.
matldiv
    Matrix division from left.
matrdiv
    Matrix division from right.
qr
    QR decomposition with broadcasting and subclass passing.
lu
    LU decomposition with broadcasting and subclass passing.
"""
import typing as ty
import numpy as np
import numpy.linalg.linalg as nla
from . import gufuncs as gf

__all__ = [
    'transpose',
    'dagger',
    'col',
    'row',
    'scal',
    'matmul',
    'rmatmul',
    'solve',
    'rsolve',
    'lstsq',
    'rlstsq',
    'matldiv',
    'matrdiv',
    'qr',
    'lu',
]

# =============================================================================
# Reshaping for linear algebra
# =============================================================================


def transpose(a: np.ndarray) -> np.ndarray:
    """Transpose last two indices.

    Transposing last two indices fits better with `np.linalg`'s broadcasting,
    which treats multi-dim arrays as stacks of matrices.

    Parameters
    ----------
    a : np.ndarray, (..., M, N)

    Returns
    -------
    transposed : np.ndarray, (..., N, M)
    """
    if a.ndim < 2:
        return a
    return nla.transpose(a)


def dagger(a: np.ndarray) -> np.ndarray:
    """Hermitian conjugate over last two indices.

    Transposing last two indices fits better with `np.linalg`'s broadcasting,
    which treats multi-dim arrays as stacks of matrices.

    Parameters
    ----------
    a : np.ndarray, (..., M, N)

    Returns
    -------
    conjugated : np.ndarray, (..., N, M)
    """
    return transpose(a.conj())


def col(a: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of column vectors.

    Achieves this by inserting a singleton dimension in last slot.
    You'll have an extra singleton after any linear algebra operation from the
    left.

    Parameters
    ----------
    a : np.ndarray, (..., N)

    Returns
    -------
    expanded : np.ndarray, (..., N, 1)
    """
    return a[..., None]


def row(a: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of row vectors.

    Achieves this by inserting a singleton dimension in second-to-last slot.
    You'll have an extra singleton after any linear algebra operation from the
    right.

    Parameters
    ----------
    a : np.ndarray, (..., N)

    Returns
    -------
    expanded : np.ndarray, (..., 1, N)
    """
    return a[..., None, :]


def scal(a: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of scalars.

    Achieves this by inserting singleton dimensions in last two slots.

    Parameters
    ----------
    a : np.ndarray, (...,)

    Returns
    -------
    expanded : np.ndarray, (..., 1, 1)
    """
    return a[..., None, None]


# =============================================================================
# Division & Multiplication
# =============================================================================
matmul = gf.vec_wrap(gf.matmul)
rmatmul = gf.vec_wrap(gf.rmatmul)
solve = gf.vec_wrap(gf.solve)
rsolve = gf.vec_wrap(gf.rsolve)
lstsq = gf.vec_wrap(gf.lstsq)
rlstsq = gf.vec_wrap(gf.rlstsq)


def matldiv(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from left.

    Computes :math:`z = x \\ y = x^{-1} y` for invertible `x`, or :math:`x^+ y`
    for non-invertible `x`.
    Pseudo-inverse version uses `lstsq`.
    Full inverse version uses `solve`.
    Both versions broadcast using gufunc machinery.

    Parameters
    ----------
    x : (..., M, N) array_like
        Divisor or Denominator.
    y : {(M,), (..., M, K)} array_like
        Dividend or Numerator.
    out : {(N,), (..., N, K)} np.ndarray
        array to store the output in.

    Returns
    -------
    z : {(N,), (..., N, K)} np.ndarray
        Quotient. It has the same type as `x` or `y`.

    Raises
    ------
    LinAlgError
        If `x` is not invertible and `lstsq` doesn't converge.

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    if x.ndim > 1 and x.shape[-1] == x.shape[-2]:
        try:
            return solve(x, y, *args, **kwargs)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return lstsq(x, y, *args, **kwargs)


def matrdiv(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from right.

    Computes :math:`z = x / y = x y^{-1}` for invertible `y`, or :math:`x y^+`
    for non-invertible `y`.
    Pseudo-inverse version uses `rlstsq`.
    Full inverse version uses `rsolve`.
    Both versions broadcast using gufunc machinery.

    Parameters
    ----------
    x : {(M,), (..., K, M)} array_like
        Dividend or Numerator.
    y : (..., N, M) array_like
        Divisor or Denominator.
    out : {(N,), (..., K, N)} np.ndarray
        array to store the output in.

    Returns
    -------
    z : {(N,), (..., K, N)} np.ndarray
        Quotient. It has the same type as `x` or `y`.

    Raises
    ------
    LinAlgError
        If `x` is not invertible and `lstsq` doesn't converge.

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    if y.ndim > 1 and y.shape[-1] == y.shape[-2]:
        try:
            return rsolve(x, y, *args, **kwargs)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return rlstsq(x, y, *args, **kwargs)


# =============================================================================
# Matrix decomposition
# =============================================================================


qr_modes = {'reduced': (gf.qr_m, gf.qr_n),
            'complete': (gf.qr_m, gf.qr_m),
            'r': (gf.qr_rm, gf.qr_rn),
            'raw': (gf.qr_rawm, gf.qr_rawn)}


def qr(x: np.ndarray, mode: str = 'reduced', *args,
       **kwds) -> ty.Tuple[np.ndarray, ...]:
    """QR decomposition.

    Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
    `K = min(M,N)`, except for mode `complete`, where `K = M`.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str
        chosen from:
        **reduced** - default, use minimum inner dimensionality ``K=min(M,N)``,
        **complete** - use inner dimensionality ``K=M``, for square `Q`,
        **r** - return `R` only,
        **raw** - return `H,tau`, which determine `Q` and `R` (see below).

    Returns
    -------
    Q: ndarray (...,M,K). Modes: `reduced, complete`.
        Matrix with orthonormal columns.
    R: ndarray (...,K,N). Modes: `reduced, complete, r`.
        Matrix with zeros below the diagonal.
    H: ndarray (...,N,M). Modes: `raw`.
        Transpose of matrix for use in Fortran. Before transpose, it contained
        the following information:
            On & super-diagonal: non-zero part of `R`.
            Sub-diagonal: lower part of Householder reflectors `v`, in columns.
    tau: ndarray (...,K,). Modes: `raw`.
        Scaling factors for Householder reflectors. The unit normal to the
        reflection plane is ``V = sqrt(tau/2) [0 ... 0 1 v^T]^T``.
    """
    if mode not in qr_modes.keys():
        raise ValueError('Modes known to qr: reduced, complete, r, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = qr_modes[mode][x.shape[-2] > x.shape[-1]]
    gf.make_errobj("QR failed: rank deficient?", kwds)
    return ufunc(x, *args, **kwds)


lu_modes = {'separate': (gf.lu_m, gf.lu_n),
            'raw': (gf.lu_rawm, gf.lu_rawn)}


def lu(x: np.ndarray, mode: str = 'separate', *args,
       **kwds) -> ty.Tuple[np.ndarray, ...]:
    """LU decomposition.

    Factor a matrix as `A = PLU` with `P` a permutation matrix,
    `L` lower- and `U` upper-triangular.
    `K = min(M,N)`. The diagonal elements of `L` are 1.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str
        chosen from:
        **separate** - default, return `L` and `U` separately,
        **raw** - return `AF`, which contains `L` and `U` (see below).

    Returns
    -------
    L: ndarray (...,M,K). Modes: `separate`.
        Matrix with zeros above the diagonal and ones on the diagonal.
    U: ndarray (...,K,N). Modes: `separate`.
        Matrix with zeros below the diagonal.
    AF: ndarray (...,N,M). Modes: `raw`.
        Raw matrix output from Lapack in Fortran.
            On & super-diagonal: non-zero part of `U`.
            Sub-diagonal: non-zero part of `L`, excluding diagonal.
    ipiv: ndarray (...,K,). Modes: `separate, raw`.
        Pivot indices
    """
    if mode not in lu_modes.keys():
        raise ValueError('Modes known to lu: separate, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = lu_modes[mode][x.shape[-2] > x.shape[-1]]
    gf.make_errobj("LU failed: rank deficient?", kwds)
    return ufunc(x, *args, **kwds)
