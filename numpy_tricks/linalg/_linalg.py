# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:10:02 2017

@author: Subhy

Linear algebra routines.

Functions
---------
transpose
    Transpose last two indices.
col
    Treat multi-dim array as a stack of column vectors.
row
    Treat multi-dim array as a stack of row vectors.
scal
    Treat multi-dim array as a stack of scalars.
matldiv
    Matrix division from left.
matrdiv
    Matrix division from right.
matmul
    Matrix multiplication with better type flexibility.
"""

import numpy as np
from numpy.linalg.linalg import transpose
from typing import Tuple
from . import gufuncs as gf

# =============================================================================
# Reshaping for linear algebra
# =============================================================================


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
# Shape for linear algebra
# =============================================================================


def return_shape_mat(x: np.ndarray, y: np.ndarray) -> Tuple[int, ...]:
    """Shape of result of broadcasted matrix multiplication
    """
    if x.ndim == 0 or y.ndim == 0:
        raise ValueError('Scalar operations not supported. Use mul.')
    if x.shape[-1] != y.shape[-2:][0]:
        msg = 'Inner matrix dimensions mismatch: {0} and {1}.'
        raise ValueError(msg.format(x.shape, y.shape))
    if y.ndim == 1:
        return x.shape[:-1]
    return np.broadcast(x[..., :1], y[..., :1, :]).shape


# =============================================================================
# Division & Multiplication
# =============================================================================


def matldiv(x: np.ndarray, y: np.ndarray, rcond=None,
            *args, **kwargs) -> np.ndarray:
    """Matrix division from left.

    Computes :math:`z = x \\ y = x^{-1} y` for square `x`, or :math:`x^+ y`
    for rectangular `x`.
    Pseudo-inverse version uses `gufunc.lstsq`.
    Full inverse version uses `gufunc.solve`.
    Both versions broadcast using gufunc machinery.

    Parameters
    ----------
    x : {(..., M, N), (..., N, N)} array_like
        Divisor or Denominator.
    y : {(..., N), (..., N, K)} array_like
        Dividend or Numerator.
    rcond : optional float = None
        passed to `numpy.linalg.lstsq`
    out : {(..., M), (..., M, K), (..., N), (..., N, K)} np.ndarray
        array to store the output in.

    Returns
    -------
    z : {(..., M), (..., M, K), (..., N), (..., N, K)} np.ndarray
        Quotient. It has the same type as `x` or `y`.

    Raises
    ------
    LinAlgError
        If `x` is not full rank (square) or if computation doesn't converge
        (rectangular).

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    try:
        return gf.solve(x, y, *args, **kwargs)
    except (np.linalg.LinAlgError, ValueError):
        pass
    return gf.lstsq(x, y, rcond, *args, **kwargs)[0]


def matrdiv(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from right.

    Computes :math:`z = x / y = x y^{-1}`, or :math:`x y^+` if y isn't square.
    Pseudo-inverse version uses `gufunc.lstsq`.
    Full inverse version uses `gufunc.solve`.
    Both versions broadcast using gufunc machinery.

    Parameters
    ----------
    x : {(..., M), (..., K, M)} array_like
        Dividend or Numerator.
    y : {(..., M, N), (..., M, M)} array_like
        Divisor or Denominator.
    rcond : float = 1e-15
        passed to `numpy.linalg.lstsq`
    out : {(..., N), (..., K, N), (..., M), (..., K, M)} np.ndarray
        array to store the output in.

    Returns
    -------
    z : {(..., N), (..., K, N), (..., M), (..., K, M)} np.ndarray
        Quotient. It has the same type as `x` or `y`.

    Raises
    ------
    LinAlgError
        If `y` is not full rank (square) or if computation doesn't converge
        (rectangular).

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    return transpose(matldiv(transpose(y), transpose(x), *args, **kwargs))


qr_modes = {'reduced': (gf.qr_m, gf.qr_n),
            'complete': (gf.qr_m, gf.qr_m),
            'r': (gf.qr_rm, gf.qr_rn),
            'raw': (gf.qr_rawm, gf.qr_rawn)}


def qr(x: np.ndarray, mode: str = 'reduced') -> (np.ndarray, np.ndarray):
    """QR decomposition.

    Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
    `K = min(M,N)`, except for mode `complete`, where `K = M`.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str, chosen from:
        reduced
            default, use minimum inner dimensionality,
        complete
            use maximum inner dimensionality
        r
            return `R` only,
        raw
            return `H`, containing both Householder reflectors and `R`,
            and `tau` containing scaling factors.

    Returns
    -------
    Q: ndarray (...,M,K)
        Matrix with orthonormal columns. Modes: reduced, complete.
    R: ndarray (...,K,N)
        Matrix with zeros below the diagonal. Modes: reduced, complete, r.
    H: ndarray (...,N,M)
        Transpose of matrix for use in Fortran. Above and on the diagonal: `R`.
        Below the diagonal: the Householder reflectors `v`. Modes: raw.
    tau: ndarray (...,K,)
        Scaling factors for Householder reflectors. Modes: raw.
    """
    if mode not in qr_modes.keys():
        raise ValueError('Modes known to qr: reduced, complete, r, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = qr_modes[mode][x.shape[-2] > x.shape[-1]]
    return ufunc[x]
