# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Jul  6 12:10:02 2017
# @author: Subhy
# module: _linalg
# =============================================================================
"""
Linear algebra routines.

Functions
---------
trnsp
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
from typing import Tuple

# =============================================================================
# Reshaping for linear algebra
# =============================================================================


def trnsp(a: np.ndarray) -> np.ndarray:
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
    if a.ndim > 1:
        return a.swapaxes(-1, -2)
    else:
        return a


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


def matldiv(x: np.ndarray, y: np.ndarray,
            out=(None,), rcond=1e-15) -> np.ndarray:
    """Matrix division from left.

    Computes :math:`z = x \ y = x^{-1} y`, or :math:`x^+ y` for non-square `x`.
    Pseudo inverse version broadcasts (if broadcasting is needed, uses
    `np.linalg.solve` and assumes full rank `x`, else, uses `np.linalg.lstsq`).
    Full inverse version broadcasts (uses `np.linalg.solve`).

    Parameters
    ----------
    x : {(..., M, N), (..., N, N)} array_like
        Divisor or Denominator.
    y : {(..., N), (..., N, K)} array_like
        Dividend or Numerator.
    out : {(..., M), (..., M, K), (..., N), (..., N, K)} np.ndarray
        array to store the output in.
    rcond : float = 1e-15
        passed to `numpy.linalg.lstsq`

    Returns
    -------
    z : {(..., M), (..., M, K), (..., N), (..., N, K)} np.ndarray
        Quotient. It has the same type as `a`, unless type(b) is a subclass of
        type(a), in which case `c` has the same type as `b`.

    Raises
    ------
    LinAlgError
        If `x` is not full rank (square, or broadcasting non-square) or
        if computation doesn't converge (non-square, no broadcasting).

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    a = np.asarray(x)
    b = np.asarray(y)
    if a.dtype == np.object_ or b.dtype == np.object_:
        return NotImplemented

    returntype = type(a)
    if isinstance(x, returntype):
        returntype = type(x)
    if isinstance(y, returntype):
        returntype = type(y)

    if out[0] is None:
        out = (np.empty(return_shape_mat(trnsp(a), b)).view(returntype),)

    if a.ndim == 0:
        out[0][...] = b / a
    elif a.ndim == 1:
        out[0][...] = a @ b / (a @ a)
    elif a.shape[-2] == a.shape[-1]:
        out[0][...] = np.linalg.solve(a, b)
    elif a.ndim == 2 and b.ndim <= 2:
        out[0][...] = np.linalg.lstsq(a, b, rcond=-1)[0]
    elif a.shape[-2] < a.shape[-1]:
        out[0][...] = trnsp(a) @ np.linalg.solve(a @ trnsp(a), b)
    elif a.shape[-2] > a.shape[-1]:
        out[0][...] = np.linalg.solve(trnsp(a) @ a, trnsp(a) @ b)
    else:
        return NotImplemented

    return out[0]


def matrdiv(a: np.ndarray, b: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from right.

    Computes :math:`c = a / b = a b^{-1}`, or :math:`a b^+` if b isn't square.
    Pseudo inverse version broadcasts  (if broadcasting is needed, uses
    `np.linalg.solve` and assumes full rank a, else, uses `np.linalg.lstsq`).
    Full inverse version broadcasts (uses `np.linalg.solve`).

    Parameters
    ----------
    a : {(..., M), (..., K, M)} array_like
        Dividend or Numerator.
    b : {(..., M, N), (..., M, M)} array_like
        Divisor or Denominator.
    out : {(..., N), (..., K, N), (..., M), (..., K, M)} np.ndarray
        array to store the output in.
    rcond : float = 1e-15
        passed to `numpy.linalg.lstsq`

    Returns
    -------
    c : {(..., N), (..., K, N), (..., M), (..., K, M)} np.ndarray
        Quotient. It has the same type as `a`, unless type(b) is a subclass of
        type(a), in which case `c` has the same type as `b`.

    Raises
    ------
    LinAlgError
        If b is not full rank (square, or broadcasting non-square) or
        if computation doesn't converge (non-square, no broadcasting).

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    return trnsp(matldiv(trnsp(b), trnsp(a), *args, **kwargs))


def matmul(x, y, out=(None,)):
    """Matrix multiplication.

    `np.matmul` doesn't pass through subclasses. This just fixes that.
    Makes sure it returns NotImplemented instead of TypeError

    See also
    --------
    `np.matmul`
    """
    a = np.asarray(x)
    b = np.asarray(y)
    if a.dtype == np.object_ or b.dtype == np.object_:
        return NotImplemented

    returntype = type(a)
    if isinstance(x, returntype):
        returntype = type(x)
    if isinstance(y, returntype):
        returntype = type(y)

    if out[0] is None:
        out = (np.empty(return_shape_mat(a, b)).view(returntype),)

    np.matmul(a, b, out=out[0])

    return out[0]
