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
matmul
    Matrix multiplication.
matldiv
    Matrix division from left.
matrdiv
    Matrix division from right.
qr
    QR decomposition with broadcasting and subclass passing.
"""
import functools
import typing
import numpy as np
from numpy.linalg.linalg import transpose
from . import gufuncs as gf

__all__ = [
    'col',
    'row',
    'scal',
    'matmul',
    'matldiv',
    'matrdiv',
    'qr',
]

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


def return_shape_mat(x: np.ndarray, y: np.ndarray) -> typing.Tuple[int, ...]:
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


_vec_doc = """Does matrix-matrix, matrix-vector, vector-matrix and vector-vector
versions, with vector versions used *only* when one-dimensional.
"""
def _vec_wrap(gufunc, *args):
    """Wrap a gufunc with special handling for vectors
    """
    @functools.wraps(gufunc)
    def wrapper(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
        squeeze = [False, False]
        if x.ndim == 1:
            x = x[None, :]
            squeeze[0] = True
        if y.ndim == 1:
            y = y[:, None]
            squeeze[1] = True
        with np.errstate(invalid='raise'):
            try:
                z = gufunc(x, y, *args, **kwargs)
            except FloatingPointError:
                raise np.linalg.LinAlgError("Failure in linalg routine "
                                            + gufunc.__name__)

        axs = (-2,) * squeeze[0] + (-1,) * squeeze[1]
        z = z.squeeze(axis=axs)
        return z[()] if z.ndim == 0 else z
    wrapper.__doc__ = wrapper.__doc__.replace("\nParameters",
                                              "\n" + _vec_doc + "\nParameters")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,M,NRHS)",
                                              "(...,M,NRHS) or (M,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,NRHS,M)",
                                              "(...,NRHS,M) or (M,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,N,NRHS)",
                                              "(...,N,NRHS) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,NRHS,N)",
                                              "(...,NRHS,N) or (N,)")
    if len(args) > 0:
        wrapper.__doc__ = wrapper.__doc__.replace(*args)
    return wrapper

# =============================================================================
# Division & Multiplication
# =============================================================================
_bin_doc = "It is intended for use in binary operators.\n"
matmul = _vec_wrap(gf.matmul)
solve = _vec_wrap(gf.solve)
rsolve = _vec_wrap(gf.rsolve)
lstsq = _vec_wrap(gf.lstsq, _bin_doc, "")
rlstsq = _vec_wrap(gf.rlstsq, _bin_doc, "")


def matldiv(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from left.

    Computes :math:`z = x \\ y = x^{-1} y` for square `x`, or :math:`x^+ y`
    for rectangular `x`.
    Pseudo-inverse version uses `gufunc.lstsq`.
    Full inverse version uses `gufunc.solve`.
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
        If `x` is not full rank or not square, and computation doesn't converge.

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    if x.shape[-1] == x.shape[-2]:
        try:
            return solve(x, y, *args, **kwargs)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return lstsq(x, y, *args, **kwargs)


def matrdiv(x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from right.

    Computes :math:`z = x / y = x y^{-1}`, or :math:`x y^+` if y isn't square.
    Pseudo-inverse version uses `gufunc.lstsq`.
    Full inverse version uses `gufunc.solve`.
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
        If `x` is not full rank or not square, and computation doesn't converge.

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
    mode: str
        chosen from:
        **reduced** - default, use minimum inner dimensionality, ``K=min(M,N)``,
        **complete** - use inner dimensionality ``K=M``, for square `Q`,
        **r** - return `R` only,
        **raw** - return `H,tau`, which determine `Q` and `R` (see below).

    Returns
    -------
    Q: ndarray (...,M,K)
        Matrix with orthonormal columns. Modes: `reduced, complete`.
    R: ndarray (...,K,N)
        Matrix with zeros below the diagonal. Modes: `reduced, complete, r`.
    H: ndarray (...,N,M)
        Transpose of matrix for use in Fortran. Above and on the diagonal: `R`.
        Below the diagonal: the Householder reflectors `v`. Modes: `raw`.
    tau: ndarray (...,K,)
        Scaling factors for Householder reflectors. Modes: `raw`.
    """
    if mode not in qr_modes.keys():
        raise ValueError('Modes known to qr: reduced, complete, r, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = qr_modes[mode][x.shape[-2] > x.shape[-1]]
        return ufunc(x)
