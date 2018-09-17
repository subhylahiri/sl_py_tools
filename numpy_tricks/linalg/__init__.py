# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Dec  7 17:20:43 2017
# @author: Subhy
# package: linalg_tricks
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
And versions of most `numpy`'s array creation routines.

Classes
-------
lnarray
    Subclass of `numpy.ndarray` with properties such as `inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
invarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Does not actually invert the matrix unless it has to: if you try to do
    anything other than matrix multiplication or multiplication by scalars.
lnmatrix
    Subclass of `lnarray` which swaps matrix/elementwise multiplication and
    division from the right.
ldarray
    `lnarray` subclass which overloads bitshift operators for matrix division.
    One of several reasons why this is a bad idea is that bitshifting has lower
    operator priority than division, so you will have to use parentheses often.

Examples
--------
>>> import numpy as np
>>> import linalg as sp
>>> x = sp.lnarray(np.random.rand(2, 3, 4))
>>> y = sp.lnarray(np.random.rand(2, 3, 4))
>>> z = x.inv @ y
>>> w = x @ y.inv
>>> u = x @ y.t
>>> v = (x.r @ y[:, None, ...].t).ur
>>> a = sp.ldarray(np.random.rand(2, 3, 4))
>>> b = sp.ldarray(np.random.rand(2, 3, 4))
>>> c = (a << b)
>>> d = (a >> b)
"""

from ._lnarray import lnarray, pinvarray, invarray, lnmatrix
from ._ldarray import ldarray
from ._linalg import trnsp, col, row, scal, matldiv, matrdiv
from ._gufuncs import (matmul, solve, rsolve, lstsq, rlstsq, qr, qr_tall,
                       lstsq_m, lstsq_n)
from ._creation_ln import *
# from ._creation_ld import *
