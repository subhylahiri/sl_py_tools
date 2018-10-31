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
    Reversed matrix multiplication.
solve
    Solve systems of linear equations.
rsolve
    Solve reversed systems of linear equations.
lstsq
    Least square solution of systems of linear equations.
rlstsq
    Least square solution of reversed systems of linear equations.
norm
    Vector 2-norm.
qr_*
    QR factorization in various forms.

They raise ValueError instead of LinAlgError

`numpy.linalg` broadcasting rules apply. 1D arrays are not dealt with here (but
see `lnarray` class).
"""

from ._gufuncs_cloop import norm, rtrue_divide  # matmul, rmatmul
from ._gufuncs_blas import matmul, rmatmul  # norm
from ._gufuncs_lapack import solve, rsolve, lstsq, rlstsq
from ._gufuncs_lapack import qr_m, qr_n, qr_rm, qr_rn, qr_rawm, qr_rawn