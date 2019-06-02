# -*- coding: utf-8 -*-
"""
Functions for testing array values.
"""
import numpy as np


def allfinite(*arrays) -> bool:
    """Check if all array elements are finite

    Returns `True` if no element of any array is `nan` or `inf`.
    """
    return all(np.isfinite(arr).all() for arr in arrays)


def anyclose(x, y, *args, **kwds) -> bool:
    """Are any elements close?

    Like numpy.allclose but with any instead of all.
    """
    return np.isclose(x, y, *args, **kwds).any()


def tri_low_rank(array, *args, **kwds):
    """Check for low rank triangular matrix

    Returns `True` if any diagonal element is close to 0.
    Does not check if the array is triangular. It can be used on the 'raw'
    forms of lu/qr factors.
    """
    return anyclose(np.diagonal(array), 0., *args, **kwds)
