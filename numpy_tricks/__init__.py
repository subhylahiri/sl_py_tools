# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:28:08 2018

@author: subhy
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
