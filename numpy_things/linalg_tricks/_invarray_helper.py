# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:42:42 2017

@author: subhy

helper functions for invarray
"""
from numbers import Number
from typing import Tuple, Union
from functools import wraps
import numpy as np
import numpy.lib.mixins as np_mixin

Shaped = Union[np.ndarray, 'invarray']

# =============================================================================
# Helpers for class: invarray
# =============================================================================


def _out_shape_inv(a: Shaped, b: Shaped) -> Tuple[int, ...]:
    """shape of inverse result of array broadcasting
    """
    # handle invarrays
    x = np.empty(a.shape)
    y = np.empty(b.shape)
    shc = np.broadcast(x, y).shape
    # this will be going in an invarray, so transpose
    return shc[:-2] + (shc[-1], shc[-2])


def _out_shape_mat(a: Shaped, b: Shaped) -> Tuple[int, ...]:
    """shape of result of array broadcasting with matrix multiplication
    """
    # handle invarrays
    x = np.empty(a.shape)
    y = np.empty(b.shape)
    return np.broadcast(x[..., :1], y[..., :1, :]).shape


def _isscalar(x) -> bool:
    """Is it a (stack of) scalars
    """
    if isinstance(x, Number):
        return True
    return isinstance(x, np.ndarray) and all(n == 1 for n in x.shape[-2:])


def _enable_ufunc(func):
    """Temporarily enable (fake) ufuncs between invarrays
    """
    @wraps(func)
    def with_ufunc(self, other):
        if not np_mixin._disables_array_ufunc(other):
            return func(self, other)
        # ufunc disabled.
        # Better be due to invarray, otherwise don't know what to do.
        if not isinstance(other, type(self)):
            return NotImplemented
        other.__array_ufunc__ = 1
        output = func(self, other)
        other.__array_ufunc__ = None
        return output
    return with_ufunc


def _inv_methods(ufunc, name: str):
    """Implement forward, reflected, inplace binary methods with a fake ufunc.
    """
    methods = np_mixin._numeric_methods(ufunc, name)
    return tuple(_enable_ufunc(fun) for fun in methods)
