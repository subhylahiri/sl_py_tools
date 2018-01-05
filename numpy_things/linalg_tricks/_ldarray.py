# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Dec  7 16:41:56 2017
# @author: Subhy
# module: _ldarray
# =============================================================================
"""
Class that provides syntax for matrix division via bitshift operators.

Classes
-------
lnarray
    Subclass of `lnarray`, which is a subclass of `numpy.ndarray` with
    properties such as `inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
    This class overloads bitshift operators to perform matrix division.
    One of several reasons why this is a bad idea is that bitshifting has lower
    operator priority than division, so you will have to use parentheses often.

Examples
--------
>>> import numpy as np
>>> import linalg as sp
>>> a = sp.ldarray(np.random.rand(2, 3, 4))
>>> b = sp.ldarray(np.random.rand(2, 3, 4))
>>> c = (a << b)
>>> d = (a >> b)
"""

import numpy as np
from ._lnarray import lnarray

# =============================================================================
# Class: ldarray
# =============================================================================


class ldarray(lnarray):
    """Array object with linear algebra customisation.

    This is a subclass of `lnarray`, which is a subclass of `np.ndarray` with
    some added properties.
    This class overloads bitshift operators to perform matrix division.
    One of several reasons why this is a bad idea is that bitshifting has lower
    operator priority than division, so you will have to use parentheses often.
    The most important added property of `lnarray` is matrix division via a
    lazy inverse, which `ldarray` inherits.
    It also has some properties to work with broadcasting rules of `np.linalg`.

    Parameters
    ----------
    input_array : array_like
        The constructed array gets its data from `input_array`.
        Data is copied if necessary, as per `np.asarray`.

    Properties
    ----------
    inv : invarray
        Lazy inverse. When matrix multiplying, performs matrix division.
    t
        Transpose last two axes.
    c
        Insert singleton in last slot -> stack of column vectors.
    r
        Insert singleton in second last slot -> stack of row vectors.
    s
        Insert singleton in last two slots -> stack of scalars.
    uc, ur, us
        Undo effect of `r`, `c`, `s`.

    Examples
    --------
    >>> import numpy as np
    >>> import linalg as sp
    >>> a = sp.ldarray(np.random.rand(2, 3, 4))
    >>> b = sp.ldarray(np.random.rand(2, 3, 4))
    >>> c = (a << b)
    >>> d = (a >> b)

    See also
    --------
    `lnarray` : the super class.
    `np.ndarray` : the super-super class.
    `np.asarray` : used to get view/copy of data from `input_array`.
    `invarray` : class that provides another interface for matrix division.
    """
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

#    def __array_finalize__(self, obj):
#        # We are not adding any attributes
#        pass

    def __lshift__(self, other: np.ndarray) -> 'lnarray':
        """Left matrix division.

        See `linalg.matldiv`
        """
        return self.__matldiv__(other)

    def __rlshift__(self, other: np.ndarray) -> 'lnarray':
        """Reverse left matrix division.

        See `linalg.matldiv`
        """
        return self.__rmatldiv__(other)

    def __ilshift__(self, other: np.ndarray) -> 'lnarray':
        """In place left matrix division.

        self = other \ self

        See `linalg.matldiv`
        """
        self = self.__imatldiv__(other)
        return self

    def __rshift__(self, other: np.ndarray) -> 'lnarray':
        """Right matrix division.

        See `linalg.matrdiv`
        """
        return self.__matrdiv__(other)

    def __rrshift__(self, other: np.ndarray) -> 'lnarray':
        """Reverse right matrix division.

        See `linalg.matrdiv`
        """
        return self.__rmatrdiv__(other)

    def __irshift__(self, other: np.ndarray) -> 'lnarray':
        """In place right matrix division.

        See `linalg.matrdiv`
        """
        self = self.__imatrdiv__(other)
        return self
