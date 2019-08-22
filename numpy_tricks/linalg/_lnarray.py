# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:37:24 2017

@author: subhy

Classes that provide nicer syntax for matrix division and tools for working
with `numpy.linalg`'s broadcasting.

Classes
-------
lnarray
    Subclass of `numpy.ndarray` with properties such as `inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
pinvarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Does not actually invert the matrix. It will raise if you try to do
    anything other than matrix multiplication or multiplication by scalars.
invarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Does not actually invert the matrix. It will raise if you try to do
    anything other than matrix multiplication or multiplication by scalars.
lnmatrix
    Subclass of `lnarray` with swapped elementwise/matrix multiplication and
    division.

Examples
--------
>>> import numpy as np
>>> import linalg as sp
>>> x = sp.lnarray(np.random.rand(2, 3, 4))
>>> y = sp.lnarray(np.random.rand(2, 3, 4))
>>> z = x.inv @ y
>>> w = x @ y.inv
>>> u = x @ y.t
>>> v = (x.r @ y.t).ur
"""
from __future__ import annotations
import numpy as np
import numpy.lib.mixins as _mix
import numpy_linalg as la


# =============================================================================
# Export functions
# =============================================================================


__all__ = [
        'lnarray',
        'lnmatrix'
        ]


# =============================================================================
# Class: lnarray
# =============================================================================


class lnarray(la.lnarray):
    """Array object with linear algebra customisation.

    This is a subclass of `np.ndarray` with some added properties.
    The most important is matrix division via a lazy inverse.
    It also has some properties to work with broadcasting rules of `np.linalg`.

    Parameters
    ----------
    input_array : array_like
        The constructed array gets its data from `input_array`.
        Data is copied if necessary, as per `np.asarray`.

    Properties
    ----------
    pinv : pinvarray
        Lazy pseudoinverse. When matrix multiplying, performs matrix division.
        Note: call as a property. If you call it as a function, you'll get the
        actual pseudoinverse.
    inv : invarray
        Lazy inverse. When matrix multiplying, performs matrix division.
        Note: call as a property. If you call it as a function, you'll get the
        actual inverse.
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
    m : lnarray
        An `lnmatrix` view of the object.

    Examples
    --------
    >>> import numpy as np
    >>> import linalg as sp
    >>> x = sp.lnarray(np.random.rand(2, 3, 4))
    >>> y = sp.lnarray(np.random.rand(2, 3, 4))
    >>> z = x.pinv @ y
    >>> w = x @ y.pinv
    >>> u = x @ y.t
    >>> v = (x.r @ y.t).ur

    See also
    --------
    `np.ndarray` : the super class.
    `np.asarray` : used to get view/copy of data from `input_array`.
    `pinvarray` : class that provides an interface for matrix division.
    `invarray` : class that provides an interface for matrix division.
    `lnmatrix` : `lnarray`, but with matrix/elementwise operators swapped
    `ldarray` : class that provides another interface for matrix division.
    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    @property
    def m(self) -> lnmatrix:
        """lnmatrix view
        """
        return self.view(lnmatrix)


# =============================================================================
# Class: lnmatrix
# =============================================================================


class lnmatrix(lnarray):
    """Matrix version of `lnarray` (Not recommended)

    This subclass of `lnarray` is solely meant for linear algebra.
    It just swaps matrix/elementwise multiplication and division.
    Note that there is no left-division.

    It is probably best to keep these objects ephemeral: keep them on the
    right-hand-side of assignments and convert back to `lnarrays` asap.
    Otherwise it could spread to the rest of the variables.
    As `lnmatrix` has higher array priority than `lnarray`, it can be highly
    contagious.

    Attributes
    ----------
    a : lnarray
        An `lnarray` view of the object.
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

    __array_priority__ = 2.

    __mul__ = lnarray.__matmul__
    __rmul__ = lnarray.__rmatmul__
    __imul__ = lnarray.__imatmul__
    __matmul__ = lnarray.__mul__
    __rmatmul__ = lnarray.__rmul__
    __imatmul__ = lnarray.__imul__
    (__truediv__,
     __rtruediv__,
     __itruediv__) = _mix._numeric_methods(la.matrdiv, 'matrdiv')

    @property
    def a(self) -> lnarray:
        """lnarray view
        """
        return self.view(lnarray)
