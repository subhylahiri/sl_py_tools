# -*- coding: utf-8 -*-
# =============================================================================
# Created on Sun Dec  3 21:37:24 2017
# @author: subhy
# module: _lnarray
# =============================================================================
"""
Classes that provide nicer syntax for matrix division and tools for working
with `numpy.linalg`'s broadcasting.

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

import numpy as np
from numpy.lib.mixins import _numeric_methods
from typing import Optional, Tuple
from . import _linalg
from . import _invarray_helper as invh


# =============================================================================
# Class: lnarray
# =============================================================================


class lnarray(np.ndarray):
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
    inv : invarray
        Lazy inverse. When matrix multiplying, performs matrix division.
        Note: call as a property. If you call it as a function, you'll get the
        actual (pseudo)inverse
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
    >>> z = x.inv @ y
    >>> w = x @ y.inv
    >>> u = x @ y.t
    >>> v = (x.r @ y.t).ur

    See also
    --------
    `np.ndarray` : the super class.
    `np.asarray` : used to get view/copy of data from `input_array`.
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

#    def __array_finalize__(self, obj):
#        # We are not adding any attributes
#        pass

    __array_priority__ = 1.
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(_linalg.matmul,
                                                            'matmul')
    __matldiv__, __rmatldiv__, __imatldiv__ = _numeric_methods(_linalg.matldiv,
                                                               'matldiv')
    __matrdiv__, __rmatrdiv__, __imatrdiv__ = _numeric_methods(_linalg.matrdiv,
                                                               'matrdiv')

    @property
    def t(self) -> 'lnarray':
        """Transpose last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as stacks of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)
        """
        return self.swapaxes(-1, -2)

    @property
    def r(self) -> 'lnarray':
        """Treat multi-dim array as a stack of row vectors.

        Inserts a singleton axis in second-last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., 1, N)
        """
        return self[..., None, :]

    @property
    def c(self) -> 'lnarray':
        """Treat multi-dim array as a stack of column vectors.

        Inserts a singleton axis in last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., N, 1)
        """
        return self[..., None]

    @property
    def s(self) -> 'lnarray':
        """Treat multi-dim array as a stack of scalars.

        Inserts singleton axes in last two slots.

        Parameters/Results
        ------------------
        a : lnarray, (...,) --> expanded : lnarray, (..., 1, 1)
        """
        return self[..., None, None]

    @property
    def ur(self) -> 'lnarray':
        """Undo effect of `r`.

        Parameters/Results
        ------------------
        a : lnarray, (..., 1, N) --> squeezed : lnarray, (..., N)

        Raises
        ------
        ValueError
            If a.shape[-2] != 1
        """
        return self.squeeze(axis=-2)

    @property
    def uc(self) -> 'lnarray':
        """Undo effect of `c`.

        Parameters/Results
        ------------------
        a : lnarray, (..., N, 1) --> squeezed : lnarray, (..., N)

        Raises
        ------
        ValueError
            If a.shape[-1] != 1
        """
        return self.squeeze(axis=-1)

    @property
    def us(self) -> 'lnarray':
        """Undo effect of `s`.

        Parameters/Results
        ------------------
        a : lnarray, (..., 1, 1) --> squeezed : lnarray, (...,)

        Raises
        ------
        ValueError
            If a.shape[-2] != 1 or a.shape[-1] != 1
        """
        return self.squeeze(axis=-2).squeeze(axis=-1)

    @property
    def inv(self) -> 'invarray':
        """Lazy matrix inverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> a_inv : invarray, (..., N, M)
            When matrix multiplying, performs matrix division.

        Raises
        ------
        ValueError
            If a.shape[-2] != 1

        See also
        --------
        `invarray`
        """
        return invarray(self)

    @property
    def m(self) -> 'lnmatrix':
        """lnmatrix view
        """
        return self.view(lnmatrix)


# =============================================================================
# %%|| Helpers for class: invarray
# =============================================================================


# guaranteed to have at least one invarray
def _inv_matmul(a, b, out=(None,)):
    """Matrix multiplication for invarrays, mimicking ufunc interface
    """
    if out[0] is None:
        fout = np.empty(invh._out_shape_mat(a, b)).view(lnarray)
        if isinstance(a, invarray) and isinstance(b, invarray):
            fout = invarray(fout.t)
        out = (fout,)

    if isinstance(a, invarray) and isinstance(b, invarray):
        # out is invarray
        _linalg.matmul(b._to_invert, a._to_invert, out=(out[0]._to_invert,))
        out[0]._inverted = None
    elif isinstance(a, invarray) and isinstance(b, np.ndarray):
        _linalg.matldiv(a._to_invert, b, out=out)
    elif isinstance(a, np.ndarray) and isinstance(b, invarray):
        _linalg.matrdiv(a, b._to_invert, out=out)
    else:
        return NotImplemented

    return out[0]


def _inv_matldiv(a, b, out=(None,)):
    """Matrix left-division for invarrays, mimicking ufunc interface
    """
    if out[0] is None:
        fout = np.empty(invh._out_shape_mat(a, b)).view(lnarray)
        if isinstance(a, np.ndarray) and isinstance(b, invarray):
            fout = invarray(fout.t)
        out = (fout,)

    if isinstance(a, invarray) and isinstance(b, invarray):
        _linalg.matrdiv(b._to_invert, a._to_invert, out=out)
    elif isinstance(a, invarray) and isinstance(b, np.ndarray):
        _linalg.matmul(a._to_invert, b, out=out)
    elif isinstance(a, np.ndarray) and isinstance(b, invarray):
        # out is invarray
        _linalg.matmul(b._to_invert, a, out=(out[0]._to_invert,))
        out[0]._inverted = None
    else:
        return NotImplemented

    return out[0]


def _inv_matrdiv(a, b, out=(None,)):
    """Matrix right-division for invarrays, mimicking ufunc interface
    """
    if out[0] is None:
        fout = np.empty(invh._out_shape_mat(a, b)).view(lnarray)
        if isinstance(a, invarray) and isinstance(b, np.ndarray):
            fout = invarray(fout.t)
        out = (fout,)

    if isinstance(a, invarray) and isinstance(b, invarray):
        _linalg.matldiv(a._to_invert, b._to_invert, out=(fout,))
    elif isinstance(a, invarray) and isinstance(b, np.ndarray):
        # out is invarray
        _linalg.matmul(b, a._to_invert, out=(out[0]._to_invert,))
        out[0]._inverted = None
    elif isinstance(a, np.ndarray) and isinstance(b, invarray):
        _linalg.matmul(b._to_invert, a, out=(fout,))
    else:
        return NotImplemented

    return out[0]


def _inv_mul(a, b, out=(None,)):
    """Elementwise multiplication for invarrays, mimicking ufunc interface
    """
    if out[0] is None:
        out = (invarray(np.empty(invh._out_shape_inv(a, b)).view(lnarray)),)

    if isinstance(a, invarray) and invh._isscalar(b):
        np.true_divide(a._to_invert, b, out=(out[0]._to_invert,))
    elif invh._isscalar(a) and isinstance(b, invarray):
        np.true_divide(b._to_invert, a, out=(out[0]._to_invert,))
    else:
        return NotImplemented

    out[0]._inverted = None
    return out[0]


def _inv_truediv(a, b, out=(None,)):
    """Elementwise division for invarrays, mimicking ufunc interface
    """
    if out[0] is None:
        out = (invarray(np.empty(invh._out_shape_inv(a, b)).view(lnarray)),)

    if isinstance(a, invarray) and invh._isscalar(b):
        np.multiply(a._to_invert, b, out=(out[0]._to_invert,))
    else:
        return NotImplemented

    out[0]._inverted = None
    return out[0]


# =============================================================================
# Class: invarray
# =============================================================================


class invarray(object):
    """Lazy matrix (pseudo)inverse of `lnarray`.

    Does not actually perform the matrix inversion unless it has to.
    It will use matrix division for matmul (@) with an `lnarray`.

    It is intended to be ephemeral, appearing in larger expressions rather than
    being stored in a variable.
    Use `invarray.get()` to get the actual (pseudo)inverse.

    Methods
    -------
    self() -> lnarray
        Returns the actual, concrete (pseudo)inverse, calculating it if it has
        not already been done.
    inv : lnarray
        Returns the original array that needed inverting.

    Notes
    -----
    It can also be multiplied and divided by a nonzero scalar or stack of
    scalars, i.e. `ndarray` with last two dimensions singletons.
    Actually divides/multiplies the pre-inversion object.

    *Any* other operation or attribute access will require actually
    performing the (pseudo)inversion and using that instead (except for `len`,
    `shape`, `size`, 'ndim`, `repr`, `str`, `t` and `inv`).

    If the `lnarray` instance to be inverted is square, uses `np.linalg.solve`
    or `np.linalg.inv`.
    If it is two-dimensional and rectangular, uses `np.linalg.lstsq` or
    `np.linalg.pinv`.
    Otherwise it uses the standard formula for the Moore-Penrose pseudoinverse,
    assuming full-rank.

    Examples
    --------
    >>> import numpy as np
    >>> import linalg as sp
    >>> x = sp.lnarray(np.random.rand(2, 3, 4))
    >>> y = sp.lnarray(np.random.rand(2, 3, 4))
    >>> z = x.inv @ y
    >>> w = x @ y.inv

    Raises
    ------
    LinAlgError
        If a is not full rank (square, or broadcasting non-square) or
        if computation doesn't converge (non-square, no broadcasting).

    See also
    --------
    `lnarray` : the array class used.
    `matldiv`, `matrdiv`
    `invarray` : class that provides another interface for matrix division.
    """
    _to_invert: lnarray
    _inverted: Optional[lnarray]

    def __init__(self, to_invert: lnarray):
        if isinstance(to_invert, lnarray) and not isinstance(to_invert,
                                                             lnmatrix):
            # don't want to mess up subclasses, so that `inv` returns input
            self._to_invert = to_invert
        else:
            # in case input is `lnmatrix`, `ndarray` or `array_like`
            self._to_invert = np.asarray(to_invert).view(lnarray)
        self._inverted = None
#        if to_invert.ndim < 2:
#            msg = "Array to be inverted must have ndim >= 2, "
#            msg += "but to_invert.ndim = {}. ".format(to_invert.ndim)
#            msg += "Must be a matrix, or stack of matrices."
#            raise ValueError(msg)

    # needed for ndarray @ invarray to work.
    __array_ufunc__ = None
#    __array_priority__ = 2.

    # not sure about including this.
#    def __getattr__(self, name):
#        """Get `np.ndarray proerties from actual (pseudo)inverse.
#
#        Get unknown attributes from `lnarray` stored as `self._inverted`.
#
#        Notes
#        -----
#        If self._to_invert has not been (pseudo)inverted, it will compute the
#        (pseudo)inverse first.
#        """
#        if hasattr(self._to_invert, name):
#            return getattr(self.get(), name)
#        else:
#            raise AttributeError

    # make this __call__? @property? get()? do()?
    def __call__(self) -> lnarray:
        """Get actual (pseudo)inverse

        Returns
        -------
        inverted
            The (pseudo)inverse of the `lnarray` whose `inv` this object is,
            stored as `self._inverted`.

        Notes
        -----
        If self._to_invert has not been (pseudo)inverted, it will compute the
        (pseudo)inverse first.
        Otherwise, it will use the stored value.
        """
        if self._inverted is None:
            self._invert()
        return self._inverted

    __matmul__, __rmatmul__, __imatmul__ = invh._inv_methods(_inv_matmul,
                                                             'matmul')
    __matldiv__, __rmatldiv__, __imatldiv__ = invh._inv_methods(_inv_matldiv,
                                                                'matldiv')
    __matrdiv__, __rmatrdiv__, __imatrdiv__ = invh._inv_methods(_inv_matrdiv,
                                                                'matrdiv')
    __mul__, __rmul__, __imul__ = invh._inv_methods(_inv_mul, 'mul')
    __truediv__, __rtruediv__, __itruediv__ = invh._inv_methods(_inv_truediv,
                                                                'truediv')

    def __pos__(self) -> 'invarray':
        return invarray(self._to_invert)

    def __neg__(self) -> 'invarray':
        return invarray(-(self._to_invert))

    def __len__(self):
        return self.shape[0]

    def __str__(self) -> str:
        return str(self._to_invert) + '**(-1)'

    def __repr__(self) -> str:
        extra = len(type(self._to_invert).__name__) - 8
        # len('invarray') == 8
        rep = repr(self._to_invert).replace("\n" + " " * extra,
                                            "\n" + " " * -extra)
        return "invarray" + rep[(extra + 8):]

    @property
    def shape(self) -> Tuple[int, ...]:
        # Matrix operations are allowed with x.inv when allowed for x.t
        return self._to_invert.t.shape

    @property
    def ndim(self) -> int:
        return self._to_invert.ndim

    @property
    def size(self) -> int:
        return self._to_invert.size

    @property
    def dtype(self) -> np.dtype:
        return self._to_invert.dtype

    @property
    def inv(self) -> lnarray:
        return self._to_invert

    @property
    def t(self) -> 'invarray':
        """A copy of object, but view of data"""
        return invarray(self._to_invert.t)

    def copy(self) -> 'invarray':
        """Copy data"""
        return invarray(self._to_invert.copy())

    def _invert(self):
        """Actually perform (pseudo)inverse
        """
        if self.ndim < 2:
            # scalar or vector
            self._inverted = (self._to_invert /
                              np.linalg.norm(self._to_invert)**2)
        elif self.shape[-2] == self.shape[-1]:
            # square
            self._inverted = np.linalg.inv(self._to_invert)
        elif self.ndim == 2:
            # no broadcasting needed
            self._inverted = np.linalg.pinv(self._to_invert)
        elif self.shape[-2] > self.shape[-1]:
            # tall and skinny
            self._inverted = (self._to_invert.t @
                              (self._to_invert @ self._to_invert.t).inv)
        elif self.shape[-2] < self.shape[-1]:
            # short and fat
            self._inverted = ((self._to_invert.t @ self._to_invert).inv @
                              self._to_invert.t)
        else:
            raise ValueError('Nothing to invert?' + str(self._to_invert))


# =============================================================================
# Class: lnmatrix
# =============================================================================


class lnmatrix(lnarray):
    """Matrix version of `lnarray`

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
    __truediv__ = lnarray.__matrdiv__
    __rtruediv__ = lnarray.__rmatrdiv__
    __itruediv__ = lnarray.__imatrdiv__
    __matrdiv__ = lnarray.__truediv__
    __rmatrdiv__ = lnarray.__rtruediv__
    __imatrdiv__ = lnarray.__itruediv__

    @property
    def a(self) -> lnarray:
        """lnarray view
        """
        return self.view(lnarray)
