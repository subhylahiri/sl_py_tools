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

from typing import Optional, Tuple, Sequence
from collections.abc import Sequence as SequenceType
import numpy as np
import numpy.lib.mixins as _mix
from . import _linalg as la
from . import gufuncs as gf
from . import convert_loop as cv


# =============================================================================
# Export functions
# =============================================================================


__all__ = [
        'lnarray',
        'invarray',
        'pinvarray',
        'lnmatrix'
        ]


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
    # Set of ufuncs that need special handling of vectors (index gives case)
    vec_ufuncs = (gf.matmul, gf.lstsq, gf.solve,
                  gf.rlstsq, gf.rmatmul, None, gf.rsolve)

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

#    def __array_finalize__(self, obj):
#        # We are not adding any attributes
#        pass
    (__matmul__,
     __rmatmul__,
     __imatmul__) = _mix._numeric_methods(gf.matmul, 'matmul')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Customise ufunc behaviour
        """
        args = list(cv.conv_loop_in_view(lnarray, inputs)[0])
        if ufunc in self.vec_ufuncs:
            args[0], args[1], squeeze = gf.vec2mat(
                args[0], args[1], self.vec_ufuncs.index(ufunc)
            )
        args = tuple(args)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = cv.conv_loop_in_view(lnarray, outputs)[0]
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        with np.errstate(invalid='raise'):
            results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)
        if ufunc in self.vec_ufuncs and any(squeeze):
            results = (gf.mat2vec(results[0], squeeze),) + results[1:]
        results = cv.conv_loop_out_view(self, results, outputs)

        return results[0] if len(results) == 1 else results

    def flattish(self, start: int, stop: int) -> 'lnarray':
        """Partial flattening.

        Flattens those axes in the range [start:stop)
        """
        newshape = self.shape[:start] + (-1,) + self.shape[stop:]
        return self.reshape(newshape)

    def expand_dims(self, axis) -> 'lnarray':
        """Expand the shape of the array with length one axes

        Alias of `numpy.expand_dims` when `axis` is a single `int`. If `axis`
        is a sequence of `int`, axis numbers are relative to the *final* shape.
        """
        if isinstance(axis, int):
            return np.expand_dims(self, axis).view(type(self))
        elif not isinstance(axis, SequenceType):
            raise TypeError("axis must be an int or a sequence of ints")
        elif len(axis) == 0:
            return self
        axes = np.sort(np.mod(axis, self.ndim + len(axis)))
        if not np.diff(axes).all():
            raise ValueError('repeated axes: '
                             + str(axes[np.nonzero(np.diff(axes) == 0)]))
        return self.expand_dims(axes[0]).expand_dims(axes[1:])

    @property
    def t(self) -> 'lnarray':
        """Transpose last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as arrays of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)
        """
        return self.swapaxes(-1, -2)

    @property
    def h(self) -> 'lnarray':
        """Hermitian-conjugate over last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as arrays of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)
        """
        return self.conj().swapaxes(-1, -2)

    @property
    def r(self) -> 'lnarray':
        """Treat multi-dim array as a stack of row vectors.

        Inserts a singleton axis in second-last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., 1, N)
        """
        return self.expand_dims(-2)

    @property
    def c(self) -> 'lnarray':
        """Treat multi-dim array as a stack of column vectors.

        Inserts a singleton axis in last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., N, 1)
        """
        return self.expand_dims(-1)

    @property
    def s(self) -> 'lnarray':
        """Treat multi-dim array as a stack of scalars.

        Inserts singleton axes in last two slots.

        Parameters/Results
        ------------------
        a : lnarray, (...,) --> expanded : lnarray, (..., 1, 1)
        """
        return self.expand_dims((-1, -2))

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
        return self.squeeze(axis=(-2, -1))

    @property
    def pinv(self) -> 'pinvarray':
        """Lazy matrix pseudoinverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> a_inv : pinvarray, (..., N, M)
            When matrix multiplying, performs matrix division.

        See also
        --------
        `pinvarray`
        """
        return pinvarray(self)

    @property
    def inv(self) -> 'invarray':
        """Lazy matrix inverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, M) --> a_inv : pinvarray, (..., M, M)
            When matrix multiplying, performs matrix division.

        Raises
        ------
        ValueError
            If a.shape[-2] != a.shape[-1]

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
# Class: pinvarray
# =============================================================================

_div_status = {gf.matmul: (False, False),
               gf.rmatmul: (False, False),
               gf.solve: (True, False),
               gf.rsolve: (False, True),
               gf.lstsq: (True, False),
               gf.rlstsq: (False, True)}


def _inv_output(ufunc, pinv_in: Sequence[bool]) -> bool:
    """Should the ufunc output be converted to (p)invarray?
    """
    if ufunc in {np.multiply, np.true_divide}:
        return True
    if ufunc not in _div_status.keys():
        return False
    return all(x ^ y for x, y in zip(pinv_in, _div_status[ufunc]))


# =============================================================================
# Class: pinvarray
# =============================================================================


class pinvarray(gf.LNArrayOperatorsMixin):
    """Lazy matrix pseudoinverse of `lnarray`.

    Does not actually perform the matrix pseudoinversion.
    It will use least-square matrix division for matmul (@) with an `lnarray`.

    It is intended to be ephemeral, appearing in larger expressions rather than
    being stored in a variable.
    Use `pinvarray()` to get the actual pseudoinverse.

    Methods
    -------
    self() -> lnarray
        Returns the actual, concrete pseudoinverse, calculating it if it has
        not already been done.
    pinv : lnarray
        Returns the original array that needed inverting.

    Notes
    -----
    It can also be multiplied and divided by a nonzero scalar or stack of
    scalars, i.e. `ndarray` with last two dimensions singletons.
    Actually divides/multiplies the pre-inversion object.
    It will also behave appropriately in the `(r)lstsq` functions from *this*
    package (not the `numpy` versions).

    *Any* other operation or attribute access would require actually
    performing the pseudoinversion and using that instead (except for `len`,
    `shape`, `size`, 'ndim`, `repr`, `str`, `t`, `h` and `inv`).

    It uses `gufuncs.lstsq`,`gufuncs.rlstsq` for @ and `np.linalg.pinv` for ().

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
        If computation does not converge.

    See also
    --------
    `lnarray` : the array class used.
    `matldiv`, `matrdiv`
    `invarray` : class that provides an interface for matrix division.
    """
    _to_invert: lnarray
    _inverted: Optional[lnarray]

    # _ufunc_map[ufunc_in][arg1][arg2] -> ufunc_out, where:
    # ufunc_in = ufunc we were given
    # ar1/arg2 = is the first/second argument a pinvarray?
    # ufunc_out = ufunc to use instead
    _ufn_map = {
        # a b, a b^+, a^+ b, a^+ b^+:
        gf.matmul: [[None, gf.rlstsq], [gf.lstsq, None]],
        # a^+ b, a^+ b^+, a^++ b, a^++ b^+
        gf.lstsq: [[None, None], [gf.matmul, gf.rlstsq]],
        # a b^+, a b^++, a^+ b^+, a^+ b^++
        gf.rlstsq: [[None, gf.matmul], [None, gf.lstsq]],
        # a*b, a*b^+, a^+*b, a^+ * b^+
        np.multiply: [[None, gf.rtrue_divide], [np.true_divide, None]],
        # a/b, a/b^+ a^+/b, a^+/b^+
        np.true_divide: [[None, None], [np.multiply, None]]
    }

    # these ufuncs are passed on to self._to_invert
    _unary_ufuncs = {np.positive, np.negative}

    def __init__(self, to_invert: lnarray):
        if isinstance(to_invert, lnarray) and not isinstance(to_invert,
                                                             lnmatrix):
            # don't want to mess up subclasses, so that `pinv` returns input
            self._to_invert = to_invert
        else:
            # in case input is `lnmatrix`, `ndarray` or `array_like`
            self._to_invert = np.asarray(to_invert).view(lnarray)
        self._inverted = None

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handling ufuncs with pinvarrays
        """
        # which inputs are we converting?
        # For most inputs, we swap multiplication & division instead of inverse
        args, pinv_in = cv.conv_loop_in_attr('_to_invert', pinvarray, inputs)

        pinv_out = [False] * ufunc.nout  # which outputs need converting back?
        if ufunc in self._ufunc_map.keys():
            pinv_out[0] = _inv_output(ufunc, pinv_in)
            ufunc = self._ufn_map[ufunc][pinv_in[0]][pinv_in[1]]
        elif ufunc in self._unary_ufuncs:
            # Apply ufunc to self._to_invert.
            # Already converted input; just need to convert output
            pinv_out[0] = True
        else:
            return NotImplemented
        if ufunc is None:
            return NotImplemented
        # Alternative: other ufuncs use implicit inversion.
        # Not used on the basis that explicit > implicit. Use __call__ instead.
        # args = []
        # for input_ in inputs:
        #     if isinstance(input_, pinvarray):
        #         args.append(input_())
        #     else:
        #         args.append(input_)
        # args = tuple(args)
        # return self._to_invert.__array_ufunc__(ufunc, method, *args,
        #                                        **kwargs)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args, pinv_out = cv.conv_loop_in_attr('_to_invert',
                                                      pinvarray, outputs)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = self._to_invert.__array_ufunc__(ufunc, method, *args,
                                                  **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = cv.conv_loop_out_init(self, results, outputs, pinv_out)

        return results[0] if len(results) == 1 else results

    # This would allow other operations to work with implicit inversion.
    # Not used on the basis that explicit > implicit. Use __call__ instead.
    # def __getattr__(self, name):
    #     """Get `np.ndarray proerties from actual (pseudo)inverse.
    #
    #     Get unknown attributes from `lnarray` stored as `self._inverted`.
    #
    #     Notes
    #     -----
    #     If self._to_invert has not been (pseudo)inverted, it will compute the
    #     (pseudo)inverse first.
    #     """
    #     if hasattr(self._to_invert, name):
    #         return getattr(self(), name)
    #     else:
    #         raise AttributeError

    # make this __call__? @property? get()? do()? __array__?
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

    def __len__(self):
        return self.shape[0]

    def __str__(self) -> str:
        return str(self._to_invert) + '**(-1)'

    def __repr__(self) -> str:
        namelen = len(type(self).__name__)
        extra = len(type(self._to_invert).__name__) - namelen
        # len('pinvarray') == 8
        rep = repr(self._to_invert).replace("\n" + " " * extra,
                                            "\n" + " " * -extra)
        return "pinvarray" + rep[(extra + namelen):]

    def swapaxes(self, axis1, axis2) -> 'pinvarray':
        """Interchange two axes in a copy

        Parameters
        ----------
        axis1 : int
            First axis.
        axis2 : int
            Second axis.

        Returns
        -------
        a_swapped : ndarray
            For NumPy >= 1.10.0, if `a.pinv` is an ndarray, then new pinvarray
            containing a view of `a.pinv` is returned; otherwise a new array is
            created.
        """
        if axis1 % self.ndim in {self.ndim - 1, self.ndim - 2}:
            axis1 = (-3 - axis1) % self.ndim
        if axis2 % self.ndim in {self.ndim - 1, self.ndim - 2}:
            axis2 = (-3 - axis2) % self.ndim
        my_t = type(self)
        return my_t(self._to_invert.copy().swapaxes(axis1, axis2))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Effective shape of pinvarray in matmul etc.
        """
        # Matrix operations are allowed with x.inv when allowed for x.t
        return self._to_invert.t.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions
        """
        return self._to_invert.ndim

    @property
    def size(self) -> int:
        """Number of elements
        """
        return self._to_invert.size

    @property
    def dtype(self) -> np.dtype:
        """Data type
        """
        return self._to_invert.dtype

    @property
    def pinv(self) -> lnarray:
        """Uninverted matrix
        """
        return self._to_invert

    @property
    def t(self) -> 'pinvarray':
        """A copy of object, but view of data"""
        return pinvarray(self._to_invert.t)

    @property
    def h(self) -> 'pinvarray':
        """A copy of object, but view of data"""
        return pinvarray(self._to_invert.h)

    def copy(self, order='C', **kwds) -> 'pinvarray':
        """Copy data"""
        my_t = type(self)
        _to_invert = kwds.pop('_to_invert', None)
        if _to_invert is None:
            _to_invert = self._to_invert.copy(order=order)
        return my_t(_to_invert, **kwds)

    def _invert(self):
        """Actually perform (pseudo)inverse
        """
        if self.ndim < 2:
            # scalar or vector
            self._inverted = (self._to_invert
                              / np.linalg.norm(self._to_invert)**2)
        elif self.ndim >= 2:
            # pinv broadcasts
            self._inverted = np.linalg.pinv(self._to_invert)
        else:
            raise ValueError('Nothing to invert? ' + str(self._to_invert))


# =============================================================================
# Class: invarray
# =============================================================================


class invarray(pinvarray):
    """Lazy matrix inverse of `lnarray`.

    Does not actually perform the matrix inversion.
    It will use matrix division for matmul (@) with an `lnarray`.

    It is intended to be ephemeral, appearing in larger expressions rather than
    being stored in a variable.
    Use `invarray()` to get the actual inverse.

    Methods
    -------
    self() -> lnarray
        Returns the actual, concrete inverse, calculating it if it has
        not already been done.
    inv : lnarray
        Returns the original array that needed inverting.

    Notes
    -----
    It can also be multiplied and divided by a nonzero scalar or stack of
    scalars, i.e. `ndarray` with last two dimensions singletons.
    Actually divides/multiplies the pre-inversion object.
    It will also behave appropriately in the `(r)solve` functions from *this*
    package (not the `numpy` versions).

    *Any* other operation or attribute access would require actually
    performing the inversion and using that instead (except for `len`,
    `shape`, `size`, 'ndim`, `repr`, `str`, `t` and `inv`).

    It uses `gufuncs.solve`, `gufuncs.rsolve` for @ and `np.linalg.inv` for ().

    Examples
    --------
    >>> import numpy as np
    >>> import linalg as sp
    >>> x = sp.lnarray(np.random.rand(2, 3, 3))
    >>> y = sp.lnarray(np.random.rand(2, 3, 3))
    >>> z = x.inv @ y
    >>> w = x @ y.inv

    Raises
    ------
    LinAlgError
        If original matrix is not full rank.
    ValueError
        if original matrix is not square.

    See also
    --------
    `lnarray` : the array class used.
    `matldiv`, `matrdiv`
    `pinvarray` : class that provides an interface for matrix psudo-division.
    """
    # _ufunc_map[ufunc_in][arg1][arg2] -> ufunc_out, where:
    # ufunc_in = ufunc we were given
    # ar1/arg2 = is the first/second argument a pinvarray?
    # ufunc_out = ufunc to use instead
    _ufn_map = {
        # a b, a b^-, a^- b, a^- b^-:
        gf.matmul: [[None, gf.rsolve], [gf.solve, gf.rmatmul]],
        # a^- b, a^- b^-, a^-- b, a^-- b^-
        gf.solve: [[None, gf.rmatmul], [gf.matmul, gf.rsolve]],
        # a b^-, a b^--, a^- b^-, a^- b^--
        gf.rsolve: [[None, gf.matmul], [gf.matmul, gf.solve]],
        # a*b, a*b^-, a^-*b, a^- * b^-
        np.multiply: [[None, gf.rtrue_divide], [np.true_divide, None]],
        # a/b, a/b^- a^-/b, a^-/b^-
        np.true_divide: [[None, None], [np.multiply, None]]
    }

    def __init__(self, to_invert: lnarray):
        super().__init__(to_invert)

        if to_invert.ndim < 2:
            msg = "Array to be inverted must have ndim >= 2, "
            msg += "but to_invert.ndim = {}. ".format(to_invert.ndim)
            msg += "Must be a matrix, or stack of matrices."
            raise ValueError(msg)

        if to_invert.shape[-1] != to_invert.shape[-2]:
            msg = "Array to be inverted must square "
            msg += "but to_invert.shape = {}. ".format(to_invert.shape)
            raise ValueError(msg)

    @property
    def pinv(self) -> lnarray:
        """Uninverted matrix
        """
        raise TypeError('This is an invarray, not a pinvarray!')

    @property
    def inv(self) -> lnarray:
        """Uninverted matrix
        """
        return self._to_invert

    def _invert(self):
        """Actually perform inverse
        """
        if self.ndim >= 2 and self.shape[-2] == self.shape[-1]:
            # square
            self._inverted = np.linalg.inv(self._to_invert)
        else:
            raise ValueError('Nothing to invert? ' + str(self._to_invert))


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
