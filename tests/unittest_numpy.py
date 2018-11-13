# -*- coding: utf-8 -*-
"""
"""
import unittest as ut
from functools import wraps
from contextlib import contextmanager
import numpy as np
import sl_py_tools.numpy_tricks.linalg._lnarray as la

__all__ = [
        'TestCaseNumpy',
        'wrap_np_test',
        'wrap_not_np_test',
        'mismatch_str',
        'miss_str',
        'asa',
        'errstate'
        ]
# =============================================================================


def wrap_np_test(func):
    """Wrap a test from numpy for use as a unittest method.

    All additional arguments must be passed via keywords.
    """
    @wraps(func)
    def wrapped_test(self, actual, desired, msg=None, **kwds):
        """Wrapped test from numpy for use in a unittest method

        All additional arguments must be passed via keywords.
        """
        try:
            func(actual, desired, **kwds)
            return
        except AssertionError:
            pass
        raise self.failureException(msg)
    return wrapped_test


def wrap_not_np_test(func):
    """Wrap negation of a test from numpy for use as a unittest method

    All additional arguments must be passed via keywords.
    """
    @wraps(func)
    def wrapped_test(self, actual, desired, msg=None, **kwds):
        """Wrapped negation of test from numpy for use in a unittest method

        All additional arguments must be passed via keywords.
        """
        try:
            func(actual, desired, **kwds)
        except AssertionError:
            return
        raise self.failureException(msg)
    return wrapped_test


class TestCaseNumpy(ut.TestCase):
    """Test case mith methods for comparing numpy arrays.

    Subclass this class to make your own unit test suite.
    It has several assertArray... methods that call numpy.testing functions and
    process the results how a unittest.TestCase method should.
    If you write a setUp method, be sure to call super().setUp().

    Methods
    -------
    assertArrayAlmostEqual
        calls numpy.testing.assert_array_almost_equal_nulp.
    assertArrayEqual
        calls numpy.testing.assert_array_equal.
    assertArrayAllClose
        calls numpy.testing.assert_allclose.
    assertArrayLess
        calls numpy.testing.assert_array_less.
    assertArrayNotAlmostEqual
        calls numpy.testing.assert_array_almost_equal_nulp and negates result.
    assertArrayNotEqual
        calls numpy.testing.assert_array_equal and negates result.
    assertArrayNotAllClose
        calls numpy.testing.assert_allclose and negates result.
    assertArrayNotLess
        calls numpy.testing.assert_array_less and negates result.
    """
    assertArrayAlmostEqual = wrap_np_test(
            np.testing.assert_array_almost_equal_nulp)
    assertArrayMaxDiff = wrap_np_test(np.testing.assert_array_max_ulp)
    assertArrayEqual = wrap_np_test(np.testing.assert_array_equal)
    assertArrayAllClose = wrap_np_test(np.testing.assert_allclose)
    assertArrayLess = wrap_np_test(np.testing.assert_array_less)

    assertArrayNotAlmostEqual = wrap_not_np_test(
            np.testing.assert_array_almost_equal_nulp)
    assertArrayNotEqual = wrap_not_np_test(np.testing.assert_array_equal)
    assertArrayNotAllClose = wrap_not_np_test(np.testing.assert_allclose)
    assertArrayNotLess = wrap_not_np_test(np.testing.assert_array_less)

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayAlmostEqual)
        self.addTypeEqualityFunc(la.lnarray, self.assertArrayAlmostEqual)


def mismatch_str(x, y):
    """Returns a string describing the maximum deviation of x and y
    """
    nulp = np.spacing(np.maximum(np.abs(x), np.abs(y)))
    mismatch = np.abs(x - y)
    mis_frac = mismatch / nulp
    ind = np.unravel_index(np.argmax(mis_frac), mis_frac.shape)
    formatter = 'Should be zero: {:.2g} or {:.2g} = {:.1f} * {:.2g}'

    return formatter.format(np.amax(mismatch), mismatch[ind], mis_frac[ind],
                            nulp[ind])


def miss_str(x, y, atol=1e-8, rtol=1e-5):
    """Returns a string describing the maximum deviation of x and y
    """
    shape = np.broadcast(x, y).shape
    thresh = atol + rtol * np.abs(np.broadcast_to(y, shape))
    mismatch = np.abs(x - y)
    mis_frac = mismatch / thresh
    ind = np.unravel_index(np.argmax(mis_frac), mis_frac.shape)
    formatter = 'Should be zero: {:.2g} or {:.2g} = {:.1f} * {:.2g}'

    return formatter.format(np.amax(mismatch), mismatch[ind], mis_frac[ind],
                            thresh[ind])


cmplx = {'i': 0, 'f': 0, 'd': 0, 'F': 1j, 'D': 1j}


def asa(x, y, sctype):
    """Convert x + iy to sctype
    """
    return (x + cmplx[sctype] * y).astype(sctype)


@contextmanager
def errstate(*args, **kwds):
    """Context manager like np.errstate that can also be used as a decorator
    """
    old_errstate = np.geterr()
    try:
        np.seterr(*args, **kwds)
        yield
    finally:
        np.seterr(**old_errstate)
