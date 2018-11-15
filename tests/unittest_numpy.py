# -*- coding: utf-8 -*-
"""
"""
import unittest
import contextlib
import functools
import types
import sys
import numpy as np
import sl_py_tools.numpy_tricks.linalg._lnarray as la

__all__ = [
        'TestCaseNumpy',
        'miss_str',
        'asa',
        'errstate'
        ]
__unittest = True
# =============================================================================


def _fake_tb(lineno):
    return types.TracebackType(None, sys._getframe(2), 1, lineno)


class TestResultNumpy(unittest.TextTestResult):
    def _is_relevant_tb_level(self, tb):
        f_vars = {**tb.tb_frame.f_globals, **tb.tb_frame.f_locals}
        return '__unittest' in f_vars and '__notunittest' not in f_vars


class TestCaseNumpy(unittest.TestCase):
    """Test case mith methods for comparing numpy arrays.

    Subclass this class to make your own unit test suite.
    It has several assertArray... methods that call numpy functions and
    process the results like a unittest.TestCase method.
    If you write a setUp method, be sure to call super().setUp().

    Methods
    -------
    assertArrayAllClose
        Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose).
    assertArrayNotAllClose
        Calls numpy.allclose and negates result.
    assertArrayEqual
        Calls numpy.all(numpy.equal(...)).
    assertArrayNotEqual
        Calls numpy.all(numpy.not_equal(...)).
    assertArrayLess
        Calls numpy.all(numpy.less(...)).
    assertArrayNotLess
        Calls numpy.all(numpy.greater_equal(...)).
    assertArrayGreater
        Calls numpy.all(numpy.greater(...)).
    assertArrayNotGreater
        Calls numpy.all(numpy.less_equal(...)).
    """
    def setUp(self):
        self.all_close_opts = {'atol': 1e-6, 'rtol': 1e-5, 'equal_nan': True}
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayAllClose)
        self.addTypeEqualityFunc(la.lnarray, self.assertArrayAllClose)

#    def defaultTestResult(self):
#        return TestResultNumpy

    @classmethod
    def loop(cls, msg=None, attr_name='sctype', attr_inds=slice(None)):
        """Decorator to loop a test over an attribute
        """
        def loop_dec(func):
            @functools.wraps(func)
            def loop_func(self, *args, **kwds):
                the_attr = getattr(self, attr_name)
#                __notunittest = True
                for val in the_attr[attr_inds]:
                    opts = {attr_name: val}
                    with self.subTest(msg=msg, **opts):
                        func(self, *args, **opts, **kwds)
            return loop_func
        return loop_dec

    def assertArrayAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose) and processes the results like a
        unittest.TestCase method.
        """
        if msg is None:
            msg = miss_str(actual, desired, **self.all_close_opts)
        if not np.allclose(actual, desired, **self.all_close_opts):
            self.fail(msg)

    def assertArrayEqual(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.equal(...)) and processes the results like a
        unittest.TestCase method.
        """
        if np.any(actual != desired):
            self.fail(msg)

    def assertArrayLess(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.less(...)) and processes the results like a
        unittest.TestCase method.
        """
        if np.any(actual >= desired):
            self.fail(msg)

    def assertArrayGreater(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.greater(...)) and processes the results like a
        unittest.TestCase method.
        """
        if np.any(actual <= desired):
            self.fail(msg)

    def assertArrayNotAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose), negates and processes the results like
        a unittest.TestCase method.
        """
        if msg is None:
            msg = miss_str(actual, desired, **self.all_close_opts)
        if np.allclose(actual, desired, **self.all_close_opts):
            self.fail(msg)

    def assertArrayNotEqual(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.not_equal(...)) and processes the results like
        a unittest.TestCase method.
        """
        if np.any(actual == desired):
            self.fail(msg)

    def assertArrayNotLess(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.greater_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        if np.any(actual < desired):
            self.fail(msg)

    def assertArrayNotGreater(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.less_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        if np.any(actual > desired):
            self.fail(msg)


def miss_str(x, y, atol=1e-8, rtol=1e-5, equal_nan=True):
    """Returns a string describing the maximum deviation of x and y

    Returns
    -------
    msg: str
        'Should be zero: <maximum devation>
         or: <max dev (rel)> = <tolerance> * <max dev relative to tolerance>'
    """
    shape = np.broadcast(x, y).shape
    thresh = atol + rtol * np.abs(np.broadcast_to(y, shape))
    mismatch = np.abs(x - y)
    mis_frac = mismatch / thresh
    ind = np.unravel_index(np.argmax(mis_frac), mis_frac.shape)
    formatter = 'Should be zero: {:.2g}\nor: {:.2g} = {:.2g} * {:.1f} at {}'

    return formatter.format(np.amax(mismatch), mismatch[ind], thresh[ind],
                            mis_frac[ind], ind)


cmplx = {'b': 0, 'h': 0, 'i': 0, 'l': 0, 'p': 0, 'q': 0,
         'f': 0, 'd': 0, 'g': 0, 'F': 1j, 'D': 1j, 'G': 1j}


def asa(x, y, sctype):
    """Convert x + iy to sctype

    Parameters
    ----------
    x,y: ndarray[float]
        real & imaginary parts, must broadcast
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    imag = cmplx.get(sctype, 0)
    return (x + imag * y).astype(sctype)


@contextlib.contextmanager
def errstate(*args, **kwds):
    """Context manager like np.errstate can also be used as a decorator
    """
    call = kwds.pop('call', None)
    old_errstate = np.geterr()
    try:
        old_errstate = np.seterr(*args, **kwds)
        if call is not None:
            old_call = np.seterrcall(call)
        yield np.geterr()
    finally:
        np.seterr(**old_errstate)
        if call is not None:
            np.seterrcall(old_call)
