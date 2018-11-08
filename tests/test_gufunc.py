# -*- coding: utf-8 -*-
import unittest as ut
import numpy as np
import sl_py_tools.numpy_tricks.linalg.gufuncs as gf
import sl_py_tools.numpy_tricks.linalg._lnarray as la
# =============================================================================


class TestCaseNumpy(ut.TestCase):
    """Test case mith method for comparing arrays"""

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayEqual)
        self.addTypeEqualityFunc(la.lnarray, self.assertArrayEqual)

    def assertArrayEqual(self, x, y, msg=None):
        """Assert that two arrays are equal
        """
        try:
            np.testing.assert_array_almost_equal_nulp(x, y)
        except AssertionError:
            raise self.failureException(msg)


class TestCloop(TestCaseNumpy):
    """Testing norm and rtrue_tdivide"""

    def test_rdiv(self):
        x = np.random.randn(3, 5, 6)
        y = np.random.randn(3, 5, 6)
        z = np.random.randn(3, 4, 6)
        self.assertEqual(gf.rtrue_divide(x, y), y / x, 'x \ y == y / x')
#        self.assertNotEqual(gf.rtrue_divide(x, y), x / y, 'x \ y != x / y')
        with self.assertRaisesRegex(ValueError,
                                    'operands could not be broadcast'):
            gf.rtrue_divide(x, z)


# =============================================================================
if __name__ == '__main__':
    ut.main()
