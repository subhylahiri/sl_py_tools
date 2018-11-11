# -*- coding: utf-8 -*-
import unittest as ut
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_cloop as gfc
# =============================================================================


class TestCloop(utn.TestCaseNumpy):
    """Testing norm and rtrue_tdivide"""

    def test_rdiv(self):
        x = np.random.randn(3, 5, 6)
        y = np.random.randn(3, 5, 6)
        z = np.random.randn(3, 4, 6)
        self.assertArrayAlmostEqual(gfc.rtrue_divide(x, y), y / x,
                                    'x \\ y == y / x')
        self.assertArrayNotEqual(gfc.rtrue_divide(x, y), x / y,
                                 'x \\ y != x / y')
        with self.assertRaisesRegex(ValueError,
                                    'operands could not be broadcast'):
            gfc.rtrue_divide(x, z)

    def test_norm(self):
        x = np.random.randn(3, 5, 6)
        y = np.arange(24).reshape((2, 3, 4))
        self.assertEqual(gfc.norm(x).shape, (3, 5))
        self.assertEqual(gfc.norm(x, axis=1).shape, (3, 6))
        self.assertEqual(gfc.norm(x, axis=1, keepdims=True).shape, (3, 1, 6))
        self.assertArrayAlmostEqual(gfc.norm(y), np.sqrt([[14, 126, 366],
                                                          [734, 1230, 1854]]))

    def test_matmul(self):
        x = np.random.randn(2, 3, 5)
        y = np.random.randn(5, 2)
        self.assertArrayAlmostEqual(gfc.matmul(x, y), x @ y)

    def test_rmatmul(self):
        x = np.random.randn(2, 3, 5)
        y = np.random.randn(5, 2)
        with self.assertRaisesRegex(ValueError,
                                    'has a mismatch in its core dimension'):
            gfc.rmatmul(x, y)
        self.assertArrayAlmostEqual(gfc.rmatmul(y, x), x @ y)


# =============================================================================
if __name__ == '__main__':
    ut.main()
