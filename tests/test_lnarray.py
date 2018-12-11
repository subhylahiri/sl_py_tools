# -*- coding: utf-8 -*-
"""Test lnarray class
"""
import unittest
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg as la

# =============================================================================
# %% Test python funcs
# =============================================================================


class TestArray(utn.TestCaseNumpy):
    """Testing row, col, scal and transpose"""

    def setUp(self):
        super().setUp()
        self.sctype.append('i')
        self.u = {}
        self.v = {}
        self.w = {}
        self.x = {}
        self.y = {}
        self.z = {}
        for sctype in self.sctype:
            self.u[sctype] = utn.randn_asa((7, 5), sctype)
            self.v[sctype] = utn.randn_asa((5, 2), sctype)
            self.w[sctype] = utn.randn_asa((3, 3), sctype).view(la.lnarray)
            self.x[sctype] = utn.randn_asa((2, 5, 3), sctype).view(la.lnarray)
            self.y[sctype] = utn.randn_asa((3, 5), sctype)
            self.z[sctype] = utn.randn_asa((3,), sctype)

    def test_type(self):
        """Check that functions & operators return the correct type
        """
        v, w, x, y = self.v['d'], self.w['d'], self.x['d'], self.y['d']
        self.assertTrue(isinstance(x @ y, la.lnarray))
        self.assertTrue(isinstance(y @ x, la.lnarray))
        self.assertTrue(isinstance(np.matmul(x, y), np.ndarray))
        self.assertTrue(isinstance(la.solve(w, y), la.lnarray))
        self.assertTrue(isinstance(np.linalg.solve(w, y), np.ndarray))
        self.assertTrue(isinstance(la.lstsq(x, v), la.lnarray))
        self.assertTrue(isinstance(np.linalg.lstsq(x[0], v, rcond=None)[0],
                                   np.ndarray))
        self.assertTrue(isinstance(la.lu(w)[0], la.lnarray))
        self.assertTrue(isinstance(la.lu(v)[0], np.ndarray))
        self.assertTrue(isinstance(la.qr(w)[0], la.lnarray))
        self.assertTrue(isinstance(la.qr(v)[0], np.ndarray))
        self.assertTrue(isinstance(np.linalg.qr(w)[0], np.ndarray))

    def test_shape(self):
        """Check that shape manipulation properties & methods work
        """
        w, x = self.w['D'], self.x['D']
        self.assertEqual(x.t.shape, (2, 3, 5))
        self.assertEqual(x.h.shape, (2, 3, 5))
        self.assertArrayNotAllClose(x.t, x.h)
        self.assertEqual(w.c.shape, (3, 3, 1))
        self.assertEqual(x.c.uc.shape, (2, 5, 3))
        self.assertEqual(w.r.shape, (3, 1, 3))
        self.assertEqual(x.r.ur.shape, (2, 5, 3))
        self.assertEqual(w.s.shape, (3, 3, 1, 1))
        self.assertEqual(x.s.us.shape, (2, 5, 3))
        self.assertEqual(w.expand_dims((1, 3)).shape, (3, 1, 3, 1))
        self.assertEqual((x.s * w).flattish(1, 4).shape, (2, 45, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
