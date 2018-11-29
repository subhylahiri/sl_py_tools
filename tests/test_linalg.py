# -*- coding: utf-8 -*-
"""
"""
import unittest
# import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg as la

errstate = utn.errstate(invalid='raise')
# =============================================================================

# =============================================================================
# %% Test BLAS ufuncs
# =============================================================================


class TestShape(utn.TestCaseNumpy):
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
            self.w[sctype] = utn.randn_asa((2, 3, 3), sctype)
            self.x[sctype] = utn.randn_asa((2, 5, 3), sctype)
            self.y[sctype] = utn.randn_asa((3, 5), sctype)
            self.z[sctype] = utn.randn_asa((3,), sctype)

    def test_shape_fn(self):
        """Check transpose, row, col, scal returns arrays of expected shape
        """
        # shape
        self.assertEqual(la.transpose(self.x['d']).shape, (2, 3, 5))
        self.assertEqual(la.row(self.x['d']).shape, (2, 5, 1, 3))
        self.assertEqual(la.col(self.x['d']).shape, (2, 5, 3, 1))
        self.assertEqual(la.scal(self.x['d']).shape, (2, 5, 3, 1, 1))

    def test_la_fn(self):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct shape
        """
        # matmul
        self.assertEqual(la.matmul(self.x['d'], self.y['d']).shape, (2, 5, 5))
        self.assertEqual(la.matmul(self.x['d'], self.z['d']).shape, (2, 5))
        # rmatmul
        self.assertEqual(la.rmatmul(self.x['d'], self.y['d']).shape, (2, 3, 3))
        self.assertEqual(la.rmatmul(self.y['d'], self.z['d']).shape, (5,))
        # solve
        self.assertEqual(la.solve(self.w['d'], self.y['d']).shape, (2, 3, 5))
        self.assertEqual(la.solve(self.w['d'], self.z['d']).shape, (2, 3))
        # rsolve
        self.assertEqual(la.rsolve(self.x['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.rsolve(self.z['d'], self.w['d']).shape, (2, 3))
        # lstsq
        self.assertEqual(la.lstsq(self.x['d'], self.v['d']).shape, (2, 3, 2))
        self.assertEqual(la.lstsq(self.y['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.lstsq(self.y['d'], self.z['d']).shape, (5,))
        self.assertEqual(la.lstsq(self.z['d'], self.w['d']).shape, (2, 3))
        # rlstsq
        self.assertEqual(la.rlstsq(self.w['d'], self.x['d']).shape, (2, 3, 5))
        self.assertEqual(la.rlstsq(self.u['d'], self.y['d']).shape, (7, 3))
        self.assertEqual(la.rlstsq(self.w['d'], self.z['d']).shape, (2, 3))
        self.assertEqual(la.rlstsq(self.z['d'], self.x['d']).shape, (2, 5))

    def test_div_fn(self):
        """Check matldiv, matrdiv return correct shape
        """
        # solve
        self.assertEqual(la.matldiv(self.w['d'], self.y['d']).shape, (2, 3, 5))
        self.assertEqual(la.matldiv(self.w['d'], self.z['d']).shape, (2, 3))
        # rsolve
        self.assertEqual(la.matrdiv(self.x['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.matrdiv(self.z['d'], self.w['d']).shape, (2, 3))
        # lstsq
        self.assertEqual(la.matldiv(self.x['d'], self.v['d']).shape, (2, 3, 2))
        self.assertEqual(la.matldiv(self.y['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.matldiv(self.y['d'], self.z['d']).shape, (5,))
        self.assertEqual(la.matldiv(self.z['d'], self.w['d']).shape, (2, 3))
        # rlstsq
        self.assertEqual(la.matrdiv(self.w['d'], self.x['d']).shape, (2, 3, 5))
        self.assertEqual(la.matrdiv(self.u['d'], self.y['d']).shape, (7, 3))
        self.assertEqual(la.matrdiv(self.w['d'], self.z['d']).shape, (2, 3))
        self.assertEqual(la.matrdiv(self.z['d'], self.x['d']).shape, (2, 5))

    def test_qr(self):
        """Check that qr returns correct shape in each mode
        """
        q, r = la.qr(self.x['d'], 'reduced')
        self.assertEqual(q.shape + r.shape, (2, 5, 3, 2, 3, 3))
        q, r = la.qr(self.y['d'], 'reduced')
        self.assertEqual(q.shape + r.shape, (3, 3, 3, 5))
        q, r = la.qr(self.x['d'], 'complete')
        self.assertEqual(q.shape + r.shape, (2, 5, 5, 2, 5, 3))
        q, r = la.qr(self.y['d'], 'complete')
        self.assertEqual(q.shape + r.shape, (3, 3, 3, 5))
        r = la.qr(self.x['d'], 'r')
        self.assertEqual(r.shape, (2, 3, 3))
        r = la.qr(self.y['d'], 'r')
        self.assertEqual(r.shape, (3, 5))
        h, tau = la.qr(self.x['d'], 'raw')
        self.assertEqual(h.shape + tau.shape, (2, 3, 5, 2, 3))
        h, tau = la.qr(self.y['d'], 'raw')
        self.assertEqual(h.shape + tau.shape, (5, 3, 3))


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
