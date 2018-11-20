# -*- coding: utf-8 -*-
import unittest
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_lapack as gfl
from sl_py_tools.numpy_tricks.linalg import transpose

# =============================================================================


class TestSolve(utn.TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""
    def setUp(self):
        super().setUp()
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.v = {}
        for sctype in self.sctype:
            self.x[sctype] = utn.asa(np.random.randn(2, 5, 5),
                                     np.random.randn(2, 5, 5), sctype)
            self.y[sctype] = utn.asa(np.random.randn(5, 2),
                                     np.random.randn(5, 2), sctype)
            self.z[sctype] = utn.asa(np.random.randn(3, 1, 5, 4),
                                     np.random.randn(3, 1, 5, 4), sctype)
            self.w[sctype] = utn.asa(np.random.randn(3, 1, 1, 5),
                                     np.random.randn(3, 1, 1, 5), sctype)
            self.v[sctype] = utn.asa(np.random.randn(4, 5),
                                     np.random.randn(4, 5), sctype)

    def test_solve_shape(self):
        a = gfl.solve(self.x['d'], self.y['d'])
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.x['d'], transpose(self.y['d']))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(transpose(self.y['d']), self.x['d'])
        b = gfl.rsolve(transpose(self.y['d']), self.x['d'])
        self.assertEqual(b.shape, (2, 2, 5))

    def test_solvelu_shape(self):
        a, xf, p = gfl.solve_lu(self.x['d'], self.y['d'])
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 5, 5))
        self.assertArrayEqual(p.shape, (2, 5))
        b = gfl.lu_solve(xf, p, self.z['d'])
        self.assertArrayEqual(b.shape, (3, 2, 5, 4))
        c = gfl.rlu_solve(self.w['d'], xf, p)
        self.assertArrayEqual(c.shape, (3, 2, 1, 5))

    def test_rsolvelu_shape(self):
        a, xf, p = gfl.rsolve_lu(self.w['d'], self.x['d'])
        self.assertArrayEqual(a.shape, (3, 2, 1, 5))
        self.assertArrayEqual(xf.shape, (3, 2, 5, 5))
        self.assertArrayEqual(p.shape, (3, 2, 5))
        b = gfl.rlu_solve(self.v['d'], xf, p)
        self.assertArrayEqual(b.shape, (3, 2, 4, 5))

    @utn.TestCaseNumpy.loop()
    def test_solve_val(self, sctype):
        a = gfl.solve(self.x[sctype], self.y[sctype])
        with self.subTest(msg='solve'):
            self.assertArrayAllClose(self.x[sctype] @ a, self.y[sctype])
        b = gfl.rsolve(self.v[sctype], self.x[sctype])
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(self.v[sctype], b @ self.x[sctype])

    @utn.TestCaseNumpy.loop(attr_inds=1)
    def test_solvelu_val(self, sctype):
        a, xf, p = gfl.solve_lu(self.x[sctype], self.y[sctype])
        aa = gfl.lu_solve(xf, p, self.y[sctype])
        with self.subTest('solve(lu)'):
            self.assertArrayAllClose(a, aa)
        b = gfl.rlu_solve(self.v[sctype], xf, p)
        with self.subTest('rsolve(lu)'):
            self.assertArrayAllClose(self.v[sctype], b @ self.x[sctype])

    @utn.TestCaseNumpy.loop(attr_inds=1)
    def test_rsolvelu_val(self, sctype):
        a, xf, p = gfl.rsolve_lu(self.w['d'], self.x['d'])
        aa = gfl.rlu_solve(self.w['d'], xf, p)
        with self.subTest('rsolve(rlu)'):
            self.assertArrayAllClose(a, aa)
        b = gfl.lu_solve(xf, p, self.z[sctype])
        with self.subTest('solve(rlu)'):
            self.assertArrayAllClose(self.x[sctype] @ b, self.z[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main()
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
