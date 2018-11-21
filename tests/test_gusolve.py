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

    @utn.loop_test()
    def test_solve_val(self, sctype):
        a = gfl.solve(self.x[sctype], self.y[sctype])
        with self.subTest(msg='solve'):
            self.assertArrayAllClose(self.x[sctype] @ a, self.y[sctype])
        b = gfl.rsolve(self.v[sctype], self.x[sctype])
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test()
    def test_solvelu_val(self, sctype):
        a0 = gfl.solve(self.x[sctype], self.y[sctype])
        a, xf, p = gfl.solve_lu(self.x[sctype], self.y[sctype])
        with self.subTest('solve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.lu_solve(xf, p, self.y[sctype])
        with self.subTest('solve(lu)'):
            self.assertArrayAllClose(aa, a)
        b = gfl.rlu_solve(self.v[sctype], xf, p)
        with self.subTest('rsolve(lu)'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test()
    def test_rsolvelu_val(self, sctype):
        a0 = gfl.rsolve(self.w[sctype], self.x[sctype])
        a, xf, p = gfl.rsolve_lu(self.w[sctype], self.x[sctype])
        with self.subTest('rsolve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.rlu_solve(self.w[sctype], xf, p)
        with self.subTest('rsolve(rlu)'):
            self.assertArrayAllClose(a, aa)
        b = gfl.lu_solve(xf, p, self.z[sctype])
        with self.subTest('solve(rlu)'):
            self.assertArrayAllClose(self.x[sctype] @ b, self.z[sctype])

    @unittest.expectedFailure
    @utn.errstate(invalid='raise')
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        yy = self.y[sctype] @ transpose(self.y[sctype])
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(yy, self.z[sctype])


class TestLstsq(utn.TestCaseNumpy):
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

    def test_lstsq_shape(self):
        a = gfl.lstsq(self.x['d'], self.y['d'])
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.x['d'], transpose(self.y['d']))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(transpose(self.y['d']), self.x['d'])
        b = gfl.rsolve(transpose(self.y['d']), self.x['d'])
        self.assertEqual(b.shape, (2, 2, 5))

    def test_lstsqqr_shape(self):
        a, xf, p = gfl.lstsq_qrn(self.x['d'], self.y['d'])
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 5, 5))
        self.assertArrayEqual(p.shape, (2, 5))
        b = gfl.qr_lstsq(xf, p, self.z['d'])
        self.assertArrayEqual(b.shape, (3, 2, 5, 4))
        c = gfl.rqr_lstsq(self.w['d'], xf, p)
        self.assertArrayEqual(c.shape, (3, 2, 1, 5))

    def test_rlstsqqr_shape(self):
        a, xf, p = gfl.rlstsq_qrn(self.w['d'], self.x['d'])
        self.assertArrayEqual(a.shape, (3, 2, 1, 5))
        self.assertArrayEqual(xf.shape, (3, 2, 5, 5))
        self.assertArrayEqual(p.shape, (3, 2, 5))
        b = gfl.rqr_lstsq(self.v['d'], xf, p)
        self.assertArrayEqual(b.shape, (3, 2, 4, 5))

    @utn.loop_test()
    def test_lstsq_val(self, sctype):
        a = gfl.lstsq(self.x[sctype], self.y[sctype])
        with self.subTest(msg='lstsq'):
            self.assertArrayAllClose(self.x[sctype] @ a, self.y[sctype])
        b = gfl.rsolve(self.v[sctype], self.x[sctype])
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test()
    def test_lstsqqr_val(self, sctype):
        a0 = gfl.lstsq(self.x[sctype], self.y[sctype])
        a, xf, p = gfl.lstsq_qrn(self.x[sctype], self.y[sctype])
        with self.subTest('solve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.qr_lstsq(xf, p, self.y[sctype])
        with self.subTest('lstsq(lu)'):
            self.assertArrayAllClose(aa, a)
        b = gfl.rqr_lstsq(self.v[sctype], xf, p)
        with self.subTest('rsolve(lu)'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test()
    def test_rlstsqqr_val(self, sctype):
        a0 = gfl.rsolve(self.w[sctype], self.x[sctype])
        a, xf, p = gfl.rlstsq_qrn(self.w[sctype], self.x[sctype])
        with self.subTest('rsolve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.rqr_lstsq(self.w[sctype], xf, p)
        with self.subTest('rsolve(rlu)'):
            self.assertArrayAllClose(a, aa)
        b = gfl.qr_lstsq(xf, p, self.z[sctype])
        with self.subTest('lstsq(rlu)'):
            self.assertArrayAllClose(self.x[sctype] @ b, self.z[sctype])

    @unittest.expectedFailure
    @utn.errstate(invalid='raise')
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        yy = self.y[sctype] @ transpose(self.y[sctype])
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq(yy, self.z[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
