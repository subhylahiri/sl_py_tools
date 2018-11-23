# -*- coding: utf-8 -*-
"""
"""
import unittest
# import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_lapack as gfl
from sl_py_tools.numpy_tricks.linalg import transpose

errstate = utn.errstate(invalid='raise')
# =============================================================================
# %% Test solve
# =============================================================================

class TestSolve(utn.TestCaseNumpy):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""
    def setUp(self):
        super().setUp()
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.v = {}
        self.yt = {}
        for sctype in self.sctype:
            self.x[sctype] = utn.randn_asa((2, 5, 5), sctype)
            self.y[sctype] = utn.randn_asa((5, 2), sctype)
            self.z[sctype] = utn.randn_asa((3, 1, 5, 4), sctype)
            self.w[sctype] = utn.randn_asa((3, 1, 1, 5), sctype)
            self.v[sctype] = utn.randn_asa((4, 5), sctype)
            self.yt[sctype] = transpose(self.y[sctype])

    def test_solve_shape(self):
        a = gfl.solve(self.x['d'], self.y['d'])
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.x['d'], self.yt['d'])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.yt['d'], self.x['d'])
        b = gfl.rsolve(self.yt['d'], self.x['d'])
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
            self.assertArrayAllClose(aa, a0)
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
            self.assertArrayAllClose(aa, a0)
        b = gfl.lu_solve(xf, p, self.z[sctype])
        with self.subTest('solve(rlu)'):
            self.assertArrayAllClose(self.x[sctype] @ b, self.z[sctype])

    @unittest.expectedFailure
    @utn.errstate(invalid='raise')
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        yy = self.y[sctype] @ self.yt[sctype]
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(yy, self.z[sctype])


# =============================================================================
# %% Test lstsq
# =============================================================================


class TestLstsq(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""
    def setUp(self):
        super().setUp()
        self.u = {}
        self.v = {}
        self.w = {}
        self.x = {}
        self.y = {}
        self.z = {}
        self.wt = {}
        self.xt = {}
        self.yt = {}
        self.zt = {}
        for sctype in self.sctype:
            self.u[sctype] = utn.randn_asa((5, 4), sctype)
            self.v[sctype] = utn.randn_asa((4, 5), sctype)
            self.w[sctype] = utn.randn_asa((3, 1, 1, 5), sctype)
            self.x[sctype] = utn.randn_asa((2, 8, 5), sctype)
            self.y[sctype] = utn.randn_asa((8, 2), sctype)
            self.z[sctype] = utn.randn_asa((3, 1, 8, 4), sctype)
            self.wt[sctype] = transpose(self.w[sctype])
            self.xt[sctype] = transpose(self.x[sctype])
            self.yt[sctype] = transpose(self.y[sctype])
            self.zt[sctype] = transpose(self.z[sctype])

    @errstate
    def test_lstsq_shape(self):
        # overconstrained
        a = gfl.lstsq(self.x['d'], self.y['d'])
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.x['d'], self.yt['d'])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.yt['d'], self.xt['d'])
        # underconstrained
        b = gfl.rlstsq(self.yt['d'], self.xt['d'])
        self.assertEqual(b.shape, (2, 2, 5))

    @errstate
    def test_lstsqqr_shape(self):
        # overconstrained
        a, xf, tau = gfl.lstsq_qrn(self.x['d'], self.y['d'])
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 8, 5))
        self.assertArrayEqual(tau.shape, (2, 5))
        # overconstrained
        b = gfl.qr_lstsq(xf, tau, self.z['d'])
        self.assertArrayEqual(b.shape, (3, 2, 5, 4))
        # underconstrained
        c = gfl.rqr_lstsq(self.w['d'], xf, tau)
        self.assertArrayEqual(c.shape, (3, 2, 1, 8))

    @errstate
    def test_rlstsqqr_shape(self):
        # underconstrained
        a, xf, tau = gfl.rlstsq_qrm(self.w['d'], self.x['d'])
        self.assertArrayEqual(a.shape, (3, 2, 1, 8))
        self.assertArrayEqual(xf.shape, (3, 2, 8, 5))
        self.assertArrayEqual(tau.shape, (3, 2, 5))
        # underconstrained
        b = gfl.rqr_lstsq(self.v['d'], xf, tau)
        self.assertArrayEqual(b.shape, (3, 2, 4, 8))

    @errstate
    @utn.loop_test(attr_inds=1)
    def test_lstsq_val(self, sctype):
        # overconstrained
        a = gfl.lstsq(self.x[sctype], self.y[sctype])
        with self.subTest(msg='lstsq(over)'):
            self.assertArrayAllClose(self.xt[sctype] @ self.x[sctype] @ a,
                                     self.xt[sctype] @ self.y[sctype])
        # underconstrained
        b = gfl.rlstsq(self.v[sctype], self.x[sctype])
        with self.subTest(msg='rlstsq(under)'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @errstate
    @utn.loop_test(attr_inds=1)
    def test_lstsqqr_val(self, sctype):
        # overconstrained
        a0 = gfl.lstsq(self.x[sctype], self.y[sctype])
        a, xf, tau = gfl.lstsq_qrn(self.x[sctype], self.y[sctype])
        with self.subTest('lstsq(qr,over)'):
            self.assertArrayAllClose(a, a0)
        # overconstrained
        aa = gfl.qr_lstsq(xf, tau, self.y[sctype])
        with self.subTest('(qr)lstsq(over)'):
            self.assertArrayAllClose(aa, a0)
        # underconstrained
        b = gfl.rqr_lstsq(self.v[sctype], xf, tau)
        with self.subTest('(rqr)lstsq(under)'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @errstate
    @utn.loop_test(attr_inds=1)
    def test_rlstsqqr_val(self, sctype):
        # underconstrained
        a0 = gfl.rlstsq(self.w[sctype], self.x[sctype])
        # underconstrained
        a, xf, tau = gfl.rlstsq_qrm(self.w[sctype], self.x[sctype])
        with self.subTest('rlstsq(qr,under)'):
            self.assertArrayAllClose(a, a0)
        # underconstrained
        aa = gfl.rqr_lstsq(self.w[sctype], xf, tau)
        with self.subTest('(rqr)rlstsq(under)'):
            self.assertArrayAllClose(aa, a0)
        # overconstrained
        b = gfl.qr_lstsq(xf, tau, self.z[sctype])
        with self.subTest('(qr)rlstsq(over)'):
            self.assertArrayAllClose(self.xt[sctype] @ self.x[sctype] @ b,
                                     self.xt[sctype] @ self.z[sctype])

    @errstate
    @utn.loop_test(attr_inds=1)
    def test_lstsqqrt_val(self, sctype):
        # underconstrained
        a0 = gfl.lstsq(self.xt[sctype], self.u[sctype])
        # underconstrained
        a, xf, tau = gfl.lstsq_qrm(self.xt[sctype], self.u[sctype])
        with self.subTest('lstsq(qr,under)'):
            self.assertArrayAllClose(a, a0)
        # underconstrained
        aa = gfl.qr_lstsq(xf, tau, self.u[sctype])
        with self.subTest('(qr)lstsq(under)'):
            self.assertArrayAllClose(aa, a0)
        # overconstrained
        b = gfl.rqr_lstsq(self.zt[sctype], xf, tau)
        with self.subTest('(rqr)lstsq(over)'):
            self.assertArrayAllClose(b @ self.xt[sctype] @ self.x[sctype],
                                     self.zt[sctype] @ self.x[sctype])

    @errstate
    @utn.loop_test(attr_inds=1)
    def test_rlstsqqrt_val(self, sctype):
        # overconstrained
        a0 = gfl.rlstsq(self.yt[sctype], self.xt[sctype])
        # overconstrained
        a, xf, tau = gfl.rlstsq_qrn(self.yt[sctype], self.xt[sctype])
        with self.subTest('rlstsq(qr,over)'):
            self.assertArrayAllClose(a, a0)
        # overconstrained
        aa = gfl.rqr_lstsq(self.yt[sctype], xf, tau)
        with self.subTest('(rqr)rlstsq(over)'):
            self.assertArrayAllClose(aa, a0)
        # underconstrained
        b = gfl.qr_lstsq(xf, tau, self.wt[sctype])
        with self.subTest('(qr)rlstsq(under)'):
            self.assertArrayAllClose(self.xt[sctype] @ b, self.wt[sctype])

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        yy = self.y[sctype] @ self.yt[sctype]
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq(yy, self.z[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
