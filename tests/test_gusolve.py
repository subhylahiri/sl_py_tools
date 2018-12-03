# -*- coding: utf-8 -*-
"""Test solve & lstsq families of gufuncs
"""
import unittest
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_lu_solve as gfl
from sl_py_tools.numpy_tricks.linalg import transpose

errstate = utn.errstate(invalid='raise')
# =============================================================================
# %% Test LU
# =============================================================================


class TestLU(utn.TestCaseNumpy):
    """Testing LU decomposition"""

    def setUp(self):
        super().setUp()
        self.square = {}
        self.tall = {}
        self.wide = {}
        for sctype in self.sctype:
            self.square[sctype] = utn.randn_asa((2, 5, 5), sctype)
            self.wide[sctype] = utn.randn_asa((3, 1, 3, 6), sctype)
            self.tall[sctype] = utn.randn_asa((5, 2), sctype)

    def test_lu_basic_shape(self):
        """Test shape of basic LU"""
        sq_l, sq_u, sq_ip = gfl.lu_m(self.square['d'])
        with self.subTest(msg="square"):
            self.assertEqual(sq_l.shape, (2, 5, 5))
            self.assertEqual(sq_u.shape, (2, 5, 5))
            self.assertEqual(sq_ip.shape, (2, 5))
        wd_l, wd_u, wd_ip = gfl.lu_m(self.wide['d'])
        with self.subTest(msg="wide"):
            self.assertEqual(wd_l.shape, (3, 1, 3, 3))
            self.assertEqual(wd_u.shape, (3, 1, 3, 6))
            self.assertEqual(wd_ip.shape, (3, 1, 3))
        tl_l, tl_u, tl_ip = gfl.lu_n(self.tall['d'])
        with self.subTest(msg="tall"):
            self.assertEqual(tl_l.shape, (5, 2))
            self.assertEqual(tl_u.shape, (2, 2))
            self.assertEqual(tl_ip.shape, (2,))

    @unittest.skip("failing")
    def test_lu_raw_shape(self):
        """Test shape of raw LU"""
        sq_f, sq_ip = gfl.lu_rawm(self.square['d'])
        with self.subTest(msg="square"):
            self.assertEqual(sq_f.shape, (2, 5, 5))
            self.assertEqual(sq_ip.shape, (2, 5))
        wd_f, wd_ip = gfl.lu_rawm(self.wide['d'])
        with self.subTest(msg="wide"):
            self.assertEqual(wd_f.shape, (3, 1, 3, 6))
            self.assertEqual(wd_ip.shape, (3, 1, 3))
        tl_f, tl_ip = gfl.lu_rawn(self.tall['d'])
        with self.subTest(msg="tall"):
            self.assertEqual(tl_f.shape, (5, 2))
            self.assertEqual(tl_ip.shape, (2,))

    @unittest.expectedFailure
    @utn.loop_test()
    def test_lu_basic_val(self, sctype):
        """Test values of basic LU"""
        sq_l, sq_u, sq_ip = gfl.lu_m(self.square[sctype])
        inds = (...,) + np.diag_indices(5, 2)
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_l @ sq_u, self.square[sctype])
            self.assertArrayAllClose(sq_l[inds], 1.)
        wd_l, wd_u, wd_ip = gfl.lu_m(self.wide[sctype])
        inds = (...,) + np.diag_indices(3, 2)
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_l @ wd_u, self.wide[sctype])
            self.assertArrayAllClose(wd_l[inds], 1.)
        tl_l, tl_u, tl_ip = gfl.lu_n(self.tall[sctype])
        inds = (...,) + np.diag_indices(2, 2)
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_l @ tl_u, self.tall[sctype])
            self.assertArrayAllClose(tl_l[inds], 1.)

    @unittest.skip("failing")
    @utn.loop_test()
    def test_lu_raw_val(self, sctype):
        """Test values of raw LU"""
        sq_l, sq_u, sq_ip0 = gfl.lu_m(self.square[sctype])
        sq_f, sq_ip = gfl.lu_rawm(self.square[sctype])
        linds = (...,) + np.tril_indices(5, -1)
        uinds = (...,) + np.triu_indices(5, 0)
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_f[linds], sq_l[linds])
            self.assertArrayAllClose(sq_f[uinds], sq_u[uinds])
            self.assertEqual(sq_ip, sq_ip0)
        wd_l, wd_u, wd_ip0 = gfl.lu_m(self.wide[sctype])
        wd_f, wd_ip = gfl.lu_rawm(self.wide[sctype])
        linds = (...,) + np.tril_indices(3, -1, 6)
        uinds = (...,) + np.triu_indices(3, 0, 6)
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_f[linds], wd_l[linds])
            self.assertArrayAllClose(wd_f[uinds], wd_u[uinds])
            self.assertEqual(wd_ip, wd_ip0)
        tl_l, tl_u, tl_ip0 = gfl.lu_n(self.tall[sctype])
        tl_f, tl_ip = gfl.lu_rawn(self.tall[sctype])
        linds = (...,) + np.tril_indices(5, -1, 2)
        uinds = (...,) + np.triu_indices(5, 0, 2)
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_f[linds], tl_l[linds])
            self.assertArrayAllClose(tl_f[uinds], tl_u[uinds])
            self.assertEqual(tl_ip, tl_ip0)


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


class TestSolveShape(TestSolve):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    def test_solve_shape(self):
        """Check if solve, rsolve return arrays with the expected shape
        """
        a = gfl.solve(self.x['d'], self.y['d'])
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.x['d'], self.yt['d'])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.yt['d'], self.x['d'])
        b = gfl.rsolve(self.yt['d'], self.x['d'])
        self.assertEqual(b.shape, (2, 2, 5))

    def test_solvelu_shape(self):
        """Check if solve_lu, lu_solve return arrays with the expected shape
        """
        a, xf, p = gfl.solve_lu(self.x['d'], self.y['d'])
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 5, 5))
        self.assertArrayEqual(p.shape, (2, 5))
        b = gfl.lu_solve(xf, p, self.z['d'])
        self.assertArrayEqual(b.shape, (3, 2, 5, 4))
        c = gfl.rlu_solve(self.w['d'], xf, p)
        self.assertArrayEqual(c.shape, (3, 2, 1, 5))

    def test_rsolvelu_shape(self):
        """Check if rsolve_lu, rlu_solve return arrays with the expected shape
        """
        a, xf, p = gfl.rsolve_lu(self.w['d'], self.x['d'])
        self.assertArrayEqual(a.shape, (3, 2, 1, 5))
        self.assertArrayEqual(xf.shape, (3, 2, 5, 5))
        self.assertArrayEqual(p.shape, (3, 2, 5))
        b = gfl.rlu_solve(self.v['d'], xf, p)
        self.assertArrayEqual(b.shape, (3, 2, 4, 5))


class TestSolveVal(TestSolve):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    @utn.loop_test()
    def test_solve_val(self, sctype):
        """Check if solve, rsolve return the expected values
        """
        a = gfl.solve(self.x[sctype], self.y[sctype])
        with self.subTest(msg='solve'):
            self.assertArrayAllClose(self.x[sctype] @ a, self.y[sctype])
        b = gfl.rsolve(self.v[sctype], self.x[sctype])
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test()
    def test_solvelu_val(self, sctype):
        """Check if solve_lu, lu_solve, rlu_solve return the expected values
        """
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
        """Check if rsolve_lu, lu_solve, rlu_solve return the expected values
        """
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
        """Check if solve raises an exception when divisor is rank deficient
        """
        yy = self.y[sctype] @ self.yt[sctype]
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(yy, self.z[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
