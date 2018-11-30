# -*- coding: utf-8 -*-
"""Test solve & lstsq families of gufuncs
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


class TestSolveShape(utn.TestCaseNumpy):
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
