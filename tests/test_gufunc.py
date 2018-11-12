# -*- coding: utf-8 -*-
import unittest as ut
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_cloop as gfc
import sl_py_tools.numpy_tricks.linalg._gufuncs_blas as gfb
# =============================================================================


class TestBlas(utn.TestCaseNumpy):
    """Testing norm and rtrue_tdivide"""
    def setUp(self):
        self.gf = gfb
        self.sctypes = ['i', 'f', 'd', 'F', 'D']
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.n = {}
        for sctype in self.sctypes:
            self.x[sctype] = utn.asa(np.random.randn(2, 3, 5),
                                     np.random.randn(2, 3, 5), sctype)
            self.y[sctype] = utn.asa(np.random.randn(5, 2),
                                     np.random.randn(5, 2), sctype)
            self.z[sctype] = self.x[sctype] @ self.y[sctype]
            self.w[sctype] = utn.asa(np.arange(24).reshape((2, 3, 4)),
                                     np.arange(8).reshape((2, 1, 4)), sctype)
            nsq = utn.asa(np.array([[14, 126, 366], [734, 1230, 1854]]),
                          np.array([[14], [126]]), sctype)
            self.n[sctype] = np.sqrt(nsq.real + nsq.imag)

    def test_norm(self):
        # shape
        self.assertEqual(self.gf.norm(self.x['d']).shape, (2, 3))
        self.assertEqual(self.gf.norm(self.x['d'], axis=1).shape, (2, 5))
        self.assertEqual(self.gf.norm(self.x['d'], keepdims=True).shape,
                         (2, 3, 1))
        # value
        for sctype in self.sctypes[1:]:
            with self.subTest(sctype=sctype):
                n = self.gf.norm(self.w[sctype])
                msg = '||w||. ' + utn.mismatch_str(n, self.n[sctype])
                self.assertArrayAlmostEqual(n, self.n[sctype], msg, nulp=3)

    def test_matmul(self):
        # value
        for sctype in self.sctypes:
            with self.subTest(sctype=sctype):
                z = self.gf.matmul(self.x[sctype], self.y[sctype])
                msg = 'x @ y. ' + utn.mismatch_str(z, self.z[sctype])
                self.assertArrayAlmostEqual(z, self.z[sctype], msg=msg, nulp=5)

    def test_rmatmul(self):
        # shape
        with self.assertRaisesRegex(ValueError,
                                    'has a mismatch in its core dimension'):
            self.gf.rmatmul(self.x['d'], self.y['d'])
        # value
        for sctype in self.sctypes:
            with self.subTest(sctype=sctype):
                z = self.gf.rmatmul(self.y[sctype], self.x[sctype])
                msg = 'y r@ x. ' + utn.mismatch_str(z, self.z[sctype])
                self.assertArrayAlmostEqual(z, self.z[sctype], msg=msg, nulp=5)


class TestCloop(TestBlas):
    """Testing norm and rtrue_tdivide"""
    def setUp(self):
        super().setUp()
        self.gf = gfc

    def test_rdiv(self):
        # shape
        with self.assertRaisesRegex(ValueError,
                                    'operands could not be broadcast'):
            self.gf.rtrue_divide(self.x['d'], self.z['d'])
        # value
        for sctype in self.sctypes[1:]:
            with self.subTest(sctype=sctype):
                x = self.x[sctype]
                y = self.y[sctype].T[:, None]
                z = self.gf.rtrue_divide(x, y)
                zz = y / x
                msg = "x \\ y == y / x. " + utn.mismatch_str(z, zz)
                self.assertArrayAlmostEqual(z, zz, msg=msg, nulp=3)
                self.assertArrayNotEqual(z, x / y, msg='x \\ y != x / y')


# =============================================================================
if __name__ == '__main__':
    ut.main()
