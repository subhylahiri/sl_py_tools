# -*- coding: utf-8 -*-
import unittest as ut
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_cloop as gfc
import sl_py_tools.numpy_tricks.linalg._gufuncs_blas as gfb
import sl_py_tools.numpy_tricks.linalg._gufuncs_lapack as gfl
from sl_py_tools.numpy_tricks.linalg import transpose

errstate = utn.errstate(invalid='raise')
# =============================================================================


class TestBlas(utn.TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""
    def setUp(self):
        self.skipTest('speed')
        super().setUp()
        self.gf = gfb
        self.nulp = 10
        self.sctype = ['i', 'f', 'd', 'F', 'D']
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.n = {}
        for sctype in self.sctype:
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

    def test_norm_shape(self):
        # shape
        self.assertEqual(self.gf.norm(self.x['d']).shape, (2, 3))
        self.assertEqual(self.gf.norm(self.x['d'], axis=1).shape, (2, 5))
        self.assertEqual(self.gf.norm(self.x['d'], keepdims=True).shape,
                         (2, 3, 1))

    @utn.TestCaseNumpy.loop(msg='norm val', attr_inds=slice(1, None))
    def test_norm_val(self, sctype):
        # value
        n = self.gf.norm(self.w[sctype])
        self.assertArrayAllClose(n, self.n[sctype])

    def test_matmul_shape(self):
        # shape
        with self.assertRaisesRegex(ValueError,
                                    'has a mismatch in its core dimension'):
            self.gf.matmul(self.y['d'], self.x['d'])

    @utn.TestCaseNumpy.loop(msg='matmul val')
    def test_matmul_val(self, sctype):
        # value
        z = self.gf.matmul(self.x[sctype], self.y[sctype])
        self.assertArrayAllClose(z, self.z[sctype])

    def test_rmatmul_shape(self):
        with self.assertRaisesRegex(ValueError,
                                    'has a mismatch in its core dimension'):
            self.gf.rmatmul(self.x['d'], self.y['d'])

    @utn.TestCaseNumpy.loop(msg='rmatmul val')
    def test_rmatmul_val(self, sctype):
        z = self.gf.rmatmul(self.y[sctype], self.x[sctype])
        self.assertArrayAllClose(z, self.z[sctype])

class TestCloop(TestBlas):
    """Testing norm, matmul, rmatmul and rtrue_tdivide
    """
    def setUp(self):
        self.skipTest('speed')
        super().setUp()
        self.gf = gfc

    def test_rdiv_shape(self):
        # shape
        with self.assertRaisesRegex(ValueError,
                                    'operands could not be broadcast'):
            self.gf.rtrue_divide(self.x['d'], self.z['d'])

    @utn.TestCaseNumpy.loop(msg="x \\ y == y / x. ", attr_inds=slice(1, None))
    def test_rdiv_val(self, sctype):
        # value
        x = self.x[sctype]
        y = self.y[sctype].T[:, None]
        z = self.gf.rtrue_divide(x, y)
        zz = y / x
        self.assertArrayAllClose(z, zz)
        self.assertArrayNotAllClose(z, x / y, msg='x \\ y != x / y')


class TestQR(utn.TestCaseNumpy):
    """Testing gufuncs_blas.qr
    """
    def setUp(self):
        super().setUp()
        self.sctype = ['f', 'd', 'F', 'D']
        self.wide = {}
        self.tall = {}
        self.wide = {}
        self.id_small = {}
        self.id_big = {}
        for sctype in self.sctype:
            x = np.random.randn(120, 10, 5, 16)
            y = np.random.randn(120, 10, 16, 5)
            self.wide[sctype] = utn.asa(x, y.swapaxes(-2, -1), sctype)
            self.tall[sctype] = utn.asa(x.swapaxes(-2, -1), y, sctype)
            self.id_small[sctype] = np.eye(5, dtype=sctype)
            self.id_big[sctype] = np.eye(16, dtype=sctype)

    @ut.skip('speed')
    @errstate
    def test_qr_shape(self):
        with self.subTest(msg='wide'):
            q, r = gfl.qr_m(self.wide['d'])
            self.assertEqual(q.shape, (120, 10, 5, 5))
            self.assertEqual(r.shape, (120, 10, 5, 16))
        with self.subTest(msg='tall'):
            q, r = gfl.qr_n(self.tall['d'])
            self.assertEqual(q.shape, (120, 10, 16, 5))
            self.assertEqual(r.shape, (120, 10, 5, 5))
            with self.assertRaisesRegex(FloatingPointError,
                                        'invalid value encountered in qr_n'):
                gfl.qr_n(self.wide['d'])
        with self.subTest(msg='complete'):
            q, r = gfl.qr_m(self.tall['d'])
            self.assertEqual(q.shape, (120, 10, 16, 16))
            self.assertEqual(r.shape, (120, 10, 16, 5))
        with self.subTest(msg='raw'):
            h, tau = gfl.qr_rawn(self.tall['d'])
            self.assertEqual(h.shape, (120, 10, 5, 16))
            self.assertEqual(tau.shape, (120, 10, 5))

    @ut.skip('speed')
    @utn.TestCaseNumpy.loop(msg='wide')
    def test_qr_wide(self, sctype):
        q, r = gfl.qr_m(self.wide[sctype])
        wide = q @ r
        eye = transpose(q.conj()) @ q
        eyet = q @ transpose(q.conj())
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(wide, self.wide[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small[sctype], eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_small[sctype], eyet)

    @ut.skip('speed')
    @utn.TestCaseNumpy.loop(msg='tall')
    def test_qr_tall(self, sctype):
        q, r = gfl.qr_n(self.tall[sctype])
        tall = q @ r
        eye = transpose(q.conj()) @ q
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small[sctype], eye)

    @ut.skip('speed')
    @utn.TestCaseNumpy.loop(msg='complete')
    def test_qr_complete(self, sctype):
        q, r = gfl.qr_m(self.tall[sctype])
        tall = q @ r
        eye = transpose(q.conj()) @ q
        eyet = q @ transpose(q.conj())
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_big[sctype], eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_big[sctype], eyet)

    @ut.skip('speed')
    @utn.TestCaseNumpy.loop()
    def test_qr_r(self, sctype):
        with self.subTest(msg='r_m'):
            r = gfl.qr_rm(self.wide[sctype])
            rr = gfl.qr_m(self.wide[sctype])[1]
            self.assertArrayAllClose(r, rr)
        with self.subTest(msg='r_n'):
            r = gfl.qr_rn(self.tall[sctype])
            rr = gfl.qr_n(self.tall[sctype])[1]
            self.assertArrayAllClose(r, rr)

    @utn.TestCaseNumpy.loop()
    def test_qr_raw(self, sctype):
        rr = gfl.qr_m(self.wide[sctype])[1]
        ht, tau = gfl.qr_rawm(self.wide[sctype])
        h = transpose(ht)
        v = np.tril(h[..., :5], -1)
        v[..., np.diag_indices(5)] = 1
        r = np.triu(h)
        with self.subTest(msg='raw_m'):
            self.assertArrayAllClose(r, rr)
        for k in range(1, 6):
            vr = v[..., None, :, -k] @ r
            r -= tau[..., None, None, -k] * v[..., -k, None] * vr
        with self.subTest(msg='h_m'):
           self.assertArrayAllClose(r, self.wide[sctype])

        rr = gfl.qr_n(self.tall[sctype])[1]
        ht, tau = gfl.qr_rawn(self.tall[sctype])
        h = transpose(ht)
        v = np.tril(h, -1)
        v[..., np.diag_indices(5)] = 1
        r = np.triu(h)
        with self.subTest(msg='raw_n'):
            self.assertArrayAllClose(r[..., :5, :], rr)
        for k in range(1, 6):
            vr = v[..., None, :, -k] @ r
            r -= tau[..., None, None, -k] * v[..., -k, None] * vr
        with self.subTest(msg='h_n'):
            self.assertArrayAllClose(r, self.tall[sctype])


# =============================================================================
if __name__ == '__main__':
    ut.main()
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
