# -*- coding: utf-8 -*-
"""
"""
import unittest
import numpy as np
import unittest_numpy as utn
import sl_py_tools.numpy_tricks.linalg._gufuncs_cloop as gfc
import sl_py_tools.numpy_tricks.linalg._gufuncs_blas as gfb
import sl_py_tools.numpy_tricks.linalg._gufuncs_lapack as gfl
from sl_py_tools.numpy_tricks.linalg import transpose

errstate = utn.errstate(invalid='raise')
# =============================================================================

# =============================================================================
# %% Test BLAS ufuncs
# =============================================================================


class TestBlas(utn.TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""
    def setUp(self):
        super().setUp()
        self.gf = gfb
        self.sctype.append('i')
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.n = {}
        for sctype in self.sctype:
            self.x[sctype] = utn.randn_asa((2, 3, 5), sctype)
            self.y[sctype] = utn.randn_asa((5, 2), sctype)
            self.z[sctype] = self.x[sctype] @ self.y[sctype]
            self.w[sctype] = utn.asa(np.arange(24).reshape((2, 3, 4)),
                                     np.arange(8).reshape((2, 1, 4)), sctype)
            nsq = utn.asa(np.array([[14, 126, 366], [734, 1230, 1854]]),
                          np.array([[14], [126]]), sctype)
            self.n[sctype] = np.sqrt(nsq.real + nsq.imag)

    def test_norm_shape(self):
        """Check that norm returns arrays with the expected shape
        """
        # shape
        self.assertEqual(self.gf.norm(self.x['d']).shape, (2, 3))
        self.assertEqual(self.gf.norm(self.x['d'], axis=1).shape, (2, 5))
        self.assertEqual(self.gf.norm(self.x['d'], keepdims=True).shape,
                         (2, 3, 1))

    @utn.loop_test(msg='norm val', attr_inds=slice(1, None))
    def test_norm_val(self, sctype):
        """Check that norm returns the expected values
        """
        # value
        n = self.gf.norm(self.w[sctype])
        self.assertArrayAllClose(n, self.n[sctype])

    def test_matmul_shape(self):
        """Check that matmul returns arrays with the expected shape
        """
        # shape
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.y['d'], self.x['d'])

    @utn.loop_test(msg='matmul val')
    def test_matmul_val(self, sctype):
        """Check that matmul returns the expected values
        """
        # value
        z = self.gf.matmul(self.x[sctype], self.y[sctype])
        self.assertArrayAllClose(z, self.z[sctype])

    def test_rmatmul_shape(self):
        """Check that rmatmul returns arrays with the expected shape
        """
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.x['d'], self.y['d'])

    @utn.loop_test(msg='rmatmul val')
    def test_rmatmul_val(self, sctype):
        """Check that rmatmul returns the expected values
        """
        z = self.gf.rmatmul(self.y[sctype], self.x[sctype])
        self.assertArrayAllClose(z, self.z[sctype])


# =============================================================================
# %% Test cloop ufuncs
# =============================================================================


class TestCloop(TestBlas):
    """Testing norm, matmul, rmatmul and rtrue_tdivide
    """
    def setUp(self):
        super().setUp()
        self.gf = gfc

    def test_rdiv_shape(self):
        """Check that rtrue_divide returns arrays with the expected shape
        """
        # shape
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rtrue_divide(self.x['d'], self.z['d'])

    @utn.loop_test(msg="x \\ y == y / x. ", attr_inds=slice(1, None))
    def test_rdiv_val(self, sctype):
        """Check that rtrue_divide returns the expected values
        """
        # value
        x = self.x[sctype]
        y = self.y[sctype].T[:, None]
        z = self.gf.rtrue_divide(x, y)
        zz = y / x
        self.assertArrayAllClose(z, zz)
        self.assertArrayNotAllClose(z, x / y, msg='x \\ y != x / y')


# =============================================================================
# %% Test qr
# =============================================================================


class TestQR(utn.TestCaseNumpy):
    """Testing gufuncs_lapack.qr_*
    """
    def setUp(self):
        super().setUp()
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

    @errstate
    def test_qr_shape(self):
        """Check that qr_* all return arrays with the expected shape
        """
        with self.subTest(msg='wide'):
            q, r = gfl.qr_m(self.wide['d'])
            self.assertEqual(q.shape, (120, 10, 5, 5))
            self.assertEqual(r.shape, (120, 10, 5, 16))
        with self.subTest(msg='tall'):
            q, r = gfl.qr_n(self.tall['d'])
            self.assertEqual(q.shape, (120, 10, 16, 5))
            self.assertEqual(r.shape, (120, 10, 5, 5))
            with self.assertRaisesRegex(*utn.invalid_err):
                gfl.qr_n(self.wide['d'])
        with self.subTest(msg='complete'):
            q, r = gfl.qr_m(self.tall['d'])
            self.assertEqual(q.shape, (120, 10, 16, 16))
            self.assertEqual(r.shape, (120, 10, 16, 5))
        with self.subTest(msg='raw'):
            h, tau = gfl.qr_rawn(self.tall['d'])
            self.assertEqual(h.shape, (120, 10, 5, 16))
            self.assertEqual(tau.shape, (120, 10, 5))

    @utn.loop_test(msg='wide')
    def test_qr_wide(self, sctype):
        """Check that qr_m returns the expected values on wide matrices
        """
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

    @utn.loop_test(msg='tall')
    def test_qr_tall(self, sctype):
        """Check that qr_n returns the expected values
        """
        q, r = gfl.qr_n(self.tall[sctype])
        tall = q @ r
        eye = transpose(q.conj()) @ q
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small[sctype], eye)

    @utn.loop_test(msg='complete')
    def test_qr_complete(self, sctype):
        """Check that qr_m returns the expected values on tall matrices
        """
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

    @utn.loop_test()
    def test_qr_r(self, sctype):
        """Check that qr_rm, qr_rn return the expected values
        """
        with self.subTest(msg='r_m'):
            r = gfl.qr_rm(self.wide[sctype])
            rr = gfl.qr_m(self.wide[sctype])[1]
            self.assertArrayAllClose(r, rr)
        with self.subTest(msg='r_n'):
            r = gfl.qr_rn(self.tall[sctype])
            rr = gfl.qr_n(self.tall[sctype])[1]
            self.assertArrayAllClose(r, rr)

    @utn.loop_test(attr_inds=1)
    def test_qr_raw(self, sctype):
        """Check that qr_rawm, qr_rawn return the expected values
        """
        rr = gfl.qr_m(self.wide[sctype])[1]
        n = rr.shape[-2]
        ht, tau = gfl.qr_rawm(self.wide[sctype])
        h = transpose(ht)
        v = np.tril(h[..., :n], -1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(v, axis=-2)**2 * tau
        r = np.triu(h)
        with self.subTest(msg='raw_m'):
            self.assertArrayAllClose(r, rr)
            self.assertArrayAllClose(vn[..., :-1], 2)
            self.assertArrayAllClose(vn[..., -1], 0)
        for k in range(2, n+1):
            vr = v[..., None, :, -k] @ r
            r -= tau[..., None, None, -k] * v[..., -k, None] * vr
        with self.subTest(msg='h_m'):
            self.assertArrayAllClose(r, self.wide[sctype])

        rr = gfl.qr_n(self.tall[sctype])[1]
        n = rr.shape[-1]
        ht, tau = gfl.qr_rawn(self.tall[sctype])
        h = transpose(ht)
        v = np.tril(h, -1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(v, axis=-2)**2 * tau
        r = np.triu(h)
        with self.subTest(msg='raw_n'):
            self.assertArrayAllClose(r[..., :5, :], rr)
            self.assertArrayAllClose(vn, 2)
        for k in range(1, n+1):
            vr = v[..., None, :, -k] @ r
            r -= tau[..., None, None, -k] * v[..., -k, None] * vr
        with self.subTest(msg='h_n'):
            self.assertArrayAllClose(r, self.tall[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
