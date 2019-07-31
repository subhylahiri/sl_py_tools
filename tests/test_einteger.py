# -*- coding: utf-8 -*-
"""
"""
from math import inf, nan
import sl_py_tools.integer_tricks as ig
from sl_py_tools.tests.unittest_numpy import main, TestCaseNumpy

__all__ = ['TestConversion', 'TestComparison', 'TestArithmetic', 'TestModulo',
           'TestGCD', 'TestInvertDivm', 'TestInplace']


class TestConversion(TestCaseNumpy):
    """docstring for TestConversion."""

    def setUp(self):
        super().setUp()

    def test_eint_converts_to_int(self):
        self.assertEqual(int(ig.eint(5)), 5)
        self.assertSameRepr(int(ig.eint(5)), 5)

    def test_eint_converts_to_float(self):
        self.assertEqual(float(ig.eint(5)), 5.0)
        self.assertEqual(float(ig.inf), inf)

    def test_naninf_wont_convert_to_int(self):
        self.assertRaises(ValueError, int, ig.nan)
        self.assertRaises(OverflowError, int, ig.inf)

    def test_finite_float_converts_to_eint(self):
        self.assertSameRepr(ig.eint(4.3131), ig.eint(4))
        self.assertSameRepr(ig.eint(4.8131), ig.eint(4))


class TestComparison(TestCaseNumpy):
    """docstring for TestComparison."""

    def setUp(self):
        super().setUp()

    def test_finite_comparisons(self):
        self.assertTrue(ig.eint(4) == 4, msg='finite eq')
        self.assertTrue(ig.eint(432) != 564, msg='finite neq')
        self.assertTrue(432 <= ig.eint(564), msg='finite leq')
        self.assertTrue(ig.eint(987) >= 987, msg='finite geq')
        self.assertTrue(ig.eint(432) < ig.eint(564), msg='finite lt')
        self.assertTrue(ig.eint(432) > 56, msg='finite gt')
        self.assertFalse(9 == ig.eint(4), msg='finite (not) equal')
        self.assertFalse(ig.eint(34) != 34, msg='finite (not) not equal')
        self.assertFalse(ig.eint(432) <= ig.eint(21), msg='finite (not) leq')
        self.assertFalse(ig.eint(8) >= 88, msg='finite (not) geq')
        self.assertFalse(708 < ig.eint(564), msg='finite (not) lt')
        self.assertFalse(ig.eint(432) > 432, msg='finite (not) gt')

    def test_inf_comparisons(self):
        self.assertTrue(ig.inf == inf, msg='infinite eq')
        self.assertTrue(ig.eint(432) != inf, msg='infinite neq')
        self.assertTrue(-ig.inf <= ig.eint(564), msg='infinite leq')
        self.assertTrue(ig.inf >= inf, msg='infinite geq')
        self.assertTrue(-ig.inf < ig.inf, msg='infinite lt')
        self.assertTrue(ig.eint(432) > -inf, msg='infinite gt')
        self.assertFalse(9 == ig.inf, msg='infinite (not) equal')
        self.assertFalse(-ig.inf != -inf, msg='infinite (not) not equal')
        self.assertFalse(ig.inf <= -ig.inf, msg='infinite (not) leq')
        self.assertFalse(ig.eint(32) >= inf, msg='infinite (not) geq')
        self.assertFalse(708 < -ig.inf, msg='infinite (not) lt')
        self.assertFalse(ig.inf > inf, msg='infinite (not) gt')

    def test_nan_comparisons(self):
        self.assertTrue(ig.eint(432) != nan, msg='nan neq')
        self.assertTrue(432 != ig.nan, msg='nan neq')
        self.assertTrue(ig.nan != nan, msg='nan neq')
        self.assertTrue(ig.nan != ig.nan, msg='nan neq')
        self.assertTrue(ig.inf != nan, msg='nan neq')
        self.assertTrue(inf != ig.nan, msg='nan neq')
        self.assertFalse(9 == ig.nan, msg='nan (not) equal')
        self.assertFalse(ig.nan <= ig.nan, msg='nan (not) leq')
        self.assertFalse(ig.nan >= nan, msg='nan (not) geq')
        self.assertFalse(nan < -ig.inf, msg='nan (not) lt')
        self.assertFalse(ig.inf > nan, msg='nan (not) gt')


class TestArithmetic(TestCaseNumpy):
    """docstring for TestArithmetic."""

    def setUp(self):
        super().setUp()

    def test_finite_arithmetic(self):
        self.assertSameRepr(ig.eint(5)**2, ig.eint(25))
        self.assertSameRepr(ig.eint(5) * 2.3, ig.eint(11))
        self.assertSameRepr(2 ** ig.eint(7), ig.eint(128))

    def test_inf_arithmetic(self):
        self.assertSameRepr(ig.inf + inf, ig.inf)
        self.assertSameRepr(ig.inf // ig.inf, ig.nan)
        self.assertSameRepr(inf - ig.inf, ig.nan)
        self.assertSameRepr(-inf * ig.inf, -ig.inf)

    def test_nan_arithmetic(self):
        self.assertSameRepr(ig.inf + -inf, ig.nan)
        self.assertSameRepr(ig.nan * 4.2, ig.nan)
        self.assertSameRepr(nan ** ig.eint(4.2), ig.nan)


class TestModulo(TestCaseNumpy):
    """docstring for TestArithmetic."""

    def setUp(self):
        super().setUp()

    def test_finite_modulo(self):
        x = 5
        self.assertSameRepr(ig.eint(x) % 2, ig.eint(1))
        self.assertSameRepr(ig.eint(x)**2 % 7, ig.eint(4))
        self.assertSameRepr(divmod(ig.eint(x)**2, 7), (ig.eint(3), ig.eint(4)))
        self.assertSameRepr(x % ig.eint(2), ig.eint(1))
        self.assertSameRepr(x**2 % ig.eint(7), ig.eint(4))
        self.assertSameRepr(divmod(x**2, ig.eint(7)), (ig.eint(3), ig.eint(4)))

    def test_inf_modulo(self):
        self.assertEqual(ig.inf % 4, 0)
        self.assertEqual(inf % ig.eint(4), 0)
        self.assertEqual(ig.eint(2) % inf, 2)
        self.assertEqual(56 % ig.inf, ig.eint(56))
        self.assertSameRepr(ig.inf % 4, ig.eint(0))
        self.assertSameRepr(inf % ig.eint(4), ig.eint(0))
        self.assertSameRepr(ig.eint(2) % inf, ig.eint(2))
        self.assertSameRepr(56 % ig.inf, ig.eint(56))
        self.assertEqual(-ig.inf % 4, 0)
        self.assertEqual(ig.inf % -7, 0)
        self.assertEqual(-inf % ig.eint(4), 0)
        self.assertEqual(inf % ig.eint(-4), 0)
        self.assertEqual(ig.eint(2) % -inf, 2)
        self.assertEqual(56 % -ig.inf, ig.eint(56))
        self.assertEqual(-56 % ig.inf, -ig.eint(56))

    def test_nan_modulo(self):
        self.assertSameRepr(ig.nan % 4, ig.nan)
        self.assertSameRepr(nan % ig.eint(4), ig.nan)
        self.assertSameRepr(ig.eint(2) % nan, ig.nan)
        self.assertSameRepr(56 % ig.nan, ig.nan)


class TestGCD(TestCaseNumpy):
    """docstring for TestGCD."""

    def setUp(self):
        super().setUp()

    def test_finite_gcd(self):
        self.assertEqual(ig.gcd(72, 21), 3)
        self.assertSameRepr(ig.gcd(72, 21), 3)
        self.assertSameRepr(ig.gcd(ig.eint(72), 21), ig.eint(3))
        self.assertSameRepr(ig.gcd(72, ig.eint(21)), ig.eint(3))
        self.assertSameRepr(ig.gcd(ig.eint(72), ig.eint(21)), ig.eint(3))
        self.assertSameRepr(ig.gcd(ig.eint(72), -ig.eint(21)), ig.eint(3))
        self.assertSameRepr(ig.gcd(-ig.eint(72), ig.eint(21)), ig.eint(3))
        self.assertSameRepr(ig.gcd(-ig.eint(72), -ig.eint(21)), ig.eint(3))

    def test_infinite_gcd(self):
        self.assertEqual(ig.gcd(inf, 21), 21)
        self.assertSameRepr(ig.gcd(72, inf), 72)
        self.assertSameRepr(ig.gcd(-ig.inf, 21), ig.eint(21))
        self.assertSameRepr(ig.nan_gcd(ig.eint(72), inf), ig.eint(72))
        self.assertSameRepr(ig.gcd(-ig.inf, ig.eint(21)), ig.eint(21))

    def test_nan_gcd(self):
        self.assertEqual(ig.nan_gcd(65, nan), 65)
        self.assertSameRepr(ig.nan_gcd(nan, 65), 65)
        self.assertTrue(ig.isnan(ig.gcd(nan, 65)))
        self.assertSameRepr(ig.gcd(65, nan), nan)
        self.assertSameRepr(ig.gcd(65, ig.nan), ig.nan)
        self.assertSameRepr(ig.gcd(ig.eint(65), nan), ig.nan)
        self.assertSameRepr(ig.gcd(ig.inf, ig.nan), ig.nan)
        self.assertSameRepr(ig.nan_gcd(ig.nan, 65), ig.eint(65))


class TestInvertDivm(TestCaseNumpy):
    """docstring for TestGCD."""

    def setUp(self):
        super().setUp()

    def test_finite_invert(self):
        self.assertEqual(ig.invert(6, 25), 21)
        self.assertEqual(ig.mod(6 * 21, 25), 1)
        self.assertSameRepr(ig.invert(6, 25), 21)
        self.assertSameRepr(ig.invert(ig.eint(6), 25), ig.eint(21))
        self.assertSameRepr(ig.invert(6, ig.eint(25)), ig.eint(21))

    def test_infinite_invert(self):
        self.assertSameRepr(ig.invert(1, inf), 1)
        self.assertSameRepr(ig.invert(1, ig.inf), ig.eint(1))
        self.assertSameRepr(ig.invert(ig.eint(1), inf), ig.eint(1))
        self.assertSameRepr(ig.invert(ig.eint(1), -inf), ig.eint(1))
        self.assertSameRepr(ig.invert(-1, ig.inf), -ig.eint(1))
        self.assertSameRepr(ig.invert(-ig.eint(1), -ig.inf), -ig.eint(1))

    def test_nan_invert(self):
        self.assertSameRepr(ig.invert(1, nan), nan)
        self.assertSameRepr(ig.invert(nan, nan), nan)
        self.assertSameRepr(ig.invert(13, ig.nan), ig.nan)
        self.assertSameRepr(ig.invert(ig.eint(9), nan), ig.nan)
        self.assertSameRepr(ig.invert(ig.nan, -inf), ig.nan)
        self.assertSameRepr(ig.invert(-7, ig.nan), ig.nan)
        self.assertSameRepr(ig.invert(-ig.inf, ig.nan), ig.nan)

    def test_noncoprime_invert_raises(self):
        self.assertRaises(ZeroDivisionError, ig.invert, 6, 21)
        self.assertRaises(ZeroDivisionError, ig.invert, ig.eint(6), 21)
        self.assertRaises(ZeroDivisionError, ig.invert, ig.inf, 21)
        self.assertRaises(ZeroDivisionError, ig.invert, ig.eint(6), inf)

    def test_finite_divm(self):
        self.assertEqual(ig.divm(14, 6, 25), 19)
        self.assertSameRepr(ig.mod(6 * 19, 25), 14)
        self.assertSameRepr(ig.divm(14, ig.eint(6), 25), ig.eint(19))
        self.assertSameRepr(ig.divm(14, 6, ig.eint(25)), ig.eint(19))
        self.assertSameRepr(ig.divm(ig.eint(14), ig.eint(6), 25), ig.eint(19))
        self.assertSameRepr(ig.divm(ig.eint(14), 6, ig.eint(25)), ig.eint(19))

    def test_infinite_divm(self):
        self.assertSameRepr(ig.divm(4, 1, inf), 4)
        self.assertSameRepr(ig.divm(34, 1, ig.inf), ig.eint(34))
        self.assertSameRepr(ig.divm(-19, ig.eint(1), inf), -ig.eint(19))
        self.assertSameRepr(ig.divm(83, ig.eint(1), -inf), ig.eint(83))
        self.assertSameRepr(ig.divm(-176, -1, ig.inf), ig.eint(176))
        self.assertSameRepr(ig.divm(1, -ig.eint(1), -ig.inf), -ig.eint(1))

    def test_nan_divm(self):
        self.assertSameRepr(ig.divm(71, 1, nan), nan)
        self.assertSameRepr(ig.divm(12, nan, nan), nan)
        self.assertSameRepr(ig.divm(1, 13, ig.nan), ig.nan)
        self.assertSameRepr(ig.divm(nan, ig.eint(9), nan), ig.nan)
        self.assertSameRepr(ig.divm(56, ig.nan, -inf), ig.nan)
        self.assertSameRepr(ig.divm(ig.inf, -7, ig.nan), ig.nan)
        self.assertSameRepr(ig.divm(91, -ig.inf, ig.nan), ig.nan)

    def test_noncoprime_divm_raises(self):
        self.assertRaises(ZeroDivisionError, ig.divm, 8, 6, 21)
        self.assertRaises(ZeroDivisionError, ig.divm, 4, ig.eint(6), 21)
        self.assertRaises(ZeroDivisionError, ig.divm, ig.eint(7), inf, 21)
        self.assertRaises(ZeroDivisionError, ig.divm, 4, ig.eint(6), ig.inf)


class TestInplace(TestCaseNumpy):
    """docstring for TestInplace."""

    def setUp(self):
        super().setUp()
        self.x = ig.eint(4)
        self.pinf = ig.inf
        self.ninf = -ig.inf
        self.nan = ig.nan

    def test_iplus(self):
        self.x += 3
        self.assertSameRepr(self.x, ig.eint(7))
        self.x += ig.eint(8)
        self.assertSameRepr(self.x, ig.eint(15))
        self.pinf += -ig.inf
        self.assertSameRepr(self.pinf, ig.nan)

    def test_imul(self):
        self.x *= 3
        self.assertSameRepr(self.x, ig.eint(12))
        self.x *= ig.eint(8)
        self.assertSameRepr(self.x, ig.eint(96))
        self.pinf *= -ig.inf
        self.assertSameRepr(self.pinf, -ig.inf)

    def test_itruediv(self):
        with self.assertRaises(TypeError):
            self.x /= ig.eint(8)

    def test_ifloordiv(self):
        self.x *= 3
        self.x //= ig.eint(8)
        self.assertSameRepr(self.x, ig.eint(1))
        self.x *= 20
        self.x //= ig.eint(7)
        self.assertSameRepr(self.x, ig.eint(2))
        self.pinf //= -ig.inf
        self.assertSameRepr(self.pinf, ig.nan)

    def test_imod(self):
        self.x *= 3
        self.x %= ig.eint(8)
        self.assertSameRepr(self.x, ig.eint(4))
        self.x *= 5
        self.x %= ig.eint(7)
        self.assertSameRepr(self.x, ig.eint(6))
        self.pinf %= -ig.inf
        self.assertSameRepr(self.pinf, ig.eint(0))


if __name__ == '__main__':
    main()
