# -*- coding: utf-8 -*-
"""Tricks for working with integers
"""
import abc
import math
from math import floor, ceil, trunc
from operator import floordiv
from functools import reduce
from typing import Optional
from numbers import Number, Real, Integral
import gmpy2
from . import number_like as _nl
from .arg_tricks import Export
Export[floor, floordiv]


def ceil_divide(numerator: Number, denominator: Number) -> int:
    """Ceiling division.

    Similar to `numerator // denominator`, but uses `ceil` on the result
    rather than `floor`.

    See Also
    --------
    operator.floordiv
    math.ceil
    """
    return ceil(numerator / denominator)


def trunc_divide(numerator: Number, denominator: Number) -> int:
    """Truncated division.

    Similar to `numerator // denominator`, but uses `trunc` on the result
    rather than `floor`.

    See Also
    --------
    operator.floordiv
    math.trunc
    """
    return trunc(numerator / denominator)


def round_divide(numerator: Number,
                 denominator: Number,
                 ndigits: Optional[int] = None) -> Number:
    """Rounded division.

    Similar to `numerator // denominator`, but uses `round` on the result
    rather than `floor`.

    See Also
    --------
    operator.floordiv

    See Also
    --------
    operator.floordiv
    round
    """
    return round(numerator / denominator, ndigits)


# =============================================================================
# %%* Modulo for extended integers
# =============================================================================


def mod(dividend: Number, divisor: Number) -> Number:
    """Modulo for extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    operator.mod
    """
    if math.isnan(dividend) or math.isnan(divisor):
        return math.nan
    if math.isfinite(dividend) and math.isfinite(divisor):
        return dividend % divisor
    if math.isinf(dividend):
        return 0
    return dividend


def divmod_(dividend: Number, divisor: Number) -> Number:
    """Quotient and remainder for extended integers.

    Roughly the same as `(dividend // divisor, mod(dividend, divisor))`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.
    - `anything // inf == 0`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    divmod
    """
    if math.isnan(dividend) or math.isnan(divisor):
        return (math.nan, math.nan)
    if math.isfinite(dividend) and math.isfinite(divisor):
        return divmod(dividend, divisor)
    if math.isinf(dividend):
        return (math.inf, 0) if (divisor > 0) else (-math.inf, 0)
    return (0, dividend)


# =============================================================================
# %%* ExtendedInt method wrappers
# =============================================================================
_types = (Real, type(gmpy2.mpz(1)))
_method_cache = set()


def _eint_conv(args):
    """Convert to Number
    """
    def _conv(arg):
        if isinstance(arg, ExtendedInt):
            return arg.value
        if isinstance(arg, _types):
            return arg
        raise TypeError("Other argument must be a number or eint")
    return [_conv(x) for x in args]


_eint_meth_in = _nl.in_method_wrapper(_eint_conv, _method_cache)
_eint_opr = _nl.opr_method_wrappers(_eint_conv, _method_cache, _types)

# =============================================================================
# %%* Extended integers
# =============================================================================


class ExtendedIntegralMeta(abc.ABCMeta):
    """Metaclass for ExtendedIntegral ABC.

    Instance checks return `True` for all `Integral`s and non-finite `Real`s.
    Subclass checks return `True` for `Integrals`.
    """

    def __instancecheck__(cls, instance):
        # Do not want this to propagate to subclasses
        if cls == ExtendedIntegral:
            if isinstance(instance, Real) and not math.isfinite(instance):
                return True
            if isinstance(instance, Integral):
                return True
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass):
        # Do not want this to propagate to subclasses
        if cls == ExtendedIntegral:
            if issubclass(subclass, Integral):
                return True
        return super().__subclasscheck__(subclass)


class ExtendedIntegral(Real, metaclass=ExtendedIntegralMeta):
    """ABC for extended Integral, including +/-inf and nan.

    Instance checks return `True` for non-finite `Real`s and all `Integral`s.
    Subclass checks return `True` for `Integrals`.
    """


ExtendedIntegral.register(Integral)


@ExtendedIntegral.register
class ExtendedInt(_nl.number_mixin(_eint_conv, _method_cache, _types)):
    """Extended integers to include +/-inf and nan.

    All of the usual operations and built in functions for numeric types are
    defined. If any argument is an `eint` the result will be too, with the
    obvious exceptions: comparisons, type conversion...

    It can be converted to an ordinary number by calling `int(eint)` or
    `float(eint)`.

    Parameters
    ----------
    value : Real
        The value being represented. Stored as an `int` (via constructor),
        unless it in `nan` or `inf`, in which case it is stored as a `float`.
    """
    value: ExtendedIntegral

    __str__ = _eint_meth_in(str)
    __hash__ = _eint_meth_in(hash)
    __mod__, __rmod__ = _eint_opr(mod)
    __divmod__, __rdivmod__ = _eint_opr(divmod_)
    __truediv__, __rtruediv__ = _eint_opr(_nl.dummy_method('truediv'))

    def __init__(self, value: ExtendedIntegral):
        try:
            self.value = int(value)
        except (ValueError, OverflowError):
            self.value = float(value)

    def __repr__(self):
        return f"eint({self})"

    def __getattr__(self, name):
        return getattr(self.value, name)


# =============================================================================
# %%* ExtendedInt finalise & function wrappers
# =============================================================================

_nl.set_objclasses(ExtendedInt, _method_cache)
eint_in, eint_out = _nl.function_wrappers(_eint_conv, ExtendedInt, _types)


# =============================================================================
# %%* Convenience
# =============================================================================

eint = ExtendedInt
Eint = ExtendedIntegral

nan = eint('nan')
inf = eint('inf')

isinf = eint_in(math.isinf)
isnan = eint_in(math.isnan)
isfinite = eint_in(math.isfinite)
mod = eint_out(mod)
divmod_ = eint_out(divmod_)

# =============================================================================
# %%* More modulo for Extended Integers
# =============================================================================


@eint_out
def nan_gcd(a: ExtendedIntegral, b: ExtendedIntegral) -> ExtendedIntegral:
    """Greatest common divisor for extended integers.

    NaN safe version: ignores `nan` if the other argument is not `nan`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    math.gcd
    gmpy2.gcd
    gcd
    """
    if math.isfinite(a) and math.isfinite(b):
        return math.gcd(a, b)
    if math.isfinite(b):
        return abs(b)
    return abs(a)


@eint_out
def gcd(a: ExtendedIntegral, b: ExtendedIntegral) -> ExtendedIntegral:
    """Greatest common divisor for extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    math.gcd
    gmpy2.gcd
    nan_gcd
    """
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return nan_gcd(a, b)


@eint_out
def nan_lcm(a: ExtendedIntegral, b: ExtendedIntegral) -> ExtendedIntegral:
    """Least common multiple for extended integers.

    NaN safe version: ignores `nan` if the other argument is not `nan`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.lcm
    lcm
    """
    if math.isfinite(a) and math.isfinite(b):
        return int(gmpy2.lcm(a, b))
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return inf


@eint_out
def lcm(a: ExtendedIntegral, b: ExtendedIntegral) -> ExtendedIntegral:
    """Least common multiple for extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.lcm
    nan_lcm
    """
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return nan_lcm(a, b)


@eint_out
def invert(x: ExtendedIntegral, m: ExtendedIntegral) -> ExtendedIntegral:
    """Multiplicative inverse (modulo m) for extended integers.

    Return `y` such that `x * y == 1 (mod m)`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.invert
    """
    if math.isnan(x) or math.isnan(m):
        return math.nan
    if m in {1, -1}:
        # everything is an inverse
        return 0
    if math.isfinite(x) and math.isfinite(m):
        return int(gmpy2.invert(x, m))
    if x in {1, -1}:
        return x
    # if m == inf: inv(x) = 1/x,  - not invertible unless in {1, -1}
    # if x == inf: inf = 0 (mod m) - not invertible
    raise ZeroDivisionError('not invertible')


@eint_out
def divm(a: ExtendedIntegral, b: ExtendedIntegral,
         m: ExtendedIntegral) -> ExtendedIntegral:
    """Division (modulo m) for extended integers.

    Return `x` such that `x * b == a (mod m)`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.divm
    """
    if math.isnan(a) or math.isnan(b) or math.isnan(m):
        return math.nan
    if math.isfinite(a) and math.isfinite(b) and math.isfinite(m):
        if m in {1, -1}:
            # everything == 0 (mod m)
            return 0
        return int(gmpy2.divm(a, b, m))
    return mod(a * invert(b, m), m)


@eint_out
def gcdn(*args, nan_safe=False):
    """Greatest common divisor of many extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    math.gcd
    gmpy2.gcd
    gcd
    nan_gcd
    """
    if nan_safe:
        return reduce(nan_gcd, args, inf)
    return reduce(gcd, args, inf)


@eint_out
def lcmn(*args, nan_safe=False):
    """Lowest common multiple of many extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.lcm
    lcm
    nan_lcm
    """
    if nan_safe:
        return reduce(nan_lcm, args, 1)
    return reduce(lcm, args, 1)
