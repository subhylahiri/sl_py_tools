# -*- coding: utf-8 -*-
"""Tricks for working with integers
"""
from __future__ import annotations

import abc
import math
from functools import reduce, wraps
from numbers import Integral, Number, Real
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

# import gmpy2
import sl_py_tools.number_like as _nl

# =============================================================================
# Rounding and division for extended integers
# =============================================================================


def _extended_pass(func: Callable[[Number, Number]]) -> Callable[[Eint], Eint]:
    """Wrap a function to pass non-finite values unchanged"""
    @wraps(func)
    def passer(val: Eint) -> Eint:
        return func(val) if isfinite(val) else val
    return passer


ceil, floor, trunc, round_ = map(_extended_pass, (math.ceil, math.floor,
                                                  math.trunc, round))


def floor_divide(numerator: Number, denominator: Number) -> int:
    """Floor division for extended integers.

    Similar to `numerator // denominator`, but uses `ceil` on the result
    rather than `floor`.

    See Also
    --------
    operator.floordiv
    math.floor
    """
    return floor(numerator / denominator)


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
    return round_(numerator / denominator, ndigits)


# =============================================================================
# Modulo for extended integers
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


def divmod_(dividend: Number, divisor: Number) -> Tuple[Number, Number]:
    """Quotient and remainder for extended integers.

    Roughly the same as `(dividend // divisor, mod(dividend, divisor))`.

    `dividend = divisor * quotient + remainder`.

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
# Extended integers ABC
# =============================================================================


class ExtendedIntegralMeta(abc.ABCMeta):
    """Metaclass for ExtendedIntegral ABC.

    If class is a virtual superclass of Integral:
    Instance checks return `True` for all `Integral`s and non-finite `Real`s.
    Subclass checks return `True` for `Integrals`.
    """

    def __instancecheck__(cls, instance):
        # Do not want this to propagate to subclasses
        if cls == ExtendedIntegral:
            if isinstance(instance, Real) and not math.isfinite(instance):
                return True
        return super().__instancecheck__(instance)


# pylint: disable=abstract-method
class ExtendedIntegral(Real, metaclass=ExtendedIntegralMeta):
    """ABC for extended Integral, including +/-inf and nan.

    Instance checks return `True` for non-finite `Real`s and all `Integral`s.
    Subclass checks return `True` for `Integrals`.
    """


ExtendedIntegral.register(Integral)


# =============================================================================
# ExtendedInt method wrappers
# =============================================================================
# _TYPES = (Real, type(gmpy2.mpz(1)))
_TYPES = ExtendedIntegral
_nmspace = SimpleNamespace()
_nmspace.floordiv = floor_divide
_nmspace.mod = mod
_nmspace.divmod = divmod_
_nmspace.round = round_
_nmspace.trunc = trunc
_nmspace.floor = floor
_nmspace.ceil = ceil
_NMSPACE = (None, None, None, _nmspace)


def _eint_conv(args):
    """Convert to Number
    """
    def _conv(arg):
        if isinstance(arg, ExtendedInt):
            return arg.value
        if isinstance(arg, Number):
            return arg
        raise TypeError("Other argument must be a number or eint")
    return [_conv(arg) for arg in args]


_eint_in = _nl.in_method_wrapper(_eint_conv)

# =============================================================================
# Extended integers
# =============================================================================


@ExtendedIntegral.register
class ExtendedInt(_nl.number_mixin(_eint_conv, _TYPES, _NMSPACE)):
    """Extended integers to include +/-inf and nan.

    All of the usual operations and built in functions for numeric types are
    defined, except for the bitwise ones. If any argument is an `eint` and the
    result is an `Eint`, it will be converted to an `eint`, with the obvious
    exceptions: comparison, casting...

    It can be converted to an ordinary number by calling `int(eint)` or
    `float(eint)`.

    Parameters
    ----------
    value : Real
        The value being represented. Stored as an `int` (via constructor),
        unless it in `nan` or `inf`, in which case it is stored as a `float`.
    """
    value: ExtendedIntegral

    __str__ = _eint_in(str)
    __hash__ = _eint_in(hash)
    # Don't define in-place ops -> immutable like numbers

    def __init__(self, value: ExtendedIntegral):
        try:
            self.value = int(value)
        except (ValueError, OverflowError):
            self.value = float(value)

    def __repr__(self):
        return f"eint({self})"

    def __getattr__(self, name):
        return getattr(self.value, name)

    @property
    def real(self) -> ExtendedInt:
        """real part = self"""
        return ExtendedInt(self.value.real)

    @property
    def imag(self) -> ExtendedInt:
        """imaginary part = 0"""
        return ExtendedInt(self.value.imag)

    def conjugate(self) -> ExtendedInt:
        """conjugate = self"""
        return ExtendedInt(self.value.conjugate())


# =============================================================================
# ExtendedInt finalise & function decorators
# =============================================================================

_nl.set_objclasses(ExtendedInt)
eint_in, eint_out = _nl.function_decorators(_eint_conv, ExtendedInt, _TYPES)


# =============================================================================
# Convenience
# =============================================================================


def isinfnone(val: Optional[Eint]) -> bool:
    """is val None/inf?"""
    return val is None or isinf(val)

# =============================================================================
# More modulo for Extended Integers
# =============================================================================


@eint_out
def nan_gcd(left: Eint,
            right: Eint) -> Eint:
    """Greatest common divisor for extended integers.

    Largest `d` such that `left % d == 0 and right % d == 0`.

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
    if math.isfinite(left) and math.isfinite(right):
        return math.gcd(left, right)
    if math.isfinite(right) or math.isnan(left):
        return abs(right)
    return abs(left)


@eint_out
def gcd(left: Eint, right: Eint) -> Eint:
    """Greatest common divisor for extended integers.

    Largest `d` such that `left % d == 0 and right % d == 0`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    math.gcd
    gmpy2.gcd
    nan_gcd
    """
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return nan_gcd(left, right)


@eint_out
def nan_lcm(left: Eint,
            right: Eint) -> Eint:
    """Least common multiple for extended integers.

    Smallest `val` such that `val % left == 0 and val % right == 0`.

    NaN safe version: ignores `nan` if the other argument is not `nan`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    lcm
    """
    if math.isfinite(left) and math.isfinite(right):
        return (left * right) // nan_gcd(left, right)
        # return int(gmpy2.lcm(left, right))
    if math.isnan(left):
        return right
    if math.isnan(right):
        return left
    return inf


@eint_out
def lcm(left: Eint, right: Eint) -> Eint:
    """Least common multiple for extended integers.

    Smallest `val` such that `val % left == 0 and val % right == 0`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.lcm
    nan_lcm
    """
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return nan_lcm(left, right)


@eint_out
def invert(val: Eint,
           period: Eint) -> Eint:
    """Multiplicative inverse (modulo period) for extended integers.

    Return `inv_val` such that `val * inv_val == 1 (mod period)`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.invert
    """
    if math.isnan(val) or math.isnan(period):
        return math.nan
    if period in {1, -1}:
        # everything == 0 (mod period), soeverything is an inverse
        return 0
    if val in {1, -1}:
        return val
    if math.isfinite(val) and math.isfinite(period):
        try:
            return pow(val, -1, period)
        except ValueError as exc:
            if exc.args[0] != 'base is not invertible for the given modulus':
                raise
    # if period == inf: inv(val) = 1/val,  - not invertible unless in {1, -1}
    # if val == inf: inf = 0 (mod period) - not invertible
    raise ZeroDivisionError(f'{val} is not invertible (mod {period})')


@eint_out
def divm(left: Eint, right: Eint,
         period: Eint) -> Eint:
    """Division (modulo period) for extended integers.

    Return `val` such that `val * right == left (mod period)`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

    If any argument is an `eint` the result will be too.

    See Also
    --------
    gmpy2.divm
    """
    if math.isnan(left) or math.isnan(right) or math.isnan(period):
        return math.nan
    if mod(left, period) == 0:
        return 0
    if math.isfinite(left) and math.isfinite(right) and period in {1, -1}:
        # everything == 0 (mod +/-1)
        return 0
    # if left == inf -> factor == right, so left -> inf, right -> 1
    # if right == inf -> factor == left, so left -> 1, right -> inf
    factor = nan_gcd(left, right)
    left, right = floor_divide(left, factor), floor_divide(right, factor)
    # now gcd(left, right) == 1 && left != 0
    # -> gcd(right, period) == 1 or no solution
    return mod(left * invert(right, period), period)


@eint_out
def gcdn(*args: Eint, nan_safe: bool = False) -> Eint:
    """Greatest common divisor of many extended integers.

    Parameters
    ----------
    *args: ExtendedIntegral
        Numbers whose gcd we are finding.
    nan_safe : bool
        If True, `nan`s will not propagate. Keyword only, default: False.

    Returns
    -------
    the_gcd: ExtendedIntegral
        Largest `d` such that `left % d == 0 for all left in args`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.

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
def lcmn(*args: Eint, nan_safe: bool = False) -> Eint:
    """Lowest common multiple of many extended integers.

    Parameters
    ----------
    *args: ExtendedIntegral
        Numbers whose lcm we are finding.
    nan_safe : bool
        If True, `nan`s will not propagate. Keyword only, default: False.

    Returns
    -------
    the_lcm: ExtendedIntegral
        Smallest `val` such that `val % left == 0 for all left in args`.

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


# =============================================================================
# Convenience
# =============================================================================
eint = ExtendedInt  # pylint: disable=invalid-name
Eint = ExtendedIntegral
nan = eint('nan')
inf = eint('inf')
# =============================================================================
# Wrapped functions
# =============================================================================
isinf, isnan, isfinite = map(eint_in, (math.isinf, math.isnan, math.isfinite))
mod, divmod_ = map(eint_out, (mod, divmod_))
ceil, floor, trunc, round_ = map(eint_out, (ceil, floor, trunc, round_))
floor_divide, ceil_divide = map(eint_out, (floor_divide, ceil_divide))
round_divide, trunc_divide = map(eint_out, (round_divide, trunc_divide))
