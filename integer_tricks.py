# -*- coding: utf-8 -*-
"""Tricks for manipulating integers
"""
import math
from math import floor, ceil, trunc
from typing import Optional
from numbers import Number, Real, Integral
from functools import wraps, total_ordering
import operator
import gmpy2
# from .arg_tricks import default


def ceil_divide(numerator: Number, denominator: Number) -> int:
    """Ceiling division
    """
    return ceil(numerator / denominator)


def trunc_divide(numerator: Number, denominator: Number) -> int:
    """Truncated division
    """
    return trunc(numerator / denominator)


def round_divide(numerator: Number,
                 denominator: Number,
                 ndigits: Optional[int] = None) -> Number:
    """Rounded division
    """
    return round(numerator / denominator, ndigits)


# =============================================================================
# %%* ExtendedInt helpers
# =============================================================================


def _arg_conv(args):
    """Convert to Number
    """
    def _conv(arg):
        if isinstance(arg, ExtendedInt):
            return arg.value
        if isinstance(arg, Number):
            return arg
        raise TypeError("Other argument must be a number or eint")
    return [_conv(x) for x in args]


def _implement_op(func, args):
    """Implement operator for ExtendedInt
    """
    try:
        return ExtendedInt(func(*_arg_conv(args)))
    except TypeError:
        return NotImplemented


def _implement_rop(func, args):
    """Implement reversed operator for ExtendedInt
    """
    try:
        return ExtendedInt(func(*reversed(_arg_conv(args))))
    except TypeError:
        return NotImplemented


def _implement_iop(func, args):
    """Implement inplace operator for ExtendedInt
    """
    try:
        args[0].value = func(*_arg_conv(args))
    except TypeError:
        return NotImplemented
    return args[0]


def _implement_ext_op(func, args):
    """Implement operator/function for ExtendedInt that returns another type
    """
    try:
        return func(*_arg_conv(args))
    except TypeError:
        return NotImplemented


def _implement_int_op(func, args):
    """Implement function for ExtendedInt or Number
    """
    try:
        result = func(*_arg_conv(args))
    except TypeError:
        return NotImplemented
    if any(isinstance(x, ExtendedInt) for x in args):
        return ExtendedInt(result)
    return result


WRAPPER_ASSIGNMENTS = ('__name__', '__doc__', '__annotations__')


def _delegate_ext_op(func):
    """Wrap operator/function for ExtendedInt that returns another type
    """
    @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
    def wrapper(*args):
        return _implement_ext_op(func, args)
    return wrapper


def _delegate_int_op(func):
    """Wrap function for ExtendedInt or Number
    """
    @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
    def wrapper(*args):
        return _implement_int_op(func, args)
    return wrapper


def _delegate_op(func):
    """Wrap operator/function for ExtendedInt
    """
    @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
    def wrapper(*args):
        return _implement_op(func, args)
    return wrapper


def _delegate_ops(func):
    """Wrap operator/function for ExtendedInt
    """
    @wraps(func, assigned=WRAPPER_ASSIGNMENTS[1:])
    def wrapper(*args):
        return _implement_op(func, args)

    @wraps(func, assigned=WRAPPER_ASSIGNMENTS[1:])
    def rwrapper(*args):
        return _implement_rop(func, args)

    @wraps(func, assigned=WRAPPER_ASSIGNMENTS[1:])
    def iwrapper(*args):
        return _implement_iop(func, args)

    wrapper.__name__ = '__' + func.__name__.strip('_') + '__'
    rwrapper.__name__ = '__r' + func.__name__.strip('_') + '__'
    iwrapper.__name__ = '__i' + func.__name__.strip('_') + '__'
    return wrapper, rwrapper, iwrapper


# =============================================================================
# %%* Modulo for extended integers
# =============================================================================


def mod(dividend: Number, divisor: Number) -> Number:
    """Modulo for extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.
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
    """
    if math.isnan(dividend) or math.isnan(divisor):
        return (math.nan, math.nan)
    if math.isfinite(dividend) and math.isfinite(divisor):
        return divmod(dividend, divisor)
    if math.isinf(dividend):
        return (math.inf, 0) if (divisor > 0) else (-math.inf, 0)
    return (0, dividend)


@_delegate_int_op
def gcd(a: Integral, b: Integral) -> Integral:
    """Greatest common divisor for extended integers.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    """
    if math.isnan(a) or math.isnan(b):
        return math.nan
    if math.isfinite(a) and math.isfinite(b):
        return math.gcd(a, b)
    if math.isinf(a):
        return b
    return a


@_delegate_int_op
def invert(x: Integral, m: Integral) -> Integral:
    """Multiplicative inverse (modulo m) for extended integers.

    Return `y` such that `x * y == 1 (mod m)`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.
    """
    if math.isnan(x) or math.isnan(m):
        return math.nan
    if m in {1, -1}:
        # everything is an inverse
        return 0
    if math.isfinite(x) and math.isfinite(m):
        return gmpy2.invert(x, m)
    if x in {1, -1}:
        return x
    raise ZeroDivisionError('not invertible')


@_delegate_int_op
def divm(a: Number, b: Number, m: Number) -> Integral:
    """Division (modulo m) for extended integers.

    Return `x` such that `x * b == a (mod m)`.

    Extended integers include `nan` and `+/-inf`. We act as if:
    - `inf` is the product of all positive numbers, so `inf % anything == 0`.
    - `inf * 0 == 0 (mod inf)`, so that `anything % inf == anything`.
    """
    if math.isnan(a) or math.isnan(b) or math.isnan(m):
        return math.nan
    if math.isfinite(a) and math.isfinite(b) and math.isfinite(m):
        if a == 0 and m in {1, -1}:
            return 0
        return gmpy2.divm(a, b, m)
    return mod(a * invert(b, m), m)


isinf = _delegate_ext_op(math.isinf)
isnan = _delegate_ext_op(math.isnan)
isfinite = _delegate_ext_op(math.isfinite)

# =============================================================================
# %%* Extended integers
# =============================================================================


class ExtendedIntegral(Real):
    """ABC for extended Integral, including +/-inf and nan."""


ExtendedIntegral.register(Integral)


@total_ordering
class ExtendedInt(ExtendedIntegral):
    """Extended integers to include +/-inf and nan.
    """
    value: Real

    __str__ = _delegate_ext_op(str)
    __hash__ = _delegate_ext_op(hash)
    __eq__ = _delegate_ext_op(operator.__eq__)
    __lt__ = _delegate_ext_op(operator.__lt__)
    __le__ = _delegate_ext_op(operator.__le__)
    __add__, __radd__, __iadd__ = _delegate_ops(operator.add)
    __mul__, __rmul__, __imul__ = _delegate_ops(operator.mul)
    __truediv__, __rtruediv__, __itruediv__ = _delegate_ops(operator.truediv)
    __floordiv__, __rfloordiv__, __ifloordiv__ = _delegate_ops(
                                                            operator.floordiv)
    __mod__, __rmod__, __imod__ = _delegate_ops(mod)
    __divmod__, __rdivmod__ = _delegate_ops(divmod_)[:2]
    __pow__, __rpow__, __ipow__ = _delegate_ops(operator.pow)
    __lshift__, __rlshift__, __ilshift__ = _delegate_ops(operator.lshift)
    __rshift__, __rrshift__, __irshift__ = _delegate_ops(operator.rshift)
    __and__, __rand__, __iand__ = _delegate_ops(operator.and_)
    __xor__, __rxor__, __ixor__ = _delegate_ops(operator.xor)
    __or__, __ror__, __ior__ = _delegate_ops(operator.or_)
    __neg__ = _delegate_op(operator.neg)
    __pos__ = _delegate_op(operator.pos)
    __abs__ = _delegate_op(operator.abs)
    __invert__ = _delegate_op(operator.invert)
    __complex__ = _delegate_ext_op(complex)
    __float__ = _delegate_ext_op(float)
    __int__ = _delegate_ext_op(int)
    __index__ = _delegate_ext_op(int)
    __round__ = _delegate_op(round)
    __trunc__ = _delegate_op(trunc)
    __floor__ = _delegate_op(floor)
    __ceil__ = _delegate_op(ceil)

    def __init__(self, value):
        try:
            self.value = int(value)
        except (ValueError, OverflowError):
            self.value = float(value)

    def __repr__(self):
        return f"eint({self})"

    def __getattr__(self, name):
        return getattr(self.value, name)


def eint(arg):
    """Convert to ExtendedInt, allowing +/-inf and nan"""
    return ExtendedInt(arg)


nan = eint('nan')
inf = eint('inf')
