# -*- coding: utf-8 -*-
"""Modular arithmetic
"""
from __future__ import annotations

import operator
from functools import wraps
from itertools import chain
from typing import Callable, ClassVar, List, Tuple

from . import integer_tricks as ig
from . import number_like as nl
from .containers import tuplify
from .integer_tricks import Eint, eint, inf, nan

# =============================================================================
# Operator helpers
# =============================================================================


def _wrap_mod_bypass(func: Callable, name: str = '') -> Callable:
    """wrap a function whose ouput needs a modulus but whose input does not.

    Wrapper function takes the last argument as the modulus, passes the rest to
    the wrapped function and appends the modulus to the returned tuple.
    """
    @wraps(func)
    def wrapper(*args):
        """wrapper to pass modulus along"""
        return tuplify(func(*args[:-1])) + args[-1:]
    if name:
        wrapper.__name__ = name
    return wrapper


def _wrap_mod_pass(func: Callable, name: str = '') -> Callable:
    """wrap a function whose input and ouput need a modulus.

    Wrapper function takes the last argument as the modulus, passes the rest to
    the wrapped function and appends the modulus to the returned tuple.
    """
    @wraps(func)
    def wrapper(*args):
        """wrapper to pass modulus along"""
        return tuplify(func(*args)) + args[-1:]
    if name:
        wrapper.__name__ = name
    return wrapper


def _wrap_mod_ignore(func: Callable, name: str = '') -> Callable:
    """wrap a function whose input and ouput do not need a modulus.

    Wrapper function takes the last argument as the modulus, passes the rest to
    the wrapped function.
    """
    @wraps(func)
    def wrapper(*args):
        """wrapper to not pass modulus along"""
        return func(*args[:-1])
    if name:
        wrapper.__name__ = name
    return wrapper


def _unwrap(func: Callable, name: str = '') -> Callable:
    """wrap a function to disable it.

    Wrapper returns NotImplemented
    """
    @wraps(func)
    def wrapper(*args):
        """wrapper to remove method"""
        return NotImplemented
    if name:
        wrapper.__name__ = name
    return wrapper


# -----------------------------------------------------------------------------
# Operator namespace
# -----------------------------------------------------------------------------


class Operators:
    """Namespace for operators"""
    complex = _wrap_mod_ignore(complex)
    float = _wrap_mod_ignore(float)
    int = _wrap_mod_ignore(int)
    eq = _wrap_mod_ignore(operator.eq)
    ne = _wrap_mod_ignore(operator.ne)
    add = _wrap_mod_bypass(operator.add)
    sub = _wrap_mod_bypass(operator.sub)
    mul = _wrap_mod_bypass(operator.mul)
    floordiv = _wrap_mod_pass(ig.divm, 'floordiv')
    # floordiv = _wrap_mod_bypass(operator.floordiv)
    truediv = _unwrap(operator.truediv)
    pow = _wrap_mod_bypass(operator.pow)
    pos = _wrap_mod_bypass(operator.pos)
    neg = _wrap_mod_bypass(operator.neg)
    abs = _wrap_mod_bypass(operator.abs)
    invert = _wrap_mod_pass(ig.invert)


# -----------------------------------------------------------------------------
# Intersection function
# -----------------------------------------------------------------------------


def and_(val1, mod1, val2, mod2):
    """Intersecton of congruence classes.

    Parameters
    ----------
    val1, val2
        The remainders of the two congruence classes.
    mod1, mod2
        The moduli of the two congruence classes.

    Returns
    -------
    remainder
        Remainder of the congruence class that is the intersection. If the
        intersection is empty, returns `nan`.
    modulus
        Modulus of the congruence class that is the intersection. Lowest common
        multiple of `mod1`, `mod2`.
    """
    gcd_mod = ig.gcd(mod1, mod2)
    factor1, factor2 = mod1 // gcd_mod, mod2 // gcd_mod
    lcm_mod = gcd_mod * factor1 * factor2
    if (val1 - val2) % gcd_mod:
        return (nan, lcm_mod)
    excess = (val1 - val2) // gcd_mod
    exc1, exc2 = excess % factor1, -excess % factor2
    mult1 = ig.divm(exc2, factor1, factor2)
    mult2 = ig.divm(exc1, factor2, factor1)
    dexcess = (excess + mult1*factor1 - mult2*factor2) // (factor1 * factor2)
    if dexcess > 0:
        mult2 += dexcess
    else:
        mult1 -= dexcess
    overlap = int(val1 + mult1 * mod1)
    assert overlap == val2 + mult2 * mod2
    return (overlap, lcm_mod)


# =============================================================================
# Converters & wrappers
# =============================================================================


def _convert_inputs(args):
    """Convert inputs from Mod to eint, using gcd(mod)"""
    if Mod.strict:
        return _convert_inputs_strict(args)
    modulus = get_common_mod(*args)
    return _get_remainders(*args) + [modulus]


def _convert_inputs_strict(args):
    """Convert inputs from Mod to eint, with identical mod"""
    moduli = _get_moduli(args)
    if any(x != moduli[0] for x in moduli):
        raise ValueError(f'Combining different moduli: {moduli}')
    return _get_remainders(*args) + moduli[:1]


def _flatten_inputs(args):
    """Convert inputs from Mod to pairs of eint"""
    return list(chain.from_iterable(_get_data(x) for x in args))


_METHOD_CACHE = set()
_mth = nl.one_method_wrapper(_flatten_inputs, _METHOD_CACHE, tuple)
_opf = nl.opr_method_wrappers(_flatten_inputs, _METHOD_CACHE, tuple)
_opr = nl.opr_method_wrappers(_convert_inputs, _METHOD_CACHE, tuple)
_Cnv = nl.convert_mixin(_convert_inputs, _METHOD_CACHE, Operators)
_Ops = nl.mathops_mixin(_convert_inputs, _METHOD_CACHE, tuple, Operators)


# =============================================================================
# Mod class
# =============================================================================

# pylint: disable=abstract-method
class Mod(_Cnv, _Ops):
    """Class representing a congruence class in modular arithmetic.

    Parameters
    ----------
    value : Eint
        An element of the congruence class being represented.
    modulus : Eint
        The gap between represented values, `m` in `Z/mZ`

    data : Tuple[Eint, Eint]
        Both paramters in a tuple, `(value, modulus)`.

    other : Mod
        Get `(value, modulus)` from `(other.remainder, other.modulus)`.

    Attributes
    ----------
    remainder : eint
        The the unique set member in `[0, modulus)`, i.e. `value % modulus`.
    modulus : eint
        The gap between represented values, `m` in `Z/mZ`.
    data : Tuple[eint, eint]
        Both paramters in a tuple, `(remainder, modulus)`.
    strict : class attribute, bool
        Disallow operations combining different moduli?

    Notes
    -----
    Numbers modulo `m` are members of the ring `Z/mZ`, or altenatively the sets
    ```
        Mod(a, m) = {x in Z | x == a (mod m)}.
    ```
    For numbers `x`, `x == Mod(a, m)` and `x in Mod(a, m)` are equivalent. If
    `x` is a `Mod`, `x == Mod(a, m)` tests equality of the remainders and
    equality of the moduli, whereas `x in Mod(a, m)` tests if `x` is a subset.

    The operator `~` is interpreted as the multiplicative inverse in the ring:
    ```
        ~ Mod(a, m) = {x in Z | x*a == 1 (mod m)}
    ```
    The operator `&` is interpreted as the intersection of the two sets:
    ```
        Mod(a, m) & Mod(b, n) = {x | x == a (mod m) and x == b (mod n)}.
    ```
    Division is performed by `//`, with `/` undefined. The objects can be
    converted into numbers with `int`, `float`, `complex`.

    From the ring point of view, operations betwee different moduli do not make
    sense. This approach can be enabled by setting the class variable
    `Mod.strict = True`, in which case combining diffrerent moduli raises a
    `ValueError`.

    But we can make some sense of operations between different moduli from by
    regarding these sets as limited information about an integer. This approach
    can be enabled by setting the class variable `Mod.strict = False`.

    After one of these operations we have limited information about the result.
    We can encapsulate this with a new `Mod(a, m)` that contains all of the
    possibilities, though it may also include some impossible values - we lose
    a bit more information by summarising the result with a `Mod(a, m)`.

    Other `~` and `&`, operators act on the Cartesian product: for some
    arithmetic operator which we denote by #,
    ```
        Mod(a, m) # Mod(b, n) = {x # y | x in Mod(a, m), y in Mod(b, n)}.
    ```
    We return the `Mod(c, p)` with the largest `p` that contains this set,
    `p = gcd(m,n)`.
    """
    _remainder: eint
    _modulus: eint
    strict: ClassVar[bool] = False

    __floordiv__, __rfloordiv__ = _opr(Operators.floordiv)
    __invert__ = _mth(Operators.invert)
    __and__, __rand__ = _opf(and_)

    def __init__(self, value: Eint, modulus: Eint = inf):
        super().__init__()
        if isinstance(value, Eint):
            value = (value, modulus)
        elif isinstance(value, Mod):
            value = (value.remainder, value.modulus)
        self.data = value

    def __getattr__(self, name):
        return getattr(self.remainder, name)

    def __eq__(self, other) -> bool:
        if isinstance(other, Mod):
            return self.data == other.data
        return (other % self.modulus) == self.remainder

    def __ne__(self, other) -> bool:
        return not self == other

    def __contains__(self, other) -> bool:
        if isinstance(other, Mod):
            return self & other == other
        return self == other

    def __repr__(self) -> str:
        return f"Mod({self.remainder}, {self.modulus})"

    def __str__(self) -> str:
        return f"{self.remainder} (mod {self.modulus})"

    @property
    def remainder(self) -> eint:
        """The the unique set member in `[0, modulus)`, i.e. `value % modulus`.
        """
        return self._remainder

    @remainder.setter
    def remainder(self, value: Eint):
        """The the unique set member in `[0, modulus)`, i.e. `value % modulus`.
        """
        self._remainder = eint(value) % self._modulus

    @property
    def modulus(self) -> eint:
        """The gap between represented values, `m` in `Z/mZ`."""
        return self._modulus

    @modulus.setter
    def modulus(self, value: Eint):
        self._modulus = eint(value)

    @property
    def data(self) -> Tuple[eint, ...]:
        """The tuple (remainder, modulus)"""
        return (self._remainder, self._modulus)

    @data.setter
    def data(self, value: Tuple[Eint, ...]):
        self._remainder, self._modulus = [eint(x) for x in value]
        self._remainder %= self._modulus

    @property
    def real(self) -> Mod:
        return +self

    @property
    def imag(self) -> Mod:
        return Mod((0, self.modulus))

    def conjugate(self) -> Mod:
        return +self


nl.set_objclasses(Mod, _METHOD_CACHE)
# =============================================================================
# Helpers
# =============================================================================


def _get_rem(arg) -> eint:
    """Convert an input from Mod.remainder to eint"""
    if isinstance(arg, Mod):
        return arg.remainder
    return arg


def _get_mod(arg) -> eint:
    """Convert an input from Mod.modulus to eint"""
    if isinstance(arg, Mod):
        return arg.modulus
    return inf


def _get_data(arg) -> eint:
    """Convert an input from Mod to (eint, eint)"""
    if isinstance(arg, Mod):
        return arg.data
    return arg, inf


def _get_remainders(*args: Mod) -> List[eint]:
    """Find remainders of Mod instances."""
    return [_get_rem(x) for x in args]


def _get_moduli(*args: Mod) -> List[eint]:
    """Find moduli of Mod instances."""
    return [_get_mod(x) for x in args]


# =============================================================================
# Common functions
# =============================================================================


def get_common_mod(*args: Mod) -> eint:
    """Find gcd of moduli of Mod instances."""
    return ig.gcdn(*_get_moduli(*args))


def set_common_mod(*args: Mod) -> Tuple[Mod, ...]:
    """Set moduli of Mod instances to gcd of moduli."""
    modulus = get_common_mod(*args)
    return tuple(Mod((_get_rem(x), modulus)) for x in args)
