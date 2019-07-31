# -*- coding: utf-8 -*-
"""Modular arithmetic
"""
from __future__ import annotations
import operator
from functools import wraps
from itertools import chain
from typing import Callable, Tuple, ClassVar
from sl_py_tools.containers import tuplify
from sl_py_tools import number_like as nl
from sl_py_tools import integer_tricks as ig
from sl_py_tools.integer_tricks import eint, inf, nan, Eint

# =============================================================================
# %%* Operator helpers
# =============================================================================


def _wrap_one_out(func: Callable, name: str = '') -> Callable:
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


def _wrap_one_out_mod(func: Callable, name: str = '') -> Callable:
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


def _wrap_no_out(func: Callable, name: str = '') -> Callable:
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

    Returns NotImplemented
    """
    @wraps(func)
    def wrapper(*args):
        """wrapper delete method"""
        return NotImplemented
    if name:
        wrapper.__name__ = name
    return wrapper


# -----------------------------------------------------------------------------
# %%* Operator namespace
# -----------------------------------------------------------------------------


class Operators:
    """Namespace for operators"""
    complex = _wrap_no_out(complex)
    float = _wrap_no_out(float)
    int = _wrap_no_out(int)
    eq = _wrap_no_out(operator.eq)
    ne = _wrap_no_out(operator.ne)
    add = _wrap_one_out(operator.add)
    sub = _wrap_one_out(operator.sub)
    mul = _wrap_one_out(operator.mul)
    floordiv = _wrap_one_out_mod(ig.divm, 'floordiv')
    # floordiv = _wrap_one_out(operator.floordiv)
    truediv = _unwrap(operator.truediv)
    pow = _wrap_one_out(operator.pow)
    pos = _wrap_one_out(operator.pos)
    neg = _wrap_one_out(operator.neg)
    abs = _wrap_one_out(operator.abs)
    invert = _wrap_one_out_mod(ig.invert)


# -----------------------------------------------------------------------------
# %%* Intersection function
# -----------------------------------------------------------------------------


def and_(val1, mod1, val2, mod2):
    """Intersecton of congruence classes"""
    delta = ig.gcd(mod1, mod2)
    dlt1, dlt2 = mod1 // delta, mod2 // delta
    if (val1 - val2) % delta:
        return (nan, delta * dlt1 * dlt2)
    excess = (val1 - val2) // delta
    exc1, exc2 = excess % dlt1, -excess % dlt2
    # this needs modular arithmetic...
    dmult1 = ig.divm(exc2, dlt1, dlt2)
    dmult2 = ig.divm(exc1, dlt2, dlt1)
    dexcess = excess + dmult1 * dlt1 - dmult2 * dlt2
    if dexcess > 0:
        mult1, mult2 = dmult1, dmult2 + dexcess // (dlt1 * dlt2)
    else:
        mult1, mult2 = dmult1 - dexcess // (dlt1 * dlt2), dmult2
    overlap = int(val1 + mult1 * mod1)
    assert overlap == val2 + mult2 * mod2
    return (overlap, delta * dlt1 * dlt2)


# =============================================================================
# %%* Converters & wrappers
# =============================================================================


def _get_rem(arg):
    """Convert an input from Mod.remainder to eint"""
    if isinstance(arg, Mod):
        return arg.remainder
    return arg


def _get_mod(arg):
    """Convert an input from Mod.modulus to eint"""
    if isinstance(arg, Mod):
        return arg.modulus
    return inf


def _convert_inputs(args):
    """Convert inputs from Mod to eint"""
    if Mod.strict:
        return _convert_inputs_strict(args)
    modulus = get_common_mod(*args)
    return [_get_rem(x) for x in args] + [modulus]


def _convert_inputs_strict(args):
    """Convert inputs from Mod to eint"""
    moduli = [_get_mod(x) for x in args]
    if any(x != moduli[0] for x in moduli):
        raise ValueError(f'Combining different moduli: {moduli}')
    return [_get_rem(x) for x in args] + moduli[:1]


def _flatten_inputs(args):
    """Convert inputs from Mod to pairs of eint"""
    return list(chain.from_iterable((_get_rem(x), _get_mod(x)) for x in args))


_method_cache = set()
_mth = nl.one_method_wrapper(_flatten_inputs, _method_cache, tuple)
_opf = nl.opr_method_wrappers(_flatten_inputs, _method_cache, tuple)
_opr = nl.opr_method_wrappers(_convert_inputs, _method_cache, tuple)
_Cnv = nl.convert_mixin(_convert_inputs, _method_cache, Operators)
_Ops = nl.mathops_mixin(_convert_inputs, _method_cache, tuple, Operators)


# =============================================================================
# %%* Mod class
# =============================================================================


class Mod(_Cnv, _Ops):
    """Class implementing Modular Arithmetic.

    Parameters
    ----------
    value : Eint
        A value from the set being represented.
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
    remainder: eint
    modulus: eint
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
        return not (self == other)

    def __contains__(self, other) -> bool:
        if isinstance(other, Mod):
            return self & other == other
        return self == other

    def __repr__(self) -> str:
        return f"Mod({self.remainder}, {self.modulus})"

    def __str__(self) -> str:
        return f"{self.remainder} (mod {self.modulus})"

    @property
    def data(self) -> Tuple[eint, ...]:
        """The tuple (remainder, modulus)"""
        return (self.remainder, self.modulus)

    @data.setter
    def data(self, value: Tuple[Eint, ...]):
        self.remainder, self.modulus = [eint(x) for x in value]
        self.remainder %= self.modulus

    @property
    def real(self) -> Mod:
        return +self

    @property
    def imag(self) -> Mod:
        return Mod((0, self.modulus))

    def conjugate(self) -> Mod:
        """"""
        return +self


nl.set_objclasses(Mod, _method_cache)
# =============================================================================
# %%* Common functions
# =============================================================================


def get_common_mod(*args: Mod) -> eint:
    """Find gcd of moduli of Mod instances."""
    return ig.gcdn(*[_get_mod(x) for x in args])


def set_common_mod(*args: Mod) -> Tuple[Mod, ...]:
    """Set moduli of Mod instances to gcd of moduli."""
    modulus = get_common_mod(*args)
    return tuple(Mod((_get_rem(x), modulus)) for x in args)
