# -*- coding: utf-8 -*-
"""Tricks for defining numeric types

Routine Listings
----------------
The following return mixin classes for defining numeric operators / functions:

convert_mixin
    Methods for conversion to `complex`, `float`, `int`.
ordered_mixin
    Comparison operators
mathops_mixin
    Arithmetic operators
rounder_mixin
    Rounding and modular arithmetic
number_mixin
    Union of all the above.
bitwise_mixin
    Bit-wise operators
integr_mixin
    Union of the two above.
imaths_mixin
    Inplace arithmetic operators
iround_mixin
    Inplace rounding and modular arithmetic
inumbr_mixin
    Union of the two above with `number_like_mixin`.
ibitws_mixin
    Inplace bit-wise operators
iintgr_mixin
    Union of the two above with `int_like_mixin`.

The following return functions to wrap methods and functions with type
conversion:

in_method_wrapper
    Decorator to wrap a method whose outputs do not require conversion
one_method_wrapper
    Decorator to wrap a method whose outputs do require conversion
inr_method_wrappers
    Decorator to turn one function into two magic methods - forward, reverse -
    whose outputs do not require conversion
opr_method_wrappers
    Decorator to turn one function into two magic methods - forward, reverse -
    whose outputs do require conversion
iop_method_wrapper
    Decorator to wrap an inplace magic method
function_wrappers
    Two decorators to wrap functions whose outputs do/do not require conversion
set_objclasses
    Finalises mehod wrappers after class definition.

Notes
-----
You should call `set_objclasses(class, cache)` after defining the `class`,
especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
The `__objclass__` attribute is needed to convert the outputs back.

To define all the abstracts for `numbers.Complex`:
    subclass `mathops_mixin(...)`,
    define `__complex__` or subclass `convert_mixin(...)`,
    define properties `real` and `imag`,
    define `conjugate()`.
To define all the abstracts for `numbers.Real`:
    subclass `mathops_mixin(...)`, `ordered_mixin(...)`, `rounder_mixin(...)`,
    define `__float__` or subclass `convert_mixin(...)`.
To define all the abstracts for `numbers.Integral`:
    subclass everything listed for `numbers.Real` and `bitwise_mixin(...)`,
    define `__int__` or subclass `convert_mixin(...)`.
"""
from __future__ import annotations
import builtins
import math
import operator
import typing
from collections.abc import Iterable
from functools import wraps
from numbers import Complex, Integral, Number, Real
from types import new_class
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar, Union

from .arg_tricks import default, default_non_eval
from .containers import tuplify, unseqify, InstanceOrSeq, Var, Val

# =============================================================================
# Wrapper helpers
# =============================================================================


def _multi_conv(single_conv: Type[Obj], result: InstanceOrSeq[Number],
                types: TypeArg = None) -> InstOrTup[Obj]:
    """helper for converting multiple outputs

    Parameters
    ----------
    single_conv : Callable
        Function used to convert a single output.
    result: types or Iterable[types]
        A single output, or an iterable of them
    types : Type or Tuple[Type] or None
        The types of output that should be converted. Others are passed

    Returns
    -------
    result
        The parameter `result` after conversion.
    """
    types = default(types, Number)

    def conv(val):
        if isinstance(val, types):
            return single_conv(val)
        return val

    return unseqify(tuple(map(conv, tuplify(result))))


def dummy_method(name: str) -> Callable:
    """Return a dummy function that always returns NotImplemented.

    This can be used to effectively remove an unwanted operator from a mixin,
    allowing fallbacks, whereas setting it to `None` would disallow it.
    """
    # pylint: disable=unused-argument
    def dummy(*args, **kwds):
        """Dummy function"""
        return NotImplemented
    dummy.__name__ = name
    return dummy


# -----------------------------------------------------------------------------
# Method wrapper helpers
# -----------------------------------------------------------------------------


def _implement_in(func: Func, args: Tuple[Other, ...], conv: Conv) -> Number:
    """Implement method for class, converting inputs, leaving outputs as is.

    Parameters
    ----------
    func : Callable
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv : Callable
        Function used to convert a tuple of inputs.
    method
        The resulting method. Its `__objclass__` property is used to convert
        the output.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    try:
        return func(*conv(args))
    except TypeError:
        return NotImplemented


def _implement_op(func: Func, args: Tuple[Other, ...], conv: Conv,
                  method: Operator, types: TypeArg = None) -> InstOrTup[Obj]:
    """Implement operator for class, converting inputs and outputs.

    Parameters
    ----------
    func : Callable
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv : Callable
        Function used to convert a tuple of inputs.
    method
        The resulting method. Its `__objclass__` property is used to convert
        the output.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    try:
        result = func(*conv(args))
    except TypeError:
        return NotImplemented
    else:
        return _multi_conv(method.__objclass__, result, types)


def _impl_mutable_iop(func: Func, args: Tuple[Other, ...], conv: Conv) -> Obj:
    """Implement mutable inplace operator for class

    Parameters
    ----------
    func : Callable
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv : Callable
        Function used to convert a tuple of inputs.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    try:
        func(*conv(args))
    except TypeError:
        return NotImplemented
    else:
        # if attr is mutable (like ndarrays), no need assign
        return args[0]


def _implement_iop(func: Func, args: Tuple[Other, ...], conv: Conv,
                   attr: Optional[str] = None) -> Obj:
    """Implement inplace operator for class

    Parameters
    ----------
    func : Callable[Number,... -> Number,...]
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv : Callable[Obj->Number]
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    if attr is None:
        return _impl_mutable_iop(func, args, conv)
    try:
        result = func(*conv(args))
    except TypeError:
        return NotImplemented
    else:
        # if attr is immutable (like numbers), no need to use inplace function
        setattr(args[0], attr, result)
        return args[0]


def _magic_name(func: Func, prefix: Optional[str] = None):
    """convert function name into magic method format"""
    prefix = default(prefix, '')
    return '__' + prefix + func.__name__.strip('_') + '__'


# =============================================================================
# Setting method __objclass__
# =============================================================================


def _add_set_objclass(meth: Union[Wrapped, Operator], cache: set) -> None:
    """content of method wrapper to set __objclass__ later.

    Parameters
    ----------
    meth
        The method(s) that will need to have __objclass__ set.
    cache : set
        Set storing methods that will need to have __objclass__ set later.
    """
    if isinstance(meth, Iterable):
        for mth in meth:
            _add_set_objclass(mth, cache)
    else:
        cache.add(meth)
        # meth.__objclass__ = type(None)


def set_objclasses(objclass: type, cache: set) -> None:
    """Set the __objclass__ attributes of methods.

    Must be called immediately after class definition.
    The `__objclass__` attribute is used here to convert outputs of methods.

    Parameters
    ----------
    objclass
        What we are setting `__objclass__` attributes to. It will be used to
        convert outputs.
    cache : set
        Set that stores methods that need to have __objclass__ set now.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    while cache:
        meth = cache.pop()
        meth.__objclass__ = objclass
    # for attr in dir(objclass):
    #     meth = getattr(objclass, attr)
    #     if getattr(meth, '__objclass__', None) is type(None):
    #         meth.__objclass__ = objclass


# =============================================================================
# Method wrappers
# =============================================================================
WRAPPER_ASSIGNMENTS_N = ('__doc__', '__annotations__', '__text_signature__')
WRAPPER_ASSIGNMENTS = ('__name__',) + WRAPPER_ASSIGNMENTS_N


def in_method_wrapper(conv: Conv, cache: set) -> Wrapper:
    """make wrappers for some class, converting inputs, leaving outputs as is.

    Make method, with inputs that are class/number & outputs that are number.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.

    Returns
    -------
    wrap_input : Callable
        A decorator for a method, so that the method's inputs are pre-converted
        and its outputs are left as is.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def wrap_input(func: Func) -> Wrapped:
        """Wrap method, so that the method's inputs are pre-converted and its
        outputs are left as is.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def method(*args) -> Number:
            return _implement_in(func, args, conv)

        method.__name__ = _magic_name(func)
        _add_set_objclass(method, cache)
        return method
    return wrap_input


def inr_methods_wrapper(conv: Conv, cache: set) -> Wrappers:
    """make wrappers for operator doublet: forward, reverse, converting inputs,
    leaving outputs as is.

    make methods, with inputs that are class/number & outputs that are numbers.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.

    Returns
    -------
    wrap_operators : Callable
        A decorator for a method, returning two methods, so that the method's
        inputs are preconverted and its outputs are left as is.
        The two methods are for forward and reverse operators.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def wrap_operators(func: Func) -> Tuple[Wrapped, Wrapped]:
        """Wrap operator set.

        A decorator for a method, returning two methods, so that the method's
        inputs are pre-converted and its outputs are left as is.
        The two methods are for forward, and reverse operators.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def method(*args) -> Number:
            return _implement_in(func, args, conv)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def rmethod(*args) -> Number:
            return _implement_in(func, reversed(args), conv)

        method.__name__ = _magic_name(func)
        rmethod.__name__ = _magic_name(func, 'r')
        _add_set_objclass((method, rmethod), cache)
        return method, rmethod

    return wrap_operators


def one_method_wrapper(conv: Conv, cache: set, types: TypeArg = None) -> OpWrapper:
    """make wrappers for some class, converting inputs and outputs.

    make method, with inputs that are class/number & outputs that are class.
    Conversion back to class uses method's `__objclass__`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    wrapper : Callable
        A decorator for a method, so that the method's inputs are pre-converted
        and its outputs are post-converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def wrap_in_out(func: Func) -> Operator:
        """Wrap method, so that the method's inputs are pre-converted and its
        outputs are post-converted.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def method(*args) -> Obj:
            return _implement_op(func, args, conv, method, types)

        method.__name__ = _magic_name(func)
        _add_set_objclass(method, cache)
        return method
    return wrap_in_out


def opr_methods_wrapper(conv: Conv, cache: set, types: TypeArg = None) -> OpsWrapper:
    """make wrapper for operator doublet: forward, reverse, converting inputs
    and outputs

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    wrap_operators : Callable
        A decorator for a method, returning two methods, so that the method's
        inputs are pre-converted and its outputs are post-converted.
        The two methods are for forward and reverse operators.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def wrap_operators(func: Func) -> Tuple[Operator, Operator]:
        """Wrap operator set.

        A decorator for a method, returning two methods, so that the method's
        inputs are pre-converted and its outputs are post-converted.
        The two methods are for forward, and reverse operators.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def method(*args) -> InstOrTup[Obj]:
            return _implement_op(func, args, conv, method, types)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def rmethod(*args) -> InstOrTup[Obj]:
            return _implement_op(func, reversed(args), conv, rmethod, types)

        method.__name__ = _magic_name(func)
        rmethod.__name__ = _magic_name(func, 'r')
        _add_set_objclass((method, rmethod), cache)
        return method, rmethod

    return wrap_operators


def iop_method_wrapper(conv: Conv, cache: set, attr: Optional[str] = None
                       ) -> OpWrapper:
    """make wrapper for inplace operator, immutable data

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.

    Returns
    -------
    wrap_inplace : Callable
        A decorator for a method, returning one method, so that its inputs are
        pre-converted and its outputs is assigned to the appropriate attribute
        if necessary. The method is for an inplace operator.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    prefix = default_non_eval(attr, lambda x: 'i', '')

    def wrap_inplace(func: Func) -> Operator:
        """Wrap inplace operator.

        A decorator for a method, returning an inplace operator method,
        so that the method's inputs are pre-converted and its output is
        assigned to the appropriate attribute if necessary.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def imethod(*args) -> Obj:
            return _implement_iop(func, args, conv, attr)

        imethod.__name__ = _magic_name(func, prefix)
        _add_set_objclass(imethod, cache)
        return imethod
    return wrap_inplace


# =============================================================================
# Function wrappers
# =============================================================================


def function_decorators(conv: Conv, class_out: Type[Obj], types: TypeArg = None
                        ) -> Tuple[Wrapper, OpWrapper]:
    """make decorators for conversting inputs/outputs from/to some class

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    class_out : type
        The class that outputs should be converted to.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    fun_input : Callable
        A decorator for a function, so that the function's inputs are
        pre-converted.
    fun_out : Callable
        A decorator for a function, so that the function's inputs are
        pre-converted and its outputs are post-converted.
    """
    def fun_input(func: Func) -> Wrapped:
        """Wrap function that returns another type
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args) -> Number:
            try:
                return func(*conv(args))
            except TypeError:
                return NotImplemented
        return wrapper

    def ext_fun(func: Func) -> Operator:
        """Wrap function to return class or Number
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args) -> InstOrTup[Obj]:
            try:
                result = func(*conv(args))
            except TypeError:
                return NotImplemented
            if any(isinstance(x, class_out) for x in args):
                return _multi_conv(class_out, result, types)
            return result
        return wrapper

    return fun_input, ext_fun


# =============================================================================
# Mixins
# =============================================================================
Convertible = Union[typing.SupportsComplex, typing.SupportsFloat,
                    typing.SupportsIndex, typing.SupportsInt]


def convert_mixin(conv: Conv, cache: set, names: Any = None
                  ) -> Type[Convertible]:
    """Mixin class for conversion to number types.

    Defines the functions `complex`, `float`, `int`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    names : Any, optional
        namespace with function attributes: {'complex', `float`, `int`}.
        By default `None -> builtins`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, builtins)
    method_input = in_method_wrapper(conv, cache)

    def exec_body(nsp: dict) -> None:
        """Mixin class for conversion to number types.
        """
        nsp['__complex__'] = method_input(names.complex)
        nsp['__float__'] = method_input(names.float)
        nsp['__int__'] = method_input(names.int)
        nsp['__index__'] = method_input(names.int)

    return new_class('ConvertibleMixin', exec_body=exec_body)


def ordered_mixin(conv: Conv, cache: set, names: Any = None) -> Type[Real]:
    """Mixin class for arithmetic comparisons.

    Defines all of the comparison operators, `==`, `!=`, `<`, `<=`, `>`, `>=`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    names : Any, optional
        namespace w/ function attributes: {'eq', `ne`, `lt`, 'le', `gt`, `ge`}.
        By default `None -> operator`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, operator)
    method_in = in_method_wrapper(conv, cache)

    # @total_ordering  # not nan friendly
    def exec_body(nsp: dict) -> None:
        """Mixin class to mimic real arithmetic number types.
        """
        nsp['__eq__'] = method_in(names.eq)
        nsp['__ne__'] = method_in(names.ne)
        nsp['__lt__'] = method_in(names.lt)
        nsp['__le__'] = method_in(names.le)
        nsp['__gt__'] = method_in(names.gt)
        nsp['__ge__'] = method_in(names.ge)

    return new_class('OrderedMixin', exec_body=exec_body)


def mathops_mixin(conv: Conv, cache: set, types: TypeArg = None,
                  names: Any = None) -> Type[Complex]:
    """Mixin class to mimic arithmetic number types.

    Defines the arithmetic operators `+`, `-`, `*`, `/`, `**`, `==`, `!=` and
    the functions `pow`, `abs`. Operators `//`, `%` are in `rounder_mixin`,
    operators `<`, `<=`, `>`, `>=` are in `ordered_mixin` and `<<`, `>>`, `&`,
    `^`, `|`, `~` are in `bit_twiddle_mixin`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None, optional
        The types of output that will be converted. By default `None -> Number`
    names : Any, optional
        namespace with function attributes: {'eq', `ne`, `add`, `sub`, `mul`,
        `truediv`, `pow`, `neg`, `pos`, `abs`}. By default `None -> operator`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, operator)
    method_in = in_method_wrapper(conv, cache)
    method = one_method_wrapper(conv, cache, types)
    ops = opr_methods_wrapper(conv, cache, types)

    def exec_body(nsp: dict) -> None:
        """Mixin class to mimic arithmetic number types.
        """
        nsp['__eq__'] = method_in(names.eq)
        nsp['__ne__'] = method_in(names.ne)
        nsp['__add__'], nsp['__radd__'] = ops(names.add)
        nsp['__sub__'], nsp['__rsub__'] = ops(names.sub)
        nsp['__mul__'], nsp['__rmul__'] = ops(names.mul)
        nsp['__truediv__'], nsp['__rtruediv__'] = ops(names.truediv)
        nsp['__pow__'], nsp['__rpow__'] = ops(names.pow)
        nsp['__neg__'] = method(names.neg)
        nsp['__pos__'] = method(names.pos)
        nsp['__abs__'] = method(names.abs)

    return new_class('ArithmeticMixin', exec_body=exec_body)


def rounder_mixin(conv: Conv, cache: set, types: TypeArg = None,
                  names: Any = None) -> Type[Real]:
    """Mixin class for rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, `round`,
    `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have `__objclass__` set.
    types : Type or Tuple[Type] or None, optional
        The types of output that are converted. By default `None -> Number`.
    names : Any or Tuple[Any, ...], optional
        Name spaces `(opspace, namespace, mathspace)`. A single namespace is
        expanded to 3. By default `None -> (operator, builtins, math)`.

        opspace : default - operator
            namespace with function attributes: {`floordiv`, `mod`}.
        defspace : default - builtins
            namespace with function attributes: {`divmod`, `round`}.
        mathspace : default - math
            namespace with function attributes: {`trunc`, `floor`, `ceil`}.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opsp, dfsp, masp = tuplify(default(names, (operator, builtins, math)), 3)
    method = one_method_wrapper(conv, cache, types)
    ops = opr_methods_wrapper(conv, cache, types)

    def exec_body(nsp: dict) -> None:
        """Mixin class for rounding/modular routines.
        """
        nsp['__floordiv__'], nsp['__rfloordiv__'] = ops(opsp.floordiv)
        nsp['__mod__'], nsp['__rmod__'] = ops(opsp.mod)
        nsp['__divmod__'], nsp['__rdivmod__'] = ops(dfsp.divmod)
        nsp['__round__'] = method(dfsp.round)
        nsp['__trunc__'] = method(masp.trunc)
        nsp['__floor__'] = method(masp.floor)
        nsp['__ceil__'] = method(masp.ceil)

    return new_class('RoundableMixin', exec_body=exec_body)


def bitwise_mixin(conv: Conv, cache: set, types: TypeArg = None,
                  names: Any = None) -> Type[Integral]:
    """Mixin class to mimic bit-string types.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None, optional
        The types of output that will be converted. By default `None -> Number`
    names : Any, optional
        namespace with function attributes: {'lshift', `rshift`, `and_`, `xor`,
        `or_`, `invert`}. By default `None -> operator`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, operator)
    method = one_method_wrapper(conv, cache, types)
    ops = opr_methods_wrapper(conv, cache, types)

    def exec_body(nsp: dict) -> None:
        """Mixin class to mimic bit-string types.
        """
        nsp['__lshift__'], nsp['__rlshift__'] = ops(names.lshift)
        nsp['__rshift__'], nsp['__rrshift__'] = ops(names.rshift)
        nsp['__and__'], nsp['__rand__'] = ops(names.and_)
        nsp['__xor__'], nsp['__rxor__'] = ops(names.xor)
        nsp['__or__'], nsp['__ror__'] = ops(names.or_)
        nsp['__invert__'] = method(names.invert)

    return new_class('BitTwiddleMixin', exec_body=exec_body)


# -----------------------------------------------------------------------------
# Inplace Mixins
# -----------------------------------------------------------------------------


def imaths_mixin(conv: Conv, cache: set, attr: Optional[str] = None,
                 names: Any = None) -> Type[Complex]:
    """Mixin class to mimic arithmetic number types.

    Defines the arithmetic operators `+`, `-`, `*`, `/`, `**`, `==`, `!=` and
    the functions `pow`, `abs`. Operators `//`, `%` are in `rounder_mixin`,
    operators `<`, `<=`, `>`, `>=` are in `ordered_mixin` and `<<`, `>>`, `&`,
    `^`, `|`, `~` are in `bit_twiddle_mixin`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    names : Any, optional
        namespace with function attributes:
        for immutable data - {`add`, `sub`, `mul`, `truediv`, `pow`};
        for mutable data - {`iadd`, `isub`, `imul`, `itruediv`, `ipow`}.
        By default `None -> operator`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, operator)
    iop = iop_method_wrapper(conv, cache, attr)
    prefix = 'i' if attr is None else ''

    def exec_body(nsp: dict) -> None:
        """Mixin class to mimic arithmetic number types.
        """
        nsp['__iadd__'] = iop(getattr(names, prefix + 'add'))
        nsp['__isub__'] = iop(getattr(names, prefix + 'sub'))
        nsp['__imul__'] = iop(getattr(names, prefix + 'mul'))
        nsp['__itruediv__'] = iop(getattr(names, prefix + 'truediv'))
        nsp['__ipow__'] = iop(getattr(names, prefix + 'pow'))

    return new_class('IArithmeticMixin', exec_body=exec_body)


def iround_mixin(conv: Conv, cache: set, attr: Optional[str] = None,
                 names: Any = None) -> Type[Real]:
    """Mixin class for rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, 'round',
    `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    names : Any, optional
        namespace with function attributes:
        for immutable data - {'floordiv', `mod`};
        for mutable data - {'ifloordiv', `imod`}.
        By default `None -> operator`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, operator)
    iop = iop_method_wrapper(conv, cache, attr)
    prefix = 'i' if attr is None else ''

    def exec_body(nsp: dict) -> None:
        """Mixin class for rounding/modular routines.
        """
        nsp['__ifloordiv__'] = iop(getattr(names, prefix + 'floordiv'))
        nsp['__imod__'] = iop(getattr(names, prefix + 'mod'))

    return new_class('IRoundableMixin', exec_body=exec_body)


def ibitws_mixin(conv: Conv, cache: set, attr: Optional[str] = None,
                 names: Any = None) -> Type[Integral]:
    """Mixin class to mimic bit-string types.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    names : Any, optional
        namespace with function attributes:
        for immutable data - {'lshift', `rshift`, `and_`, `xor`, `or_`};
        for mutable data - {'ilshift', `irshift`, `iand`, `ixor`, `ior`}.
        By default `None -> operator`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    names = default(names, operator)
    iop = iop_method_wrapper(conv, cache, attr)
    prefix = 'i' if attr is None else ''
    suffix = '' if attr is None else '_'

    def exec_body(nsp: dict) -> None:
        """Mixin class to mimic bit-string types.
        """
        nsp['__ilshift__'] = iop(getattr(names, prefix + 'lshift'))
        nsp['__irshift__'] = iop(getattr(names, prefix + 'rshift'))
        nsp['__iand__'] = iop(getattr(names, prefix + 'and' + suffix))
        nsp['__ixor__'] = iop(getattr(names, prefix + 'xor'))
        nsp['__ior__'] = iop(getattr(names, prefix + 'or' + suffix))

    return new_class('IBitTwiddleMixin', exec_body=exec_body)


# -----------------------------------------------------------------------------
# Combined Mixins
# -----------------------------------------------------------------------------


def number_mixin(conv: Conv, cache: set, types: TypeArg = None,
                 names: Any = None) -> Type[Real]:
    """Mixin class to mimic number types.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`, *except for*
    the bit-wise and the inplace operators.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None, optional
        The types of output that will be converted. By default `None -> Number`
    names : Any or Tuple[Any, ...], optional
        For `convert_mixin, ordered_mixin, mathops_mixin, rounder_mixin`.
         A single namespace is expanded to 4. By default `None`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    convspace, ordspace, mathspace, rndspace = tuplify(names, 4)
    bases = (convert_mixin(conv, cache, convspace),
             ordered_mixin(conv, cache, ordspace),
             mathops_mixin(conv, cache, types, mathspace),
             rounder_mixin(conv, cache, types, rndspace))
    return new_class('NumberLikeMixin', bases=bases)


def integr_mixin(conv: Conv, cache: set, types: TypeArg = None,
                 names: Any = None) -> Type[Integral]:
    """Mixin class to mimic integer types.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`, *except for*
    the inplace operators.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None, optional
        The types of output that will be converted. By default `None -> Number`
    names : Any or Tuple[Any, ...], optional
        For `number_mixin, bitwise_mixin`. A single namespace is expanded to 2.
        By default `None`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    numspace, bitspace = tuplify(names, 2)
    bases = (number_mixin(conv, cache, types, numspace),
             bitwise_mixin(conv, cache, types, bitspace))
    return new_class('IntegerLikeMixin', bases=bases)


def inumbr_mixin(conv: Conv, cache: set, types: TypeArg = None,
                 attr: Optional[str] = None, names: Any = None) -> Type[Real]:
    """Mixin class to mimic number types, with inplace operators.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`, *except for*
    the bit-wise operators.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None, optional
        The types of output that will be converted. By default `None -> Number`
    names : Any or Tuple[Any, ...], optional
        For `number_mixin, imaths_mixin, iround_mixin`. A single namespace is
        expanded to 3. By default `None`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    numspace, imspace, irndspace = tuplify(names, 3)
    bases = (number_mixin(conv, cache, types, numspace),
             imaths_mixin(conv, cache, attr, imspace),
             iround_mixin(conv, cache, attr, irndspace))
    return new_class('NumberLikeMixin', bases=bases)


def iintgr_mixin(conv: Conv, cache: set, types: TypeArg = None,
                 attr: Optional[str] = None, names: Any = None
                 ) -> Type[Integral]:
    """Mixin class to mimic integer types, with inplace operators.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    names : Tuple[Any, ...] or None, optional
        For `inumbr_mixin, bitwise_mixin, ibitwise_mixin`. A single namespace
        is expanded to 3. By default `None`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    inumspace, bitspace, ibitspace = tuplify(names, 3)
    bases = (inumbr_mixin(conv, cache, types, attr, inumspace),
             bitwise_mixin(conv, cache, types, bitspace),
             ibitws_mixin(conv, cache, attr, ibitspace))
    return new_class('IntegerLikeMixin', bases=bases)


# =============================================================================
# Typing aliases
# =============================================================================
Obj = TypeVar("Obj")
Other = Union[Number, Obj]
InstOrTup = Union[Var, Tuple[Var, ...]]
OfOneOrTwo = Union[Callable[[Var], Val], Callable[[Var, Var], Val]]
Conv = Callable[[Sequence[Obj]], Sequence[Number]]
Func = OfOneOrTwo[Number, InstOrTup[Number]]
Wrapped = OfOneOrTwo[Other, Number]
Operator = Callable[[Obj, Other], InstOrTup[Obj]]
Wrapper = Callable[[Func], Wrapped]
OpWrapper = Callable[[Func], Operator]
Wrappers = Callable[[Func], Tuple[Wrapped, Wrapped]]
OpsWrapper = Callable[[Func], Tuple[Operator, Operator]]
TypeArg = Optional[InstanceOrSeq[Type[Number]]]
