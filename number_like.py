# -*- coding: utf-8 -*-
"""Tricks for manipulating integers
"""
import builtins
import math
import operator
from numbers import Complex, Real, Integral, Number
from collections.abc import Iterable
from functools import wraps
from .arg_tricks import default

# =============================================================================
# %%* Wrapper helpers
# =============================================================================

WRAPPER_ASSIGNMENTS_N = ('__doc__', '__annotations__', '__text_signature__')
WRAPPER_ASSIGNMENTS = ('__name__',) + WRAPPER_ASSIGNMENTS_N


def _multi_conv(single_conv, result, types=None):
    """helper for converting multiple outputs"""
    types = default(types, Number)
    if isinstance(result, types):
        return single_conv(result)
    if not isinstance(result, Iterable):
        return NotImplemented
    return tuple(single_conv(res) for res in result)

# =============================================================================
# %%* Method wrapper helpers
# =============================================================================


def _implement_op(func, args, conv_in, method, types=None):
    """Implement operator for class
    """
    try:
        result = func(*conv_in(args))
        return _multi_conv(method.__objclass__, result, types)
    except TypeError:
        return NotImplemented


def _implement_iop(func, args, conv_in, attr):
    """Implement inplace operator for class
    """
    try:
        result = func(*conv_in(args))
    except TypeError:
        return NotImplemented
    else:
        setattr(args[0], attr, result)
        return args[0]


def _magic_name(func, prefix=''):
    """convert function name into magic method format"""
    return '__' + prefix + func.__name__.strip('_') + '__'


_to_set_objclass = set()


def _add_set_objclass(meth, cache):
    if isinstance(meth, tuple):
        for m in meth:
            _add_set_objclass(m, cache)
    else:
        cache.add(meth)


def _method_factory_factory(cache=None):
    """set __objclass__"""
    def method_factory(factory):
        @wraps(factory)
        def wrapped_factory(func, *args, **kwds):
            methods = factory(func, *args, **kwds)
            _add_set_objclass(methods, default(cache, _to_set_objclass))
            return methods
        return wrapped_factory
    return method_factory


def set_objclasses(objclass, cache=None):
    """Set the __objclass__ properties of methods.

    Must be called immediately after class definition.
    """
    cache = default(cache, _to_set_objclass)
    while cache:
        meth = cache.pop()
        meth.__objclass__ = objclass


# =============================================================================
# %%* Method wrappers
# =============================================================================


def in_method_wrapper(conv_in, cache=None):
    """make wrappers for some class

    make method, with inputs that are class/number & outputs that are number.
    When conversion back to class is required, it uses method's `__objclass__`.
    """
    @_method_factory_factory(cache)
    def method_input(func):
        """Wrap method that returns another type
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args):
            try:
                return func(*conv_in(args))
            except TypeError:
                return NotImplemented
        return wrapper
    return method_input


def one_method_wrapper(conv_in, cache=None, types=None):
    """make wrappers for some class

    make method, with inputs that are class/number & outputs that are class.
    Conversion back to class uses method's `__objclass__`.
    """
    @_method_factory_factory(cache)
    def method(func):
        """Wrap method that returns class
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper, types)
        return wrapper
    return method


def out_method_wrappers(conv_in, attr, cache=None, types=None):
    """make wrappers for some class

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.
    """
    @_method_factory_factory(cache)
    def operator_set(func, types=None):
        """Wrap operator set
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper, types)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def rwrapper(*args):
            return _implement_op(func, reversed(args), conv_in, rwrapper,
                                 types)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def iwrapper(*args):
            return _implement_iop(func, args, conv_in, attr)

        wrapper.__name__ = _magic_name(func)
        rwrapper.__name__ = _magic_name(func, 'r')
        iwrapper.__name__ = _magic_name(func, 'i')

        return wrapper, rwrapper, iwrapper

    return one_method_wrapper(conv_in, cache, types), operator_set


def method_wrappers(conv_in, attr, cache=None, types=None):
    """make wrappers for some class

    make methods, with inputs & outputs that are class/number.
    Conversion back to class uses method's `__objclass__`.
    """
    return ((in_method_wrapper(conv_in, cache),)
            + out_method_wrappers(conv_in, attr, cache, types))


# =============================================================================
# %%* Function wrappers
# =============================================================================


def function_wrappers(conv_in, class_out, types=None):
    """make wrappers for some class

    make functions, with inputs & outputs that are class or number
    """
    def fun_input(func):
        """Wrap function that returns another type
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args):
            try:
                return func(*conv_in(args))
            except TypeError:
                return NotImplemented
        return wrapper

    def ext_fun(func):
        """Wrap function to return class or Number
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args):
            try:
                result = func(*conv_in(args))
            except TypeError:
                return NotImplemented
            if any(isinstance(x, class_out) for x in args):
                return _multi_conv(class_out, result, types)
            return result
        return wrapper

    return fun_input, ext_fun


# =============================================================================
# %%* Mixins
# =============================================================================


def convertible_mixin(conv_in, cache=None, namespace=builtins):
    """Mixin class for conversion to number types.
    """
    method_input = in_method_wrapper(conv_in, cache)

    class ConvertibleMixin(Complex):
        """Mixin class for conversion to number types.
        """
        __complex__ = method_input(namespace.complex)
        __float__ = method_input(namespace.float)
        __int__ = method_input(namespace.int)
        __index__ = method_input(namespace.int)

    return ConvertibleMixin


def arithmetic_mixin(conv_in, attr, cache=None, types=None, opspace=operator):
    """Mixin class to mimic arithmetic number types.
    """
    method_in, method, ops = method_wrappers(conv_in, attr, cache, types)

    # @total_ordering  # not nan friendly
    class ArithmeticMixin(Complex):
        """Mixin class to mimic arithmetic number types.
        """
        __eq__ = method_in(opspace.__eq__)
        __ne__ = method_in(opspace.__ne__)
        __add__, __radd__, __iadd__ = ops(opspace.add)
        __sub__, __rsub__, __isub__ = ops(opspace.sub)
        __mul__, __rmul__, __imul__ = ops(opspace.mul)
        __truediv__, __rtruediv__, __itruediv__ = ops(opspace.truediv)
        __pow__, __rpow__, __ipow__ = ops(opspace.pow)
        __neg__ = method(opspace.neg)
        __pos__ = method(opspace.pos)
        __abs__ = method(opspace.abs)

    return ArithmeticMixin


def ordered_mixin(conv_in, cache=None, opspace=operator):
    """Mixin class to mimic arithmetic number types.
    """
    method_in = in_method_wrapper(conv_in, cache)

    # @total_ordering  # not nan friendly
    class OrderedMixin(Real):
        """Mixin class to mimic real arithmetic number types.
        """
        __eq__ = method_in(opspace.__eq__)
        __ne__ = method_in(opspace.__ne__)
        __lt__ = method_in(opspace.__lt__)
        __le__ = method_in(opspace.__le__)
        __gt__ = method_in(opspace.__gt__)
        __ge__ = method_in(opspace.__ge__)

    return OrderedMixin


def roundable_mixin(conv_in, attr, cache=None, types=None,
                    opspace=operator, namespace=builtins, mathspace=math):
    """Mixin class for conversion to number types.
    """
    method, ops = out_method_wrappers(conv_in, attr, cache, types)

    class RoundableMixin(Real):
        """Mixin class for conversion to number types.
        """
        __floordiv__, __rfloordiv__, __ifloordiv__ = ops(opspace.floordiv)
        __mod__, __rmod__, __imod__ = ops(opspace.mod)
        __divmod__, __rdivmod__ = ops(namespace.divmod)[:2]
        __round__ = method(namespace.round)
        __trunc__ = method(mathspace.trunc)
        __floor__ = method(mathspace.floor)
        __ceil__ = method(mathspace.ceil)

    return RoundableMixin


def bit_twiddle_mixin(conv_in, attr, cache=None, types=None, opspace=operator):
    """Mixin class to mimic bit-field types.
    """
    method, ops = out_method_wrappers(conv_in, attr, cache, types)

    class BitTwiddleMixin(Integral):
        """Mixin class to mimic bit-field types.
        """
        __lshift__, __rlshift__, __ilshift__ = ops(opspace.lshift)
        __rshift__, __rrshift__, __irshift__ = ops(opspace.rshift)
        __and__, __rand__, __iand__ = ops(opspace.and_)
        __xor__, __rxor__, __ixor__ = ops(opspace.xor)
        __or__, __ror__, __ior__ = ops(opspace.or_)
        __invert__ = method(opspace.invert)

    return BitTwiddleMixin


def number_like_mixin(conv_in, attr, cache=None, types=None):
    """Mixin class to mimic number types.
    """

    class NumberLikeMixin(convertible_mixin(conv_in, cache),
                          arithmetic_mixin(conv_in, attr, cache, types),
                          ordered_mixin(conv_in, cache),
                          roundable_mixin(conv_in, attr, cache, types),
                          bit_twiddle_mixin(conv_in, attr, cache, types),
                          ):
        """Mixin class to mimic number types.
        """

    return NumberLikeMixin
