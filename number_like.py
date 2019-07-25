# -*- coding: utf-8 -*-
"""Tricks for manipulating integers
"""
import builtins
import math
import operator
from numbers import Real
from collections.abc import Iterable
from functools import wraps
from .arg_tricks import default

# =============================================================================
# %%* Wrapper helpers
# =============================================================================

WRAPPER_ASSIGNMENTS_N = ('__doc__', '__annotations__', '__text_signature__')
WRAPPER_ASSIGNMENTS = ('__name__',) + WRAPPER_ASSIGNMENTS_N

# =============================================================================
# %%* Function wrappers
# =============================================================================


def _multi_conv(single_conv, result):
    """helper for converting multiple outputs"""
    if isinstance(result, Real):
        return single_conv(result)
    if not isinstance(result, Iterable):
        return NotImplemented
    return tuple(single_conv(res) for res in result)


def make_fn_wrappers(conv_in, class_out):
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
                return _multi_conv(class_out, result)
            return result
        return wrapper

    return fun_input, ext_fun


# =============================================================================
# %%* Method wrapper helpers
# =============================================================================


def _implement_op(func, args, conv_in, method):
    """Implement operator for class
    """
    try:
        result = func(*conv_in(args))
        return _multi_conv(method.__objclass__, result)
    except TypeError:
        return NotImplemented


def _implement_iop(func, args, conv_in, attr):
    """Implement inplace operator for class
    """
    try:
        setattr(args[0], attr, func(*conv_in(args)))
    except TypeError:
        return NotImplemented
    return args[0]


_to_set_objclass = set()


def _add_set_objclass(meth, cache):
    if isinstance(meth, tuple):
        for m in meth:
            _add_set_objclass(m, cache)
    else:
        cache.add(meth)


def _method_factory(factory):
    """set __objclass__"""
    @wraps(factory)
    def wrapped_factory(func, *args, **kwds):
        methods = factory(func, *args, **kwds)
        _add_set_objclass(methods, _to_set_objclass)
        return methods
    return wrapped_factory


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


def make_in_method_wrapper(conv_in, cache=None):
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


def make_out_method_wrappers(conv_in, attr, cache=None):
    """make wrappers for some class

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses method's `__objclass__`.
    """
    @_method_factory_factory(cache)
    def method(func):
        """Wrap method that returns class
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper)
        return wrapper

    @_method_factory_factory(cache)
    def operator_set(func):
        """Wrap operator set
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def rwrapper(*args):
            return _implement_op(func, reversed(args), conv_in, rwrapper)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def iwrapper(*args):
            return _implement_iop(func, args, conv_in, attr)

        wrapper.__name__ = '__' + func.__name__.strip('_') + '__'
        rwrapper.__name__ = '__r' + func.__name__.strip('_') + '__'
        iwrapper.__name__ = '__i' + func.__name__.strip('_') + '__'

        return wrapper, rwrapper, iwrapper

    return method, operator_set


def make_method_wrappers(conv_in, attr, cache=None):
    """make wrappers for some class

    make methods, with inputs & outputs that are class/number.
    Conversion back to class uses method's `__objclass__`.
    """
    return ((make_in_method_wrapper(conv_in, cache),)
            + make_out_method_wrappers(conv_in, attr, cache))


# =============================================================================
# %%* Mixins
# =============================================================================


def arithmetic_ops_mixin(conv_in, attr, namespace, cache=None):
    """Mixin class to mimic arithmetic number types.
    """
    method_input, method, ops = make_method_wrappers(conv_in, attr, cache)

    # @total_ordering  # not nan friendly
    class ArithmeticOpsMixin(Real):
        """Mixin class to mimic arithmetic number types.
        """
        __eq__ = method_input(namespace.__eq__)
        __ne__ = method_input(namespace.__ne__)
        __lt__ = method_input(namespace.__lt__)
        __le__ = method_input(namespace.__le__)
        __gt__ = method_input(namespace.__gt__)
        __ge__ = method_input(namespace.__ge__)
        __add__, __radd__, __iadd__ = ops(namespace.add)
        __sub__, __rsub__, __isub__ = ops(namespace.sub)
        __mul__, __rmul__, __imul__ = ops(namespace.mul)
        __truediv__, __rtruediv__, __itruediv__ = ops(namespace.truediv)
        __floordiv__, __rfloordiv__, __ifloordiv__ = ops(namespace.floordiv)
        __mod__, __rmod__, __imod__ = ops(namespace.mod)
        __pow__, __rpow__, __ipow__ = ops(namespace.pow)
        __neg__ = method(namespace.neg)
        __pos__ = method(namespace.pos)
        __abs__ = method(namespace.abs)
        # exception
        __divmod__, __rdivmod__ = ops(divmod)[:2]

    return ArithmeticOpsMixin


def bit_twiddle_mixin(conv_in, attr, namespace, cache=None):
    """Mixin class to mimic bit-field types.
    """
    method, ops = make_out_method_wrappers(conv_in, attr, cache)

    class BitTwiddleMixin(Real):
        """Mixin class to mimic bit-field types.
        """
        __lshift__, __rlshift__, __ilshift__ = ops(namespace.lshift)
        __rshift__, __rrshift__, __irshift__ = ops(namespace.rshift)
        __and__, __rand__, __iand__ = ops(namespace.and_)
        __xor__, __rxor__, __ixor__ = ops(namespace.xor)
        __or__, __ror__, __ior__ = ops(namespace.or_)
        __invert__ = method(namespace.invert)

    return BitTwiddleMixin


def convertible_mixin(conv_in, namespace, cache=None):
    """Mixin class for conversion to number types.
    """
    method_input = make_in_method_wrapper(conv_in, cache)

    class ConvertibleMixin(Real):
        """Mixin class for conversion to number types.
        """
        __complex__ = method_input(namespace.complex)
        __float__ = method_input(namespace.float)
        __int__ = method_input(namespace.int)
        __index__ = method_input(namespace.int)

    return ConvertibleMixin


def roundable_mixin(conv_in, namespace, mathspace, cache=None):
    """Mixin class for conversion to number types.
    """
    method = make_out_method_wrappers(conv_in, '', cache)[0]

    class RoundableMixin(Real):
        """Mixin class for conversion to number types.
        """
        __round__ = method(namespace.round)
        __trunc__ = method(mathspace.trunc)
        __floor__ = method(mathspace.floor)
        __ceil__ = method(mathspace.ceil)

    return RoundableMixin


def number_like_mixin(conv_in, attr, cache=None):
    """Mixin class to mimic number types.
    """
    method_input, method, _ = make_method_wrappers(conv_in, attr, cache)

    class NumberLikeMixin(arithmetic_ops_mixin(conv_in, attr, operator, cache),
                          bit_twiddle_mixin(conv_in, attr, operator, cache),
                          convertible_mixin(conv_in, builtins, cache),
                          roundable_mixin(conv_in, builtins, math, cache)):
        """Mixin class to mimic number types.
        """

    return NumberLikeMixin
