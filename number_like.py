# -*- coding: utf-8 -*-
"""Tricks for defining numeric types

Routine Listings
----------------
The following return mixin classes for defining numeric operators / functions:

convertible_mixin
    Methods for conversion to `complex`, `float`, `int`.
arithmetic_mixin
    Arithmetic operators
ordered_mixin
    Comparison operators
roundable_mixin
    Rounding and modular arithmetic
bit_twiddle_mixin
    Bit-wise operators
number_like_mixin
    Union of the above.

The following return functions to wrap methods and functions with type
conversion:

in_method_wrapper
    Decorator to wrap a method whose outputs do not require conversion
one_method_wrapper
    Decorator to wrap a method whose outputs do require conversion
opr_method_wrappers
    Turns one function into two magic methods - forward, reverse
iop_method_wrapper
    Decorator to wrap an inplace magic method
ops_method_wrappers
    Turns one function into three magic methods - forward, reverse, inplace
function_wrappers
    Two decorators to wrap functions whose outputs do/do not require conversion
set_objclasses
    Finalises mehod wrappers after class definition.

Separate `opr_method_wrappers` and `iop_method_wrapper` assumes mutable data
and use of a special inplace function which takes care of assignment.
Combined `ops_method_wrappers` does not allow use of a special inplace
function, which would not have made sense for immutable data anyway.

Notes
-----
You should call `set_objclasses(class, cache)` after the `class` definition,
especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
`out_method_wrappers` or the last two outputs of `method_wrappers`. The
`__objclass__` attribute is needed to convert the outputs back.
"""
import builtins
import math
import operator
from numbers import Complex, Real, Integral, Number
from collections.abc import Iterable, Callable
from functools import wraps
from .arg_tricks import default

# =============================================================================
# %%* Wrapper helpers
# =============================================================================

WRAPPER_ASSIGNMENTS_N = ('__doc__', '__annotations__', '__text_signature__')
WRAPPER_ASSIGNMENTS = ('__name__',) + WRAPPER_ASSIGNMENTS_N


def _multi_conv(single_conv: Callable, result, types=None):
    """helper for converting multiple outputs

    Parameters
    ----------
    single_conv : Callable
        Function used to convert a single output.
    result: types or Iterable[types]
        A single output, or an iterable of them
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    result
        The parameter `result` after conversion.
    """
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

    Parameters
    ----------
    func : Callable
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv_in : Callable
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
        result = func(*conv_in(args))
        return _multi_conv(method.__objclass__, result, types)
    except TypeError:
        return NotImplemented


def _implement_iop(func, args, conv_in, attr):
    """Implement inplace operator for class

    Parameters
    ----------
    func : Callable
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is update for inplace operations.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    try:
        result = func(*conv_in(args))
    except TypeError:
        return NotImplemented
    else:
        # if attr is immutable (like numbers), no need to use inplace function
        setattr(args[0], attr, result)
        return args[0]


def _implement_mutable_iop(func, args, conv_in):
    """Implement mutable inplace operator for class

    Parameters
    ----------
    func : Callable
        The function being wrapped.
    args : Tuple[types]
        The inputs to the function being wrapped.
    conv_in : Callable
        Function used to convert a tuple of inputs.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    try:
        func(*conv_in(args))
    except TypeError:
        return NotImplemented
    else:
        # if attr is mutable (like ndarrays), no need assign
        return args[0]


def _magic_name(func, prefix=''):
    """convert function name into magic method format"""
    return '__' + prefix + func.__name__.strip('_') + '__'


# =============================================================================
# %%* Setting method __objclass__
# =============================================================================
_to_set_objclass = set()


def _add_set_objclass(meth, cache: set):
    """content of method wrapper to set __objclass__ later.

    Parameters
    ----------
    meth
        The method(s) that will need to have __objclass__ set.
    cache : set
        Set storing methods that will need to have __objclass__ set later.
    """
    if isinstance(meth, tuple):
        for m in meth:
            _add_set_objclass(m, cache)
    else:
        cache.add(meth)


def _objclass_decorator_factory(cache=None):
    """wrap method to set __objclass__ later

    Parameters
    ----------
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.

    Returns
    -------
    objclass_decorator : Callable
        A function that returns a decorator for another method factory
    """
    def objclass_decorator(factory):
        """A function that returns a decorator for another method factory
        """
        @wraps(factory)
        def wrapped_factory(func, *args, **kwds):
            methods = factory(func, *args, **kwds)
            _add_set_objclass(methods, default(cache, _to_set_objclass))
            return methods
        return wrapped_factory
    return objclass_decorator


def set_objclasses(objclass, cache=None):
    """Set the __objclass__ attributes of methods.

    Must be called immediately after class definition.
    The `__objclass__` attribute is used here to convert outputs of methods.

    Parameters
    ----------
    objclass
        What we are setting `__objclass__` attributes to. It will be used to
        convert outputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
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

    Make method, with inputs that are class/number & outputs that are number.
    When conversion back to class is required, it uses method's `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.

    Returns
    -------
    method_input : Callable
        A decorator for a method, so that the method's inputs are preconverted.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    @_objclass_decorator_factory(cache)
    def method_input(func):
        """Wrap method that returns another type
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            try:
                return func(*conv_in(args))
            except TypeError:
                return NotImplemented
        wrapper.__name__ = _magic_name(func)
        return wrapper
    return method_input


def one_method_wrapper(conv_in, cache=None, types=None):
    """make wrappers for some class

    make method, with inputs that are class/number & outputs that are class.
    Conversion back to class uses method's `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    method_out : Callable
        A decorator for a method, so that the method's inputs are preconverted
        and its outputs are post converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    @_objclass_decorator_factory(cache)
    def method(func):
        """Wrap method, so that the method's inputs are preconverted
        and its outputs are post converted.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper, types)
        wrapper.__name__ = _magic_name(func)
        return wrapper
    return method


def opr_method_wrappers(conv_in, cache=None, types=None):
    """make wrappers for operator doublet: forward, reverse,

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    operators_out : Callable
        A decorator for a method, returning three methods, so that the method's
        inputs are preconverted and its outputs are post converted.
        The three methods are for forward, reverse, and inplace operators.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    @_objclass_decorator_factory(cache)
    def operator_set(func, types=None):
        """Wrap operator set.

        A decorator for a method, returning three methods, so that the method's
        inputs are preconverted and its outputs are post converted.
        The three methods are for forward, reverse, and inplace operators.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper, types)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def rwrapper(*args):
            return _implement_op(func, reversed(args), conv_in, rwrapper,
                                 types)

        wrapper.__name__ = _magic_name(func)
        rwrapper.__name__ = _magic_name(func, 'r')

        return wrapper, rwrapper

    return operator_set


def iop_method_wrapper(conv_in, cache=None):
    """make wrapper for inplace operator

    make inplace methods, with inputs that are class/number

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.

    Returns
    -------
    operators_out : Callable
        A decorator for a method, returning one inplace method, so that the
        method's inputs are preconverted

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    @_objclass_decorator_factory(cache)
    def operator_set(func):
        """Wrap operator set.

        A decorator for an inplace method, so that the method's inputs are
        preconverted. func should be an inplace function
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def iwrapper(*args):
            return _implement_mutable_iop(func, args, conv_in)

        iwrapper.__name__ = _magic_name(func)
        return iwrapper
    return operator_set


def ops_method_wrappers(conv_in, attr, cache=None, types=None):
    """make wrappers for operator triplet: forward, reverse, inplace

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is update for inplace operations.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    operators_out : Callable
        A decorator for a method, returning three methods, so that the method's
        inputs are preconverted and its outputs are post converted.
        The three methods are for forward, reverse, and inplace operators.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    def operator_set(func, types=None):
        """Wrap operator set.

        A decorator for a method, returning three methods, so that the method's
        inputs are preconverted and its outputs are post converted.
        The three methods are for forward, reverse, and inplace operators.
        """
        opr_methods = opr_method_wrappers(conv_in, cache, types)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def iwrapper(*args):
            return _implement_iop(func, args, conv_in, attr)

        iwrapper.__name__ = _magic_name(func, 'i')

        return opr_methods(func, types) + (iwrapper,)
    return operator_set


# =============================================================================
# %%* Function wrappers
# =============================================================================


def function_wrappers(conv_in, class_out, types=None):
    """make wrappers for some class

    make functions, with inputs & outputs that are class or number

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    class_out : type
        The class that outputs should be converted to.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Returns
    -------
    fun_input : Callable
        A decorator for a function, so that the function's inputs are
        preconverted.
    fun_out : Callable
        A decorator for a function, so that the function's inputs are
        preconverted and its outputs are post converted.
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

    Defines the functions `complex`, `float`, `int`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    namespace : default - builtins
        namespace with function attributes: 'complex', `float`, `int`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
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

    Defines the arithmetic operators `+`, `-`, `*`, `/`, `**`, `==`, `!=` and
    the functions `pow`, `abs`. Operators `//`, `%` are in `roundable_mixin`,
    operators `<`, `<=`, `>`, `>=` are in `ordered_mixin` and `<<`, `>>`, `&`,
    `^`, `|`, `~` are in `bit_twiddle_mixin`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is update for inplace operations.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    opspace : default - operator
        namespace with function attributes: 'eq', `ne`, `add`, `sub`, `mul`,
        `truediv`, `pow`, `neg`, `pos`, `abs`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method_in = in_method_wrapper(conv_in, cache)
    method = one_method_wrapper(conv_in, cache, types)
    ops = ops_method_wrappers(conv_in, attr, cache, types)

    # @total_ordering  # not nan friendly
    class ArithmeticMixin(Complex):
        """Mixin class to mimic arithmetic number types.
        """
        __eq__ = method_in(opspace.eq)
        __ne__ = method_in(opspace.ne)
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
    """Mixin class for arithmetic comparisons.

    Defines all of the comparison operators, `==`, `!=`, `<`, `<=`, `>`, `>=`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    opspace : default - operator
        namespace with function attributes: 'eq', `ne`, `lt`, 'le', `gt`, `ge`.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method_in = in_method_wrapper(conv_in, cache)

    # @total_ordering  # not nan friendly
    class OrderedMixin(Real):
        """Mixin class to mimic real arithmetic number types.
        """
        __eq__ = method_in(opspace.eq)
        __ne__ = method_in(opspace.ne)
        __lt__ = method_in(opspace.lt)
        __le__ = method_in(opspace.le)
        __gt__ = method_in(opspace.gt)
        __ge__ = method_in(opspace.ge)

    return OrderedMixin


def roundable_mixin(conv_in, attr, cache=None, types=None,
                    opspace=operator, namespace=builtins, mathspace=math):
    """Mixin class for rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, 'round',
    `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is update for inplace operations.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    opspace : default - operator
        namespace with function attributes: 'floordiv', `mod`.
    namespace : default - builtins
        namespace with function attributes: 'divmod', `round`.
    mathspace : default - math
        namespace with function attributes: 'trunc', `floor`, `ceil`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method = one_method_wrapper(conv_in, cache, types)
    ops = ops_method_wrappers(conv_in, attr, cache, types)

    class RoundableMixin(Real):
        """Mixin class for rounding/modular routines.
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
    """Mixin class to mimic bit-string types.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is update for inplace operations.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    opspace : default - operator
        namespace with function attributes: 'lshift', `rshift`, `and_`, `xor`,
        `or_`, `invert`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method = one_method_wrapper(conv_in, cache, types)
    ops = ops_method_wrappers(conv_in, attr, cache, types)

    class BitTwiddleMixin(Integral):
        """Mixin class to mimic bit-string types.
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

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is update for inplace operations.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
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


def rithmetic_mutable_mixin(conv_in, cache=None, types=None, opspace=operator):
    """Mixin class to mimic arithmetic numeric types.

    Defines the arithmetic operators `+`, `-`, `*`, `/`, `**`, `==`, `!=` and
    the functions `pow`, `abs`. Operators `//`, `%` can be found in
    `round_mutable_mixin`, `<`, `<=`, `>`, `>=` in `ordered_mixin` and `<<`,
    `>>`, `&`, `^`, `|`, `~` in `bit_mutable_mixin`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    opspace : default - operator
        namespace with function attributes: 'eq', `ne`, `add`, `sub`, `mul`,
        `truediv`, `pow`, `neg`, `pos`, `abs`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method_in = in_method_wrapper(conv_in, cache)
    method = one_method_wrapper(conv_in, cache, types)
    ops = opr_method_wrappers(conv_in, cache, types)
    iop = iop_method_wrapper(conv_in, cache)

    # @total_ordering  # not nan friendly
    class ArithmeticMixin(Complex):
        """Mixin class to mimic arithmetic number types.
        """
        __eq__ = method_in(opspace.eq)
        __ne__ = method_in(opspace.ne)
        __add__, __radd__ = ops(opspace.add)
        __sub__, __rsub__ = ops(opspace.sub)
        __mul__, __rmul__ = ops(opspace.mul)
        __truediv__, __rtruediv__ = ops(opspace.truediv)
        __pow__, __rpow__ = ops(opspace.pow)
        __iadd__ = iop(opspace.iadd)
        __isub__ = iop(opspace.isub)
        __imul__ = iop(opspace.imul)
        __itruediv__ = iop(opspace.itruediv)
        __ipow__ = iop(opspace.ipow)
        __neg__ = method(opspace.neg)
        __pos__ = method(opspace.pos)
        __abs__ = method(opspace.abs)

    return ArithmeticMixin


def round_mutable_mixin(conv_in, cache=None, types=None,
                        opspace=operator, namespace=builtins, mathspace=math):
    """Mixin class for rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, 'round',
    `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    opspace : default - operator
        namespace with function attributes: 'floordiv', `mod`.
    namespace : default - builtins
        namespace with function attributes: 'divmod', `round`.
    mathspace : default - math
        namespace with function attributes: 'trunc', `floor`, `ceil`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method = one_method_wrapper(conv_in, cache, types)
    ops = opr_method_wrappers(conv_in, cache, types)
    iop = iop_method_wrapper(conv_in, cache)

    class RoundableMixin(Real):
        """Mixin class for rounding/modular routines.
        """
        __ifloordiv__ = iop(opspace.ifloordiv)
        __imod__ = iop(opspace.imod)
        __floordiv__, __rfloordiv__ = ops(opspace.floordiv)
        __mod__, __rmod__ = ops(opspace.mod)
        __divmod__, __rdivmod__ = ops(namespace.divmod)
        __round__ = method(namespace.round)
        __trunc__ = method(mathspace.trunc)
        __floor__ = method(mathspace.floor)
        __ceil__ = method(mathspace.ceil)

    return RoundableMixin


def bit_mutable_mixin(conv_in, cache=None, types=None, opspace=operator):
    """Mixin class to mimic bit-string types.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    opspace : default - operator
        namespace with function attributes: 'lshift', `rshift`, `and_`, `xor`,
        `or_`, `invert`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """
    method = one_method_wrapper(conv_in, cache, types)
    ops = opr_method_wrappers(conv_in, cache, types)
    iop = iop_method_wrapper(conv_in, cache)

    class BitTwiddleMixin(Integral):
        """Mixin class to mimic bit-string types.
        """
        __lshift__, __rlshift__ = ops(opspace.lshift)
        __rshift__, __rrshift__ = ops(opspace.rshift)
        __and__, __rand__ = ops(opspace.and_)
        __xor__, __rxor__ = ops(opspace.xor)
        __or__, __ror__ = ops(opspace.or_)
        __ilshift__ = iop(opspace.ilshift)
        __irshift__ = iop(opspace.irshift)
        __iand__ = iop(opspace.iand)
        __ixor__ = iop(opspace.ixor)
        __ior__ = iop(opspace.ior)
        __invert__ = method(opspace.invert)

    return BitTwiddleMixin


def mutable_mixin(conv_in, cache=None, types=None):
    """Mixin class to mimic mutable numeric types.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set or None
        Set storing methods that will need to have __objclass__ set later. When
        `None` the default is used: a set private to this module.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `ops_method_wrappers`,
    `out_method_wrappers` or the last two outputs of `method_wrappers`. The
    `__objclass__` attribute is needed to convert the outputs back.
    """

    class NumberLikeMixin(convertible_mixin(conv_in, cache),
                          rithmetic_mutable_mixin(conv_in, cache, types),
                          ordered_mixin(conv_in, cache),
                          round_mutable_mixin(conv_in, cache, types),
                          bit_mutable_mixin(conv_in, cache, types),
                          ):
        """Mixin class to mimic number types.
        """

    return NumberLikeMixin
