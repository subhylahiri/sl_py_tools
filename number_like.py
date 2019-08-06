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
opr_method_wrappers
    Decorator to turn one function into two magic methods - forward, reverse
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
"""
import builtins
import math
import operator
from numbers import Complex, Real, Integral, Number
from collections.abc import Iterable
from typing import Callable, Tuple
from functools import wraps
from .arg_tricks import default, default_non_eval

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


def dummy_method(name: str) -> Callable:
    """Return a dummy function that always returns NotImplemented.

    This can be used to effectively delete an unwanted operator from a mixin.
    """
    def dummy(*args):
        """Dummy function"""
        return NotImplemented
    dummy.__name__ = name
    return dummy


# -----------------------------------------------------------------------------
# %%* Method wrapper helpers
# -----------------------------------------------------------------------------


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
    except TypeError:
        return NotImplemented
    else:
        return _multi_conv(method.__objclass__, result, types)


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


def _implement_iop(func, args, conv_in, attr=None):
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
        The name of the attribute that is updated for inplace operations.

    Returns
    -------
    result
        The results of `func` after conversion.
    """
    if attr is None:
        return _implement_mutable_iop(func, args, conv_in)
    try:
        result = func(*conv_in(args))
    except TypeError:
        return NotImplemented
    else:
        # if attr is immutable (like numbers), no need to use inplace function
        setattr(args[0], attr, result)
        return args[0]


def _magic_name(func: Callable, prefix: str = None):
    """convert function name into magic method format"""
    prefix = default(prefix, '')
    return '__' + prefix + func.__name__.strip('_') + '__'


# =============================================================================
# %%* Setting method __objclass__
# =============================================================================


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


def set_objclasses(objclass: type, cache: set):
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


# =============================================================================
# %%* Method wrappers
# =============================================================================


def in_method_wrapper(conv_in: Callable, cache: set) -> Callable:
    """make wrappers for some class

    Make method, with inputs that are class/number & outputs that are number.
    When conversion back to class is required, it uses method's `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.

    Returns
    -------
    method_input : Callable
        A decorator for a method, so that the method's inputs are preconverted.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def method_input(func: Callable) -> Callable:
        """Wrap method that returns another type
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            try:
                return func(*conv_in(args))
            except TypeError:
                return NotImplemented
        wrapper.__name__ = _magic_name(func)
        _add_set_objclass(wrapper, cache)
        return wrapper
    return method_input


def one_method_wrapper(conv_in: Callable, cache: set, types=None) -> Callable:
    """make wrappers for some class

    make method, with inputs that are class/number & outputs that are class.
    Conversion back to class uses method's `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
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
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def method(func: Callable) -> Callable:
        """Wrap method, so that the method's inputs are preconverted
        and its outputs are post converted.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper, types)
        wrapper.__name__ = _magic_name(func)
        _add_set_objclass(wrapper, cache)
        return wrapper
    return method


def opr_method_wrappers(conv_in: Callable, cache: set, types=None) -> Callable:
    """make wrappers for operator doublet: forward, reverse,

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
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
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    def operator_set(func: Callable) -> Tuple[Callable, ...]:
        """Wrap operator set.

        A decorator for a method, returning three methods, so that the method's
        inputs are preconverted and its outputs are post converted.
        The three methods are for forward, reverse, and inplace operators.
        """
        # @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def wrapper(*args):
            return _implement_op(func, args, conv_in, wrapper, types)

        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def rwrapper(*args):
            return _implement_op(func, reversed(args), conv_in, rwrapper,
                                 types)

        wrapper.__name__ = _magic_name(func)
        rwrapper.__name__ = _magic_name(func, 'r')
        _add_set_objclass((wrapper, rwrapper), cache)
        return wrapper, rwrapper

    return operator_set


def iop_method_wrapper(conv_in: Callable, cache: set, attr=None) -> Callable:
    """make wrapper for inplace operator, immutable data

    make methods, with inputs that are class/number & outputs that are class.
    Conversion back to class uses methods' `__objclass__`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
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
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    prefix = default_non_eval(attr, lambda x: 'i', '')

    def operator_set(func: Callable) -> Callable:
        """Wrap operator set.

        A decorator for a method, returning three methods, so that the method's
        inputs are preconverted and its outputs are post converted.
        The three methods are for forward, reverse, and inplace operators.
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS_N)
        def iwrapper(*args):
            return _implement_iop(func, args, conv_in, attr)

        iwrapper.__name__ = _magic_name(func, prefix)
        _add_set_objclass(iwrapper, cache)
        return iwrapper
    return operator_set


# =============================================================================
# %%* Function wrappers
# =============================================================================


def function_wrappers(conv_in: Callable, class_out: type, types=None):
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
    def fun_input(func: Callable) -> Callable:
        """Wrap function that returns another type
        """
        @wraps(func, assigned=WRAPPER_ASSIGNMENTS)
        def wrapper(*args):
            try:
                return func(*conv_in(args))
            except TypeError:
                return NotImplemented
        return wrapper

    def ext_fun(func: Callable) -> Callable:
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


def convert_mixin(conv_in: Callable, cache: set, nmspace=None) -> type:
    """Mixin class for conversion to number types.

    Defines the functions `complex`, `float`, `int`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    nmspace : default - builtins
        namespace with function attributes: {'complex', `float`, `int`}.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    defspace = default(nmspace, builtins)
    method_input = in_method_wrapper(conv_in, cache)

    class ConvertibleMixin(Complex):
        """Mixin class for conversion to number types.
        """
        __complex__ = method_input(defspace.complex)
        __float__ = method_input(defspace.float)
        __int__ = method_input(defspace.int)
        __index__ = method_input(defspace.int)

    return ConvertibleMixin


def ordered_mixin(conv_in: Callable, cache: set, nmspace=None) -> type:
    """Mixin class for arithmetic comparisons.

    Defines all of the comparison operators, `==`, `!=`, `<`, `<=`, `>`, `>=`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    nmspace : default - operator
        namespace w/ function attributes: {'eq', `ne`, `lt`, 'le', `gt`, `ge`}.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace = default(nmspace, operator)
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


def mathops_mixin(conv_in, cache: set, types=None, nmspace=None) -> type:
    """Mixin class to mimic arithmetic number types.

    Defines the arithmetic operators `+`, `-`, `*`, `/`, `**`, `==`, `!=` and
    the functions `pow`, `abs`. Operators `//`, `%` are in `rounder_mixin`,
    operators `<`, `<=`, `>`, `>=` are in `ordered_mixin` and `<<`, `>>`, `&`,
    `^`, `|`, `~` are in `bit_twiddle_mixin`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    nmspace : default - operator
        namespace with function attributes: {'eq', `ne`, `add`, `sub`, `mul`,
        `truediv`, `pow`, `neg`, `pos`, `abs`}.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace = default(nmspace, operator)
    method_in = in_method_wrapper(conv_in, cache)
    method = one_method_wrapper(conv_in, cache, types)
    ops = opr_method_wrappers(conv_in, cache, types)

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
        __neg__ = method(opspace.neg)
        __pos__ = method(opspace.pos)
        __abs__ = method(opspace.abs)

    return ArithmeticMixin


def rounder_mixin(conv_in, cache: set, types=None, nmspace=None) -> type:
    """Mixin class for rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, 'round',
    `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    nmspace:
        (opspace, namespace, mathspace)

        opspace : default - operator
            namespace with function attributes: {'floordiv', `mod`}.
        defspace : default - builtins
            namespace with function attributes: {'divmod', `round`}.
        mathspace : default - math
            namespace with function attributes: {'trunc', `floor`, `ceil}`.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace, defspace, mathspace = default(nmspace, (operator, builtins, math))
    method = one_method_wrapper(conv_in, cache, types)
    ops = opr_method_wrappers(conv_in, cache, types)

    class RoundableMixin(Real):
        """Mixin class for rounding/modular routines.
        """
        __floordiv__, __rfloordiv__ = ops(opspace.floordiv)
        __mod__, __rmod__ = ops(opspace.mod)
        __divmod__, __rdivmod__ = ops(defspace.divmod)
        __round__ = method(defspace.round)
        __trunc__ = method(mathspace.trunc)
        __floor__ = method(mathspace.floor)
        __ceil__ = method(mathspace.ceil)

    return RoundableMixin


def bitwise_mixin(conv_in, cache: set, types=None, nmspace=None) -> type:
    """Mixin class to mimic bit-string types.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.
    nmspace : default - operator
        namespace with function attributes: {'lshift', `rshift`, `and_`, `xor`,
        `or_`, `invert`}.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace = default(nmspace, operator)
    method = one_method_wrapper(conv_in, cache, types)
    ops = opr_method_wrappers(conv_in, cache, types)

    class BitTwiddleMixin(Integral):
        """Mixin class to mimic bit-string types.
        """
        __lshift__, __rlshift__ = ops(opspace.lshift)
        __rshift__, __rrshift__ = ops(opspace.rshift)
        __and__, __rand__ = ops(opspace.and_)
        __xor__, __rxor__ = ops(opspace.xor)
        __or__, __ror__ = ops(opspace.or_)
        __invert__ = method(opspace.invert)

    return BitTwiddleMixin


# -----------------------------------------------------------------------------
# %%* Inplace Mixins
# -----------------------------------------------------------------------------


def imaths_mixin(conv_in, cache: set, attr=None, nmspace=None) -> type:
    """Mixin class to mimic arithmetic number types.

    Defines the arithmetic operators `+`, `-`, `*`, `/`, `**`, `==`, `!=` and
    the functions `pow`, `abs`. Operators `//`, `%` are in `rounder_mixin`,
    operators `<`, `<=`, `>`, `>=` are in `ordered_mixin` and `<<`, `>>`, `&`,
    `^`, `|`, `~` are in `bit_twiddle_mixin`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    nmspace : default - operator
        namespace with function attributes:
        for immutable data - {`add`, `sub`, `mul`, `truediv`, `pow`};
        for mutable data - {`iadd`, `isub`, `imul`, `itruediv`, `ipow`}.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace = default(nmspace, operator)
    iop = iop_method_wrapper(conv_in, cache, attr)
    prefix = default_non_eval(attr, lambda x: 'i', '')

    # @total_ordering  # not nan friendly
    class IArithmeticMixin(Complex):
        """Mixin class to mimic arithmetic number types.
        """
        __iadd__ = iop(getattr(opspace, prefix + 'add'))
        __isub__ = iop(getattr(opspace, prefix + 'sub'))
        __imul__ = iop(getattr(opspace, prefix + 'mul'))
        __itruediv__ = iop(getattr(opspace, prefix + 'truediv'))
        __ipow__ = iop(getattr(opspace, prefix + 'pow'))

    return IArithmeticMixin


def iround_mixin(conv_in, cache: set, attr=None, nmspace=None) -> type:
    """Mixin class for rounding/modular routines.

    Defines the operators `%`, `//`, and the functions  `divmod`, 'round',
    `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    nmspace : default - operator
        namespace with function attributes:
        for immutable data - {'floordiv', `mod`};
        for mutable data - {'ifloordiv', `imod`}.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace = default(nmspace, operator)
    iop = iop_method_wrapper(conv_in, cache, attr)
    prefix = default_non_eval(attr, lambda x: 'i', '')

    class IRoundableMixin(Real):
        """Mixin class for rounding/modular routines.
        """
        __ifloordiv__ = iop(getattr(opspace, prefix + 'floordiv'))
        __imod__ = iop(getattr(opspace, prefix + 'mod'))

    return IRoundableMixin


def ibitws_mixin(conv_in, cache: set, attr=None, nmspace=None) -> type:
    """Mixin class to mimic bit-string types.

    Defines all of the bit-wise operators: `<<`, `>>`, `&`, `^`, `|`, `~`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    nmspace : default - operator
        namespace with function attributes:
        for immutable data - {'lshift', `rshift`, `and_`, `xor`, `or_`};
        for mutable data - {'ilshift', `irshift`, `iand`, `ixor`, `ior`}.

    Notes
    -----
    You should call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """
    opspace = default(nmspace, operator)
    iop = iop_method_wrapper(conv_in, cache, attr)
    prefix = default_non_eval(attr, lambda x: '', 'i')
    suffix = default_non_eval(attr, lambda x: '_', '')

    class IBitTwiddleMixin(Integral):
        """Mixin class to mimic bit-string types.
        """
        __ilshift__ = iop(getattr(opspace, prefix + 'lshift'))
        __irshift__ = iop(getattr(opspace, prefix + 'rshift'))
        __iand__ = iop(getattr(opspace, prefix + 'and' + suffix))
        __ixor__ = iop(getattr(opspace, prefix + 'xor'))
        __ior__ = iop(getattr(opspace, prefix + 'or' + suffix))

    return IBitTwiddleMixin


# -----------------------------------------------------------------------------
# %%* Combined Mixins
# -----------------------------------------------------------------------------


def number_mixin(conv_in: Callable, cache: set, types=None) -> type:
    """Mixin class to mimic number types.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`, *except for*
    the bit-wise and the inplace operators.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """

    class NumberLikeMixin(convert_mixin(conv_in, cache),
                          ordered_mixin(conv_in, cache),
                          mathops_mixin(conv_in, cache, types),
                          rounder_mixin(conv_in, cache, types),
                          ):
        """Mixin class to mimic number types.
        """

    return NumberLikeMixin


def integr_mixin(conv_in: Callable, cache: set, types=None) -> type:
    """Mixin class to mimic integer types.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`, *except for*
    the inplace operators.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """

    class IntegerLikeMixin(number_mixin(conv_in, cache, types),
                           bitwise_mixin(conv_in, cache, types),
                           ):
        """Mixin class to mimic integer types.
        """

    return IntegerLikeMixin


def inumbr_mixin(conv_in: Callable, cache: set, types=None, attr=None) -> type:
    """Mixin class to mimic number types, with inplace operators.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`, *except for*
    the bit-wise operators.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """

    class NumberLikeMixin(number_mixin(conv_in, cache, types),
                          imaths_mixin(conv_in, cache, attr),
                          iround_mixin(conv_in, cache, attr),
                          ):
        """Mixin class to mimic number types.
        """

    return NumberLikeMixin


def iintgr_mixin(conv_in: Callable, cache: set, types=None, attr=None) -> type:
    """Mixin class to mimic integer types, with inplace operators.

    Defines all of the operators and the functions `complex`, `float`, `int`,
    `pow`, `abs`, `divmod`, 'round', `math.floor,ceil,trunc`.

    Parameters
    ----------
    conv_in : Callable
        Function used to convert a tuple of inputs.
    attr : str
        The name of the attribute that is updated for inplace operations on
        immutable data, or `None` for mutable data.
    cache : set
        Set that stores methods that will need to have __objclass__ set later.
    types : Type or Tuple[Type] or None
        The types of output that should be converted.

    Notes
    -----
    You must call `set_objclasses(class, cache)` after defining the `class`,
    especially if you used any of `one_method_wrapper`, `opr_method_wrappers`.
    The `__objclass__`attribute is needed to convert the outputs back.
    """

    class IntegerLikeMixin(inumbr_mixin(conv_in, cache, types, attr),
                           bitwise_mixin(conv_in, cache, types),
                           ibitws_mixin(conv_in, cache, attr),
                           ):
        """Mixin class to mimic integer types.
        """

    return IntegerLikeMixin
