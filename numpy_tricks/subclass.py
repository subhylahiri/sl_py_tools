# -*- coding: utf-8 -*-
"""Helpers for writing __array_ufunc__ and __array_function__ methods.

Routine Listings
----------------
conv_loop_in_attr
    Process inputs in an `__array_ufunc__` method using an attribute.
conv_loop_in_view
    Process inputs in an `__array_ufunc__` method using a view method.
conv_loop_out_attr
    Process outputs in an `__array_ufunc__` method using an attribute.
conv_loop_out_init
    Process outputs in an `__array_ufunc__` method using a constructor.
conv_loop_out_view
    Process outputs in an `__array_ufunc__` method using a view method.
array_ufunc_help_attr
    Implement an `__array_ufunc__` method using an attribute.
array_ufunc_help_view
    Implement an `__array_ufunc__` method using a view method.
array_function_help
    Implement an `__array_function__` method using decorated functions.
make_implements_decorator
    Create a decorator for implementations of array functions.

Example
-------
```
import sl_py_tools.numpy_tricks.subclass as sc

HANDLED_FNS = {}
implements = make_implements_decorator(HANDLED_FNS)

class MyClass(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return sc.array_ufunc_help_attr('arr', MyClass, self,
                                        ufunc, method, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwds):
        return sc.array_function_help(self, HANDLED_FNS, func, types, args, kwds)
```
"""
__all__ = [
    'array_ufunc_help_attr', 'array_ufunc_help_view', 'array_ufunc_help',
    'array_function_help', 'make_implements_decorator',
    'conv_loop_in_attr', 'conv_loop_in_view',
    'conv_loop_out_attr', 'conv_loop_out_init', 'conv_loop_out_view',
    'conv_loop_input', 'conv_loop_in_out', 'conv_loop_out',
    'prepare_via_attr', 'prepare_via_view',
    'restore_via_attr', 'restore_via_init', 'restore_via_view'
]

import itertools
import typing
from typing import Dict, Tuple, Callable, Type, TypeVar, Any, Iterable
import numpy as np

# ======================================================================
# Ufunc Inputs
# ======================================================================


def conv_loop_input(converter: Converter[Custom],
                    obj_typ: Type[Custom],
                    args: ArraysOr[Custom]) -> (OutTuple, BoolList):
    """Process inputs in an __array_ufunc__ method of a custom class.

    Parameters
    ----------
    converter : Callable[Custom -> ndarray]
        Function that converts custom class to `numpy.ndarray`.
    obj_typ : Type[Custom]
        The type of the custom class and the objects that needs converting.
    args: Tuple[Custom or ndarray]
        Tuple of inputs to `ufunc` (or ``out`` argument)

    Returns
    -------
    out : Tuple[ndarray]
        New tuple of inputs to `ufunc` (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    out = []
    conv = []
    for obj in args:
        if isinstance(obj, obj_typ):
            out.append(converter(obj))
            conv.append(True)
        else:
            out.append(obj)
            conv.append(False)
    return out, conv


def conv_loop_in_out(converter: Converter[Custom],
                     obj_typ: Type[Custom],
                     kwargs: ArgDict,
                     num_out: int) -> (OutTuple, BoolList):
    """Process the out keyword in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[Custom -> ndarray]
        Function that converts custom class to `numpy.ndarray`.
    obj_typ : Type[Custom]
        The type of object that needs converting.
    kwargs : Dict[str, Any]
        Dict of keyword inputs to `ufunc`.
    num_out : int
        Number of `ufunc` outputs.

    Returns
    -------
    out : Tuple[ndarray or None]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv_out : List[bool]
        List of bools telling us if each input was converted.
    """
    outputs = kwargs.pop('out', None)
    if outputs:
        out_args, conv_out = conv_loop_input(converter, obj_typ, outputs)
        kwargs['out'] = tuple(out_args)
    else:
        outputs = (None,) * num_out
    return outputs, conv_out


def _conv_loop_in(converter: Converter[Custom], obj_typ: Type[Custom],
                  *args) -> (OutTuple, BoolList):
    """Call one of conv_loop_input or conv_loop_in_out"""
    if len(args) == 1:
        return conv_loop_input(converter, obj_typ, *args)
    return conv_loop_in_out(converter, obj_typ, *args)


def prepare_via_view() -> Converter[Custom]:
    """Create function to convert object to an array using view method.
    """
    def converter(thing: Custom) -> np.ndarray:
        """convert to array using view method
        """
        return thing.view(np.ndarray)
    return converter


def prepare_via_attr(attr: str) -> Converter[Custom]:
    """Create function to convert object to an array using an attribute.

    Parameters
    ----------
    attr: str, None
        The name of the ``obj_typ`` attribute to use in place of class.
    """
    def converter(thing: Custom) -> np.ndarray:
        """convert to array using an attribute
        """
        return getattr(thing, attr)
    return converter


def conv_loop_in_view(obj_typ: Type, *args) -> (OutTuple, BoolList):
    """Process inputs in an __array_ufunc__ method using view method.

    Parameters
    ----------
    obj_typ : Type[Custom]
        The type of object that needs converting via ``view`` method.
    args : Tuple[ndarray or Custom] or Dict[str, Any]
        Tuple of inputs to ufunc (or ``out`` argument), or
        Dict of keyword inputs to ufunc.
    num_out : int, optional
        Number of `ufunc` outputs. If present, assumes that ``args`` is the
        keyword dictionary to process the ``out`` argument.

    Returns
    -------
    out : Tuple[ndarray or None]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    return _conv_loop_in(prepare_via_view(), obj_typ, *args)


def conv_loop_in_attr(attr: str, obj_typ: Type, *args) -> (OutTuple, BoolList):
    """Process inputs in an __array_ufunc__ method using an attribute.

    Parameters
    ----------
    attr : str
        The name of the ``obj_typ`` attribute to use in place of class.
    obj_typ : Type[Custom]
        The type of object that needs converting with its ``attr`` attribute.
    args : Tuple[ndarray or Custom] or Dict[str, Any]
        Tuple of inputs to ufunc (or ``out`` argument), or
        Dict of key word inputs to ufunc
    num_out : int, optional
        Number of `ufunc` outputs. If present, assumes that ``args`` is the
        keyword dictionary to process the ``out`` argument.

    Returns
    -------
    out : Tuple[ndarray or None]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    return _conv_loop_in(prepare_via_attr(attr), obj_typ, *args)


# ======================================================================
# Ufunc Outputs
# ======================================================================


def conv_loop_out(converter: UnConverter[Custom],
                  results: ArgTuple,
                  outputs: OutTuple,
                  conv: BoolSequence = ()) -> ArraysOr[Custom]:
    """Process outputs in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[ndarray -> Custom]
        Function to perform reverse conversions.
    results : Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs : Tuple[ndarray or None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv : Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results : Tuple[ndarray or Custom]
        New tuple of results from ufunc with conversions.
    """
    if results is NotImplemented:
        return NotImplemented
    if not isinstance(results, tuple):
        results = (results,)
    if not conv:
        conv = itertools.repeat(True)
    results_out = []
    for result, output, cout in zip(results, outputs, conv):
        if output is None:
            if cout:
                results_out.append(converter(result))
            else:
                results_out.append(result)
        else:
            results_out.append(output)
    if len(results_out) == 1:
        return results_out[0]
    return tuple(results_out)


def restore_via_copy_attr(obj: Custom, attr: str) -> UnConverter[Custom]:
    """Create function to convert arrays by setting obj.attr in obj.copy.

    Parameters
    ----------
    obj : Custom
        The template object for conversions. Must have a `copy` method.
    attr: str
        The name of the ``type(obj)`` attribute returned in place of class.
        It will try to use ``obj.copy(attr=result)``. If that fails, it will
        use ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    def converter(thing: np.ndarray) -> Custom:
        """convert arrays by setting obj.attr
        """
        try:
            return obj.copy(**{attr: thing})
        except TypeError:
            pass
        thing_out = obj.copy()
        setattr(thing_out, attr, thing)
        return thing_out
    return converter


def restore_via_init_attr(obj: Custom, attr: str) -> UnConverter[Custom]:
    """Create function to convert arrays by setting obj.attr in obj.__init__.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    attr: str
        The name of the ``type(obj)`` attribute returned in place of class.
        It will try to use ``obj.__class__(attr=result)``. If that fails,
        it will use ``obj.__class__(result)``.
    """
    def converter(thing: np.ndarray) -> Custom:
        """convert arrays by setting obj.attr
        """
        try:
            return obj.__class__(**{attr: thing})
        except TypeError:
            return obj.__class__(thing)
    return converter


def restore_via_attr(obj: Custom, attr: str) -> UnConverter[Custom]:
    """Create function to convert arrays by setting obj.attr.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    attr: str
        The name of the ``type(obj)`` attribute returned in place of class.
        It will try to use ``obj.copy(attr=result)``. If that fails, it will
        try to use ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
        If that fails, it will try to use ``obj.__class__(attr=result)``.
        If that fails, it will use ``obj.__class__(result)``.
    """
    conv_copy = restore_via_copy_attr(obj, attr)
    conv_init = restore_via_init_attr(obj, attr)
    def converter(thing: np.ndarray) -> Custom:
        """convert arrays by setting obj.attr
        """
        try:
            return conv_copy(thing)
        except AttributeError:
            return conv_init(thing)
    return converter


def restore_via_init(obj: Custom) -> UnConverter[Custom]:
    """Create function to convert arrays  using obj.__init__.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    """
    def converter(thing: np.ndarray) -> Custom:
        """convert arrays using obj.__init__
        """
        return obj.__class__(thing)
    return converter


def restore_via_view(obj: Custom) -> UnConverter[Custom]:
    """Create function to convert arrays using array.view.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    """
    def converter(thing: np.ndarray) -> Custom:
        """convert arrays using array.view
        """
        return thing.view(type(obj))
    return converter


def conv_loop_out_attr(obj: Custom,
                       attr: str,
                       results: ArgTuple,
                       outputs: OutTuple,
                       conv: BoolSequence = ()) -> ArraysOr[Custom]:
    """Process outputs in an __array_ufunc__ method using an attribute.

    Makes a copy of ``obj`` with ``obj.attr = result``.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    attr: str, None
        The name of the ``type(obj)`` attribute used in place of class.
    results: Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs: Tuple[ndarray or None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[ndarray or Custom]
        New tuple of results from ufunc with conversions.

    Notes
    -----
    It will try to use ``obj.copy(attr=result)``. If that fails, it will use
    ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    return conv_loop_out(restore_via_attr(obj, attr), results, outputs, conv)


def conv_loop_out_init(obj: Custom,
                       results: ArgTuple,
                       outputs: OutTuple,
                       conv: BoolSequence = ()) -> ArraysOr[Custom]:
    """Process outputs in an __array_ufunc__ method using a constructor.

    Creates an instance of ``type(obj)`` with ``result`` as its argument.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    results: Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs: Tuple[ndarray or None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[ndarray or Custom]
        New tuple of results from ufunc with conversions.
    """
    return conv_loop_out(restore_via_init(obj), results, outputs, conv)


def conv_loop_out_view(obj: Custom,
                       results: ArgTuple,
                       outputs: OutTuple,
                       conv: BoolSequence = ()) -> ArraysOr[Custom]:
    """Process outputs in an __array_ufunc__ method using a view method.

    Calls ``result.view`` with ``type(obj)`` with as its argument.

    Parameters
    ----------
    obj : Custom
        The template object for conversions.
    results: Tuple[ndarray]
        Tuple of outputs from ``ufunc``
    outputs: Tuple[ndarray or None]
        ``out`` argument of ``ufunc``, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[ndarray or Custom]
        New tuple of results from ufunc with conversions.
    """
    return conv_loop_out(restore_via_view(obj), results, outputs, conv)


# =============================================================================
# Complete array ufunc helpers
# =============================================================================


def array_ufunc_help(converters: Tuple[Converter[Custom], UnConverter[Custom]],
                     delegate: ArrayUfunc, obj_type: Type[Custom],
                     ufunc: np.ufunc, method: str,
                     *inputs, **kwargs) -> ArraysOr[Custom]:
    """Helper for __array_ufunc__ using an ndarray attribute of the class

    Parameters
    ----------
    converters : Tuple[conv_in, conv_out]
        conv_in : Callable[Custom -> ndarray]
            Function that converts custom class to `numpy.ndarray`.
        conv_out : Callable[ndarray -> Custom]
            Function to perform reverse conversions.
    delegate : numpy.ndarray
        The implementation of `__array_ufunc__` we will delegate to.
    obj_typ : Type[Custom]
        The type of object that needs converting via ``view`` method.
    ufunc : numpy.ufunc
        The `ufunc` that was oroginally called
    method : str
        The name of the ufunc method that was called
    inputs : Tuple[Any, ...]
        Positional arguments from the original function call.
    kwargs : _Dict[str, Any]
        Keyword arguments from the original function call.

    Returns
    -------
    results : Any
        The output of the custom implementation of `ufunc`.
    """
    conv_in, conv_out = converters
    args, _ = conv_loop_input(conv_in, obj_type, inputs)
    outputs, conv = conv_loop_in_out(conv_in, obj_type, kwargs, ufunc.nout)
    results = delegate(ufunc, method, *args, **kwargs)
    return conv_loop_out(conv_out, results, outputs, conv)


def array_ufunc_help_attr(attr: str, obj_type: Type[Custom],
                          obj: Custom, ufunc: np.ufunc, method: str,
                          *inputs, **kwargs) -> ArraysOr[Custom]:
    """Helper for __array_ufunc__ using an ndarray attribute of the class

    Parameters
    ----------
    attr : str
        The name of the ``type(obj)`` attribute used in place of class.
    obj_type : Type[Custom]
        The type of object that needs converting via ``view`` method.
    obj : Custom
        The object whose `__array_function__` method was called.
    ufunc : numpy.ufunc
        The `ufunc` that was oroginally called
    method : str
        The name of the ufunc method that was called
    inputs : Tuple[ndarray or Custom, ...]
        Positional arguments from the original function call.
    kwargs : _Dict[str, Any]
        Keyword arguments from the original function call.

    Returns
    -------
    results : Tuple[ndarray or Custom, ...]
        The output of the custom implementation of `ufunc`.
    """
    converters = prepare_via_attr(attr), restore_via_attr(obj, attr)
    delegate = getattr(obj, attr).__array_ufunc__
    return array_ufunc_help(converters, delegate, obj_type,
                            ufunc, method, *inputs, **kwargs)


def array_ufunc_help_view(obj_type: Type[Custom], obj: Custom, ufunc: np.ufunc,
                          method: str, *inputs, **kwargs) -> ArraysOr[Custom]:
    """Helper for __array_ufunc__ using the view method of the class

    Parameters
    ----------
    obj_type : Type[Custom]
        The type of object that needs converting via ``view`` method.
    obj : Custom
        The object whose `__array_function__` method was called.
    ufunc : numpy.ufunc
        The `ufunc` that was oroginally called
    method : str
        The name of the ufunc method that was called
    inputs : Tuple[ndarray or Custom, ...]
        Positional arguments from the original function call.
    kwargs : _Dict[str, Any]
        Keyword arguments from the original function call.

    Returns
    -------
    results : Tuple[ndarray or Custom, ...]
        The output of the custom implementation of `ufunc`.
    """
    converters = prepare_via_view(), restore_via_view(obj)
    delegate = super(obj_type, obj).__array_ufunc__
    return array_ufunc_help(converters, delegate, obj_type,
                            ufunc, method, *inputs, **kwargs)



# =============================================================================
# Array function helpers
# =============================================================================


def make_implements_decorator(handled: HandleMap[Custom, Result]
                              ) -> DecoratorParamd[NumpyFun, Custom, Result]:
    """Create decorator to register __array_function__ implementations.

    Parameters
    ----------
    handled : Dict[Callable, Callable], NumpyFun -> MyFun
        Module level dictionary mapping `numpy` function to its implementation.

    Returns
    -------
    decorator : Callable[Callable -> Callable]
        The decorator for functions that implement a `numpy` function.
        Takes the `numpy` function as a parameter before decorating.

    Example
    -------
    see module docstring.
    """
    def implements(numpy_function: NumpyFun) -> Decorator:
        """Register an implementation of an array function for custom objects.
        """
        def decorator(func: MyFun[Custom, Result]) -> MyFun[Custom, Result]:
            handled[numpy_function] = func
            return func
        return decorator
    return implements


def array_function_help(obj: Custom,
                        handled: HandleMap[Custom, Result],
                        func: NumpyFun,
                        types: Iterable[Type],
                        args: ArraysOr[Custom],
                        kwargs: ArgDict) -> Result:
    """Helper function for writing __array_function__ methods

    Parameters
    ----------
    obj : Custom
        The object whose `__array_function__` method was called.
    handled : Dict[NumpyFun -> MyFun]
        Dictionary mapping `numpy` functions to their custom implementations.
    func : NumpyFun = Callable[ndarray -> Any]
        The `numpy` function that dispatched to `obj.__array_function__`.
    types : Collection[Type[Any]]
        Collection of types of arguments from the original function call.
    args : Tuple[ndarray or Custom, ...]
        Positional arguments from the original function call.
    kwargs : _Dict[str, Any]
        Keyword arguments from the original function call.

    Returns
    -------
    results : Result
        The output of the custom implementation, `handled[func]`.

    Example
    -------
    see module docstring.
    """
    if func not in handled:
        return NotImplemented
    if not all(issubclass(t, obj.__class__) for t in types):
        return NotImplemented
    return handled[func](*args, **kwargs)


# =============================================================================
# Type hint classes
# =============================================================================

Param = TypeVar('Param')
Result = TypeVar('Result')
Custom = TypeVar('Custom')

ArraysOr = Tuple[typing.Union[np.ndarray, Param], ...]
ArgTuple = Tuple[np.ndarray, ...]
OutTuple = ArraysOr[None]
ArgDict = Dict[str, Any]

Converter = Callable[[Custom], np.ndarray]
UnConverter = Callable[[np.ndarray], Custom]
BoolList = typing.List[bool]
BoolSequence = Iterable[bool]
ArrayUfunc = Callable[[np.ufunc, str, ArgTuple, ArgDict], ArgTuple]

NumpyFun = Callable[[np.ndarray], Any]
MyFun = Callable[[Custom], Result]
HandleMap = Dict[NumpyFun, MyFun[Custom, Result]]
Decorator = Callable[[MyFun[Custom, Result]], MyFun[Custom, Result]]
DecoratorParamd = Callable[[Param], Decorator[Custom, Result]]
