# -*- coding: utf-8 -*-
"""Helpers for writing __array_ufunc__ methods.

Routine Listings
----------------
conv_loop_in_attr
    Process inputs in an __array_ufunc__ method using an attribute.
conv_loop_in_view
    Process inputs in an __array_ufunc__ method using view method.
conv_loop_out_attr
    Process outputs in an __array_ufunc__ method using an attribute.
conv_loop_out_init
    Process outputs in an __array_ufunc__ method using a constructor.
conv_loop_out_view
    Process outputs in an __array_ufunc__ method using a view method.

Example
-------
```
import sl_py_tools.numpy_tricks.subclass as sc

HANDLED_FUNCTIONS = {}
implements = make_implements_decorator(HANDLED_FUNCTIONS)

class MyClass(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args, _ = sc.conv_loop_in_attr('attr', MyClass, inputs)
        outputs, conv = sc.conv_loop_in_attr('arr', MyClass, kwargs, ufunc.nout)
        results = self.attr.__array_ufunc__(ufunc, method, *args, **kwargs)
        return sc.conv_loop_out_attr(self, 'arr', results, outputs, conv)

    def __array_function__(self, func, types, args, kwargs):
        sc.array_function_helper(self, HANDLED_FUNCTIONS, func, types, args, kwargs)
```
"""
import itertools as _itertools
import typing as _ty
import numpy as _np
Param = _ty.TypeVar('Param')
Result = _ty.TypeVar('Result')
ArrayOr = _ty.Union[_np.ndarray, Param]
Converter = _ty.Callable[[ArrayOr[Param]], _np.ndarray]
UnConverter = _ty.Callable[[_np.ndarray], ArrayOr[Param]]
ArgTuple = _ty.Tuple[Param, ...]
OutTuple = _ty.Tuple[_ty.Optional[_np.ndarray], ...]
ArgDict = _ty.Dict[str, _ty.Any]
BoolList = _ty.List[bool]
BoolSequence = _ty.Sequence[bool]
NumpyFun = _ty.Callable[[_np.ndarray], _np.ndarray]
MyFun = _ty.Callable[[Param], Result]
HandleMap = _ty.Dict[NumpyFun, MyFun]
Decorator = _ty.Callable[[MyFun], MyFun]
DecoratorParamd = _ty.Callable[[Param], Decorator]
# ======================================================================
# Ufunc Inputs
# ======================================================================


def conv_loop_input(converter: Converter[Param],
                    obj_typ: _ty.Type[Param],
                    args: ArgTuple[ArrayOr[Param]]) -> (OutTuple, BoolList):
    """Process inputs in an __array_ufunc__ method of a custom class.

    Parameters
    ----------
    converter : Callable[Param -> ndarray]
        Function that converts custom class to `numpy.ndarray`.
    obj_typ : Type[Param]
        The type of the custom class and the objects that needs converting.
    args: Tuple[Param or ndarray]
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


def conv_loop_in_out(converter: Converter[Param],
                     obj_typ: _ty.Type[Param],
                     kwargs: ArgDict,
                     num_out: int) -> (OutTuple, BoolList):
    """Process the out keyword in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[Param -> ndarray]
        Function that converts custom class to `numpy.ndarray`.
    obj_typ : Type[Param]
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


def _conv_loop_in(converter: Converter[Param], obj_typ: _ty.Type[Param],
                  *args) -> (OutTuple, BoolList):
    """Call one of conv_loop_input or conv_loop_in_out"""
    if len(args) == 1:
        return conv_loop_input(converter, obj_typ, *args)
    return conv_loop_in_out(converter, obj_typ, *args)


def prepare_via_view() -> Converter:
    """Create function to convert object to an array using view method.
    """
    def converter(thing):
        """convert to array using view method
        """
        return thing.view(_np.ndarray)
    return converter


def prepare_via_attr(attr: str) -> Converter:
    """Create function to convert object to an array using an attribute.

    Parameters
    ----------
    attr: str, None
        The name of the ``obj_typ`` attribute to use in place of class.
    """
    def converter(thing):
        """convert to array using an attribute
        """
        return getattr(thing, attr)
    return converter


def conv_loop_in_view(obj_typ: _ty.Type, *args) -> (OutTuple, BoolList):
    """Process inputs in an __array_ufunc__ method using view method.

    Parameters
    ----------
    obj_typ : Type[Param]
        The type of object that needs converting via ``view`` method.
    args : Tuple[ndarray or Param] or Dict[str, Any]
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


def conv_loop_in_attr(attr: str, obj_typ: _ty.Type, *args) -> (OutTuple, BoolList):
    """Process inputs in an __array_ufunc__ method using an attribute.

    Parameters
    ----------
    attr : str
        The name of the ``obj_typ`` attribute to use in place of class.
    obj_typ : Type[Param]
        The type of object that needs converting with its ``attr`` attribute.
    args : Tuple[ndarray or Param] or Dict[str, Any]
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


def conv_loop_out(converter: UnConverter[Param],
                  results: ArgTuple[_np.ndarray],
                  outputs: OutTuple,
                  conv: BoolSequence = ()) -> ArgTuple[ArrayOr[Param]]:
    """Process outputs in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[ndarray -> Param]
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
    results : Tuple[ndarray or Param]
        New tuple of results from ufunc with conversions.
    """
    if results is NotImplemented:
        return NotImplemented
    if not isinstance(results, tuple):
        results = (results,)
    if not conv:
        conv = _itertools.repeat(True)
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


def restore_via_attr(obj: Param, attr: str) -> UnConverter[Param]:
    """Create function to convert arrays by setting obj.attr.

    Parameters
    ----------
    obj : Param
        The template object for conversions.
    attr: str
        The name of the ``type(obj)`` attribute returned in place of class.
        It will try to use ``obj.copy(attr=result)``. If that fails, it will
        use ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    def converter(thing: _np.ndarray) -> Param:
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


def restore_via_init(obj: Param) -> UnConverter[Param]:
    """Create function to convert arrays  using obj.__init__.

    Parameters
    ----------
    obj : Param
        The template object for conversions.
    """
    def converter(thing: _np.ndarray) -> Param:
        """convert arrays using obj.__init__
        """
        return type(obj)(thing)
    return converter


def restore_via_view(obj: Param) -> UnConverter[Param]:
    """Create function to convert arrays using array.view.

    Parameters
    ----------
    obj : Param
        The template object for conversions.
    """
    def converter(thing: _np.ndarray) -> Param:
        """convert arrays using array.view
        """
        return thing.view(type(obj))
    return converter


def conv_loop_out_attr(obj: Param,
                       attr: str,
                       results: ArgTuple[_np.ndarray],
                       outputs: OutTuple,
                       conv: BoolSequence = ()) -> ArgTuple[ArrayOr[Param]]:
    """Process outputs in an __array_ufunc__ method using an attribute.

    Makes a copy of ``obj`` with ``obj.attr = result``.

    Parameters
    ----------
    obj : Param
        The template object for conversions.
    attr: str, None
        The name of the ``type(obj)`` attribute returned in place of class.
    results: Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs: Tuple[ndarray or None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[ndarray or Param]
        New tuple of results from ufunc with conversions.

    Notes
    -----
    It will try to use ``obj.copy(attr=result)``. If that fails, it will use
    ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    return conv_loop_out(restore_via_attr(obj, attr), results, outputs, conv)


def conv_loop_out_init(obj: Param,
                       results: ArgTuple[_np.ndarray],
                       outputs: OutTuple,
                       conv: BoolSequence = ()) -> ArgTuple[ArrayOr[Param]]:
    """Process outputs in an __array_ufunc__ method using a constructor.

    Creates an instance of ``type(obj)`` with ``result`` as its argument.

    Parameters
    ----------
    obj : Param
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
    results: Tuple[ndarray or Param]
        New tuple of results from ufunc with conversions.
    """
    return conv_loop_out(restore_via_init(obj), results, outputs, conv)


def conv_loop_out_view(obj: Param,
                       results: ArgTuple[_np.ndarray],
                       outputs: OutTuple,
                       conv: BoolSequence = ()) -> ArgTuple[ArrayOr[Param]]:
    """Process outputs in an __array_ufunc__ method using a view method.

    Calls ``result.view`` with ``type(obj)`` with as its argument.

    Parameters
    ----------
    obj : Param
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
    results: Tuple[ndarray or Param]
        New tuple of results from ufunc with conversions.
    """
    return conv_loop_out(restore_via_view(obj), results, outputs, conv)


# =============================================================================
# Array function helpers
# =============================================================================


def make_implements_decorator(handled: HandleMap) -> DecoratorParamd[NumpyFun]:
    """Create decorator to register an `__array_function__` implementations.

    Parameters
    ----------
    handled : Dict[Callable: Callable]
        Module level dictionary for map from `numpy` function to implementation.

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
        """Register an `__array_function__` implementation for my array objects.
        """
        def decorator(func: MyFun) -> MyFun:
            handled[numpy_function] = func
            return func
        return decorator
    return implements


def array_function_helper(obj: _ty.Any,
                          handled: HandleMap,
                          func: NumpyFun,
                          types: _ty.Collection[_ty.Type],
                          args: ArgTuple,
                          kwargs: ArgDict) -> _ty.Any:
    """Helper function for writing __array_function__ methods

    Parameters
    ----------
    obj : Any
        The object whose `__array_function__` method was called.
    handled : HandleMap
        Dictionary mapping `numpy` functions to their custom implementations.
    func : NumpyFun
        The `numpy` function that dispatched to `obj.__array_function__`.
    types : Collection[Type[Any]]
        Collection of types of arguments from the original function call.
    args : Tuple[Any, ...]
        Positional arguments from the original function call.
    kwargs : _Dict[str, Any]
        Keyword arguments from the original function call.

    Returns
    -------
    results : Any
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
