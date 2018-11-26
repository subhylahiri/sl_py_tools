# -*- coding: utf-8 -*-
"""
Helper for writing __array_ufunc__
"""
import itertools
from typing import Tuple, Any, List, Optional, Sequence
import numpy as np


def conv_loop_in(obj_typ, tup: Tuple[Any]) -> (Tuple[Any], List[bool]):
    """Process inputs in an __array_ufunc__ method

    Parameters
    ----------
    obj_typ
        The type of object that needs converting via ``view`` method.
    tup: Tuple[Any]
        Tuple of inputs to ufunc (or ``out`` argument)
    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    out = []
    conv = []
    for obj in tup:
        if isinstance(obj, obj_typ):
            out.append(obj.view(np.ndarray))
            conv.append(True)
        else:
            out.append(obj)
            conv.append(False)
    return tuple(out), conv


def conv_loop_out(obj, attr: Optional[str], results: Tuple[Any],
                  outputs: Tuple[Any], conv: Sequence[bool] = ()) -> Tuple[Any]:
    """Process outputs in an __array_ufunc__ method

    Parameters
    ----------
    obj
        The template object for conversions.
    attr: str, None
        The name of the ``type(obj)`` attribute returned in place of class.
        Set it to '' to convert via constructor, or None for view method.
    results: Tuple[Any]
        Tuple of outputs from ufunc
    outputs: Tuple[Any]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[Any]
        New tuple of results from ufunc with conversions.

    Notes
    -----
    If ``attr`` is specified, it will try to use ``obj.copy(attr=result)``.
    If that fails, it will use ``obj.copy()`` followed by
    ``setattr(newobj, attr, result)``.
    """
    if attr:
        def converter(thing):
            """"""""
            try:
                return obj.copy(**{attr: thing})
            except TypeError:
                pass
            thing_out = obj.copy()
            setattr(thing_out, attr, thing)
            return thing_out
    elif attr is None:
        def converter(thing):
            """"""
            return thing.view(type(obj))
    else:
        def converter(thing):
            """"""
            return type(obj)(thing)

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
    return tuple(results_out)
