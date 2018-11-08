# -*- coding: utf-8 -*-
"""
Helper for writing __array_ufunc__
"""
from typing import Tuple, Any, List
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


def conv_loop_out(obj, attr: str, results: Tuple[Any], outargs: Tuple[Any],
                  conv: List[bool]) -> Tuple[Any]:
    """Process outputs in an __array_ufunc__ method

    Parameters
    ----------
    obj
        The template object for conversions.
    attr: str
        The name of the ``type(obj)`` attribute returned in place of class.
        Set it to '' to convert via constructor
    results: Tuple[Any]
        Tuple of outputs from ufunc
    outargs: Tuple[Any]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: List[bool]
        List of bools telling us if each input was converted.

    Returns
    -------
    results: Tuple[Any]
        New tuple of results from ufunc with conversions.
    """
    if attr:
        def converter(thing):
            """"""""
            thing_out = obj.copy()
            setattr(thing_out, attr, thing)
            return thing_out
    else:
        def converter(thing):
            """"""
            return type(obj)(thing)

    results_out = []
    for result, outarg, cout in zip(results, outargs, conv):
        if outarg is None:
            if cout:
                results_out.append(converter(result))
            else:
                results_out.append(result)
        else:
            results_out.append(outarg)
    return tuple(results_out)
