# -*- coding: utf-8 -*-
"""
Helper for writing __array_ufunc__
"""
from typing import Tuple, Any, List


def conv_loop_in(cls, attr: str, tup: Tuple[Any]) -> (Tuple[Any], List[bool]):
    """Process inputs in an __array_ufunc__ method

    Parameters
    ----------
    cls
        The type of object that needs converting.
    attr: str
        The name of the ``cls`` attribute to use in place of ``cls``.
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
        if isinstance(obj, cls):
            out.append(getattr(obj, attr))
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
    out_args = []
    for result, outarg, cout in zip(results, outargs, conv):
        if outarg is None:
            if cout:
                conv = obj.copy()
                setattr(conv, attr, result)
                out_args.append(conv)
            else:
                out_args.append(result)
        else:
            out_args.append(outarg)
    return tuple(out_args)
