# -*- coding: utf-8 -*-
"""Tricks for manipulating integers
"""
import math
from math import floor, ceil, trunc
from typing import Optional
from numbers import Number, Real, Integral
# from .arg_tricks import default


def ceil_divide(numerator: Number, denominator: Number) -> int:
    """Ceiling division
    """
    return ceil(numerator / denominator)


def trunc_divide(numerator: Number, denominator: Number) -> int:
    """Truncated division
    """
    return trunc(numerator / denominator)


def round_divide(numerator: Number,
                 denominator: Number,
                 ndigits: Optional[int] = None) -> Number:
    """Rounded division
    """
    return round(numerator / denominator, ndigits)

