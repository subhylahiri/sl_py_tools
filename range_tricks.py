# -*- coding: utf-8 -*-
"""Tricks for manipulating ranges and slices
"""
from __future__ import annotations

import itertools
from abc import abstractmethod
from numbers import Number
from typing import Optional, Tuple, Union

from . import _iter_base as _ib
from . import arg_tricks as _ag
from . import integer_tricks as _ig
from .abc_tricks import ABCauto
from .modular_arithmetic import and_

RangeArg = Optional[_ig.Eint]
RangeArgs = Tuple[RangeArg, ...]

# =============================================================================
# ABCs & mixins
# =============================================================================


class RangeIsh(ABCauto, typecheckonly=True):
    """ABC for range-ish objects - those with start, stop, step attributes.

    Intended for instance/subclass checks only.
    """

    @property
    @abstractmethod
    def start(self) -> RangeArg:
        """Start of range"""

    @property
    @abstractmethod
    def stop(self) -> RangeArg:
        """End of range"""

    @property
    @abstractmethod
    def step(self) -> RangeArg:
        """Step between members of range"""


class RangeLike(RangeIsh, typecheckonly=True):
    """ABC for range-like objects: range-ish objects with index, count methods.

    Intended for instance/subclass checks only.
    """

    @abstractmethod
    def count(self, value: _ig.Eint) -> int:
        """return number of occurences of value"""

    @abstractmethod
    def index(self, value: _ig.Eint) -> _ig.Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """


class ContainerMixin:
    """Mixin class to add extra Collection methods to RangeIsh classes

    Should be used with `RangeCollectionMixin`
    """

    def count(self, value: _ig.Eint) -> int:
        """return number of occurences of value"""
        return int(value in self)

    def index(self, value: _ig.Eint) -> _ig.Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """
        if value not in self:
            raise ValueError(f"{value} is not in range")
        return (value - self.start) // self.step

    @abstractmethod
    def __contains__(self, arg: _ig.Eint) -> bool:
        pass


class RangeCollectionMixin(ContainerMixin):
    """Mixin class to add range-container methods to RangeIsh classes"""

    def __len__(self) -> int:
        # iterable behaviour
        if _isinf(self):
            return _ig.inf
        return len(range(*range_args_def(self)))

    def __contains__(self, arg: _ig.Eint) -> bool:
        # iterable behaviour
        if ((arg - self.start) * self.step < 0
                or (arg - self.start) % self.step != 0):
            return False
        if not _isinf(self) and (arg - self.stop) * self.step >= 0:
            return False
        return True

# =============================================================================
# Displaying ranges
# =============================================================================


def range_repr(the_range: RangeIsh, bracket: bool = True) -> str:
    """Faithful string representation of range

    Minimal string such that `'range' + range_repr(input)` evaluates to `input`

    Parameters
    ----------
    the_range : RangeIsh
        Instance to represent.
    bracket : bool, optional
        Do we enclose result in ()? default: True

    Returns
    -------
    rng_str : str
        String representing range.
    """
    if bracket:
        return f'({range_repr(the_range, False)})'
    return (_default_str(the_range.start, '{},') + str(the_range.stop)
            + _default_str(the_range.step, ',{}'))

# =============================================================================
# Extended range
# =============================================================================


class ExtendedRange(RangeCollectionMixin):
    """Combination of range and itertools.count

    Any parameter can be given as `None` and the default will be used. `stop`
    can also be `+/-inf`.

    Parameters
    ----------
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=inf*sign(step)
        value of counter at or above which the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.
    """
    start: _ig.Integral
    stop: _ig.Eint
    step: _ig.Integral
    _iter: Union[range, itertools.count]

    def __init__(self, *args, **kwds):
        super().__init__()
        self.start, self.stop, self.step = _ib.extract_slice(args, kwds)
        self.start, self.stop, self.step = range_args_def(self)
        if self.step == 0:
            raise ValueError("step cannot be 0.")
        if _isinf(self):
            self._iter = itertools.count(self.start, self.step)
        else:
            self._iter = range(self.start, self.stop, self.step)

    def index(self, value: _ig.Eint) -> _ig.Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """
        if not _isinf(self):
            return self._iter.index(value)
        return super().index(value)

    def __iter__(self) -> Union[range, itertools.count]:
        return iter(self._iter)

    def __next__(self) -> int:
        return next(self._iter)

    def __len__(self) -> int:
        if _isinf(self):
            return _ig.inf
        return len(self._iter)

    def __contains__(self, arg: _ig.Eint) -> bool:
        if not _isinf(self):
            return arg in self._iter
        return super().__contains__(arg)

    def __reversed__(self) -> ExtendedRange:
        _raise_if_no_stop(self)
        args = self.stop - self.step, self.start - self.step, -self.step
        return type(self)(*args)

    def __repr__(self) -> str:
        return "erange" + range_repr(self)

    def __getitem__(self, ind: Union[_ig.Eint, slice]
                    ) -> Union[_ig.Eint, ExtendedRange]:
        if isinstance(ind, _ig.Eint):
            val = _nth_value(self, ind)
            if val is None:
                raise IndexError(f'{ind} out of range when len={len(self)}')
            return val
        start, stop, step = range_args(ind)
        step = _ag.default(step, 1)
        if step < 0:
            return self.__reversed__()[start:stop:-step]
        start, stop, step = range_args_def(ind)
        nstart, nstop = _nth_value(self, start), _nth_value(self, stop)
        nstart, nstop = _ag.defaults((nstart, nstop), (self.start, self.stop))
        nstep = step * self.step
        return type(self)(nstart, nstop, nstep)


erange = ExtendedRange
sr_ = _ib.SliceToIter(erange)
# =============================================================================
# Range properties
# =============================================================================


def range_args(the_range: RangeIsh) -> RangeArgs:
    """Extract start, stop, step from range
    """
    return the_range.start, the_range.stop, the_range.step


def range_args_def(the_range: RangeIsh) -> RangeArgs:
    """Extract start, stop, step from range, using defaults where possible

    Also sets `stop = start + n * step` for some non-negative integer `n`
    (without changing last value) when possible.

    Parameters
    ----------
    the_range : RangeIsh
        An object that has integer attributes named `start`, `stop`, `step`,
        e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    start : int or None
        Start of range, with default 0/inf for positive/negative step.
    stop : int or None
        Past end of range, with default sign(step) * inf.
    step : int
        Increment of range, with default 1.
    """
    start, stop, step = range_args(the_range)
    step = _ag.default(step, 1)
    if step == 0:
        raise ValueError('range step cannot be zero')
    start = _ag.default(start, 0 if step > 0 else _ig.inf)
    stop = _ag.default(stop, _ig.inf * step)
    if (stop - start) * step < 0:
        stop = start
    if _ig.isfinite(stop):
        remainder = (stop - start) % step
        if remainder:
            stop += step - remainder
    return start, stop, step


def nonify_args(the_range: RangeIsh) -> RangeArgs:
    """Replace inf with None in range args
    """
    def nonify(val: RangeArg) -> RangeArgs:
        """Replace inf with None
        """
        if _ig.isinfnone(val):
            return None
        return int(val)
    return [nonify(x) for x in range_args(the_range)]

# =============================================================================
# Range tests
# =============================================================================


def is_subrange(subrange: RangeLike, the_range: RangeLike) -> bool:
    """Does range contain subrange?
    """
    return all(subrange.step % the_range.step == 0,
               subrange.start in the_range,
               subrange.stop in the_range)


def disjoint_range(rng1: RangeLike, rng2: RangeLike) -> bool:
    """Do ranges fail to overlap?
    """
    rng1, rng2 = _rectify(rng1), _rectify(rng2)
    overlap, _ = and_(rng1.start, rng1.step, rng2.start, rng2.step)
    return not ((overlap in rng1) and (overlap in rng2))


# =============================================================================
# Range arithmetic
# =============================================================================
RangeOrNum = Union[RangeIsh, Number]


def range_add(left: RangeOrNum, right: RangeOrNum) -> erange:
    """Add ranges / numbers.

    Parameters
    ----------
    left, right : RangeIsh or Number
        Arguments to add.

    Raises
    ------
    ValueError
     If `step`s are incompatible.
    """
    return _ib.arg_add(range_args, erange, left, right)


def range_sub(left: RangeOrNum, right: RangeOrNum) -> erange:
    """Subtract ranges / numbers.

    Parameters
    ----------
    left, right : RangeIsh or Number
        Arguments to subtract.any(iterable)

    Raises
    ------
    ValueError
     If `step`s are incompatible
    """
    return _ib.arg_sub(range_args, erange, left, right)


def range_mul(arg1: RangeOrNum, arg2: RangeOrNum, step: bool = True) -> erange:
    """Multiply range by a number.

    Parameters
    ----------
    left, right : RangeIsh or Number
        Arguments to multiply. Cannot both be `RangeIsh`.
    step : bool
        Also multiply step?

    Raises
    ------
    TypeError
        If neither `left` nor `right is a number.`
    """
    return _ib.arg_mul(range_args, erange, arg1, arg2, step)


def range_div(left: RangeOrNum, right: Number, step: bool = True) -> erange:
    """Divide range by a number.

    Parameters
    ----------
    left : RangeIsh or Number
        Argument to divide.
    right : Number
        Argument to divide by.
    step : bool
        Also divide step?

    Raises
    ------
    TypeError
        If `right` is not a number.
    """
    return _ib.arg_div(range_args, erange, left, right, step)

# =============================================================================
# Utilities
# =============================================================================


def _nth_value(rng: RangeIsh, ind: _ig.Eint) -> _ig.Eint:
    """Get n'th value from iterating over range"""
    start, stop, step = range_args_def(rng)
    val = (start if ind >= 0 else stop) + ind * step
    if (val - start) * step < 0 or (stop - val) * step <= 0:
        return None
    return val

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def _isinf(obj: RangeIsh) -> bool:
    """is obj.stop None/inf? Can iterable go to +/- infinity?"""
    return _ig.isinfnone(obj.stop)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


def _raise_if_no_stop(obj: RangeIsh):
    """raise ValueError if obj.stop is None/inf"""
    if _isinf(obj):
        raise ValueError("Need a finite value for stop")

# -----------------------------------------------------------------------------
# Standardising
# -----------------------------------------------------------------------------


def _rectify(the_range: RangeLike) -> erange:
    """Equivalent erange with positive step

    Raises
    ------
    ValueError
        If `length, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    the_range = erange(*range_args_def(the_range))
    if the_range.step > 0:
        return the_range
    _raise_if_no_stop(the_range)
    return range_add(erange(the_range.stop, the_range.start, -the_range.step),
                     -the_range.step)


def _default_str(optional: RangeArg, template: str) -> str:
    """Evaluate format on optional if it is not None/inf

    Parameters
    ----------
    optional : int, inf or None
        The optional argument, where `None`/`inf` indicates that the default
        value should be used instead.
    template : str
        Evaluate `template.format` on `optional`if it is not `None`/`inf`.

    Returns
    -------
    use_value : str
        Either `template.format(optional)`, if `optional` is not `None` or inf,
        or `''` if it is.
    """
    if _ig.isinfnone(optional):
        return ''
    return template.format(optional)
