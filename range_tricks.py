# -*- coding: utf-8 -*-
"""Tricks for manipulating ranges and slices
"""
from __future__ import annotations
import typing as _ty
import itertools as _it
import operator as _op
from abc import abstractmethod
import collections.abc as _abc
from numbers import Number
from . import arg_tricks as _ag
from . import integer_tricks as _ig
from . import _iter_base as _ib
from . import abc_tricks as _ab

S = _ty.TypeVar('S')
NumOp = _ty.Callable[[Number, Number], Number]
RangeArg = _ty.Optional[_ig.Eint]
RangeArgs = _ty.Tuple[RangeArg, RangeArg, RangeArg]

# =============================================================================
# %%* ABCs
# =============================================================================


class RangeIsh(_ab.ABCauto, typecheckonly=True):
    """ABC for range-ish objects - those with start, stop, step attributes.

    Intended for instance/subclass checks only.
    """

    @property
    @abstractmethod
    def start(self) -> RangeArg:
        pass

    @property
    @abstractmethod
    def stop(self) -> RangeArg:
        pass

    @property
    @abstractmethod
    def step(self) -> RangeArg:
        pass


class RangeLike(RangeIsh, typecheckonly=True):
    """ABC for range-like objects: range-ish objects with index, count methods.

    Intended for instance/subclass checks only.
    """

    @abstractmethod
    def count(self, value: _ig.Eint) -> int:
        pass

    @abstractmethod
    def index(self, value: _ig.Eint) -> _ig.Eint:
        pass


class ContainerMixin(_abc.Container):
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
        if not _isinf(self) and (self.stop - arg) * self.step >= 0:
            return False
        return True

# =============================================================================
# %%* Displaying ranges
# =============================================================================


def range_repr(the_range: RangeIsh, bracket: bool = True) -> str:
    """Faithful string representation of range

    Minimal string such that 'range(' range_repr(input) ')' evaluates to input.

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
    return (_default_non_eval(the_range.start, lambda x: f'{x},', '')
            + str(the_range.stop)
            + _default_non_eval(the_range.step, lambda x: f',{x}', ''))

# =============================================================================
# %%* Extended range
# =============================================================================


class ExtendedRange(_abc.Iterator, RangeCollectionMixin):
    """Combination of range and itertools.count

    Any parameter can be given as `None` and the default will be used. `stop`
    can also be `+/-inf`.

    Parameters
    ----------
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default={inf,-1} if step {>0,<0}
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.
    """
    start: _ig.Integral
    stop: _ig.Eint
    step: _ig.Integral
    _iter: _ty.Union[range, _it.count]

    def __init__(self, *args, **kwds):
        super().__init__()
        self.start, self.stop, self.step = _ib.extract_slice(args, kwds)
        self.start, self.stop, self.step = range_args_def(self)
        if _isinf(self):
            self._iter = _it.count(self.start, self.step)
        else:
            self._iter = range(self.start, self.stop, self.step)

    def index(self, value: _ig.Eint) -> _ig.Eint:
        """return index of value.
        Raise ValueError if the value is not present.
        """
        if not _isinf(self):
            return self._iter.index(value)
        return super().index(value)

    def __iter__(self) -> ExtendedRange:
        return iter(self._iter)

    def __next__(self) -> int:
        return next(self._iter)

    def __len__(self) -> int:
        if _isinf(self):
            return _ig.inf
        return len(self._iter)

    def __contains__(self, arg: _ig.Eint) -> bool:
        if not _isinf(self):
            return (arg in self._iter)
        return super().__contains__(arg)

    def __reversed__(self) -> ExtendedRange:
        _raise_if_no_stop(self)
        args = self.stop - self.step, self.start - self.step, -self.step
        return type(self)(*args)

    def __repr__(self) -> str:
        return "erange" + range_repr(self)


erange = ExtendedRange
# =============================================================================
# %%* Range properties
# =============================================================================


def range_args(the_range: RangeIsh) -> RangeArgs:
    """Extract start, stop, step from range
    """
    return the_range.start, the_range.stop, the_range.step


def range_args_def(the_range: RangeIsh) -> RangeArgs:
    """Extract start, stop, step from range, using defaults where possible

    Parameters
    ----------
    the_range : RangeIsh
        An object that has integer attributes named `start`, `stop`, `step`,
        e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    start : int or None
        Start of range, with default 0.
    stop : int or None
        Past end of range, with no default.
    step : int
        Increment of range, with default 1.
    """
    start, stop, step = range_args(the_range)
    step = _ag.default(step, 1)
    if step == 0:
        raise ValueError('range step cannot be zero')
    start = _ag.default(start, 0)
    stop = _ag.default(stop, _ig.inf * step)
    if _ig.isfinite(stop):
        remainder = (stop - start) % step
        if remainder:
            stop += step - remainder
    return start, stop, step

# =============================================================================
# %%* Range tests
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
    from .modular_arithmetic import and_
    rng1, rng2 = _rectify(rng1), _rectify(rng2)
    overlap, step = and_(rng1.start, rng1.step, rng2.start, rng2.step)
    return not ((overlap in rng1) and (overlap in rng2))


# =============================================================================
# %%* Range arithmetic
# =============================================================================
RangeOrNum = _ty.Union[RangeIsh, Number]


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
    return arg_add(erange, left, right)


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
    return arg_sub(erange, left, right)


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
    return arg_mul(erange, arg1, arg2, step)


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
    return arg_div(erange, left, right, step)

# =============================================================================
# %%* Utilities
# =============================================================================

# -----------------------------------------------------------------------------
# %%* Tests
# -----------------------------------------------------------------------------


def _isinf(obj: RangeIsh) -> bool:
    """is obj.stop None/inf? Can iterable go to +/- infinity?"""
    return _ig.isinfnone(obj.stop)


# -----------------------------------------------------------------------------
# %%* Exceptions
# -----------------------------------------------------------------------------


def _raise_if_no_stop(obj: RangeIsh):
    """raise ValueError if obj.stop is None/inf"""
    if _isinf(obj):
        raise ValueError("Need a finite value for stop")


def _raise_if_none(obj: _ty.Any):
    """raise TypeError if obj is None"""
    if obj is None:
        raise TypeError("Unsupported operation")


def _raise_if_steps(left: RangeIsh, right: RangeIsh):
    """raise ValueError if steps do not match"""
    lstep, rstep = left.step, right.step
    if not ((lstep is None) or (rstep is None) or (lstep == rstep)):
        raise ValueError(f"Incompatible steps: {lstep} and {rstep}")

# -----------------------------------------------------------------------------
# %%* Standardising
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


def _default_non_eval(optional: _ty.Optional[_ig.Eint],
                      non_default_fn: _ty.Callable[[int], str],
                      default_value: str) -> str:
    """Evaluate function on optional if it is not None

    Parameters
    ----------
    optional : A or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    non_default_fn : Callable[(A)->B]
        Evaluated on `optional`if it is not `None`.
    default_value : B
        Default value for the argument, used when `optional` is `None`.

    Returns
    -------
    use_value : B
        Either `non_default_fn(optional)`, if `optional` is not `None` or
        `default_value` if it is.
    """
    if _ig.isinfnone(optional):
        return default_value
    return non_default_fn(optional)

# -----------------------------------------------------------------------------
# %%* Arithmetic
# -----------------------------------------------------------------------------


def _num_only_l(op: NumOp) -> _ty.Callable[[S, Number], S]:
    """Wrap an operator to only act on numbers
    """
    def wrapper(left: S, right: Number) -> S:
        if isinstance(left, Number):
            return op(left, right)
        return left
    return wrapper


def _num_only_r(op: NumOp) -> _ty.Callable[[Number, S], S]:
    """Wrap an operator to only act on numbers
    """
    def wrapper(left: Number, right: S) -> S:
        if isinstance(right, Number):
            return op(left, right)
        return right
    return wrapper


def _num_only(op: NumOp) -> _ty.Callable[[S, S], S]:
    """Wrap an operator to only act on numbers
    """
    return _num_only_l(_num_only_r(op))


def _wrap_op(args: RangeArgs, op: NumOp, step: bool) -> RangeArgs:
    """Perform operation on two eranges."""
    flex_op = _num_only(op)
    if step:
        return erange(*[flex_op(s, t) for s, t in zip(*args)])
    arg1, arg2 = args
    _raise_if_steps(arg1[2], arg2[2])
    lrng_args, rrng_args = [range_args(x)[:2] for x in args]
    new_args = [flex_op(s, t) for s, t in zip(arg1[:2], arg2[:2])]
    new_args.append(_ag.default(arg1[2], arg2[2]))
    return new_args


def _conv_range_args(arg):
    """return range args, True or (arg,arg,aeg), False."""
    if isinstance(arg, RangeIsh):
        return range_args(arg), True
    return (arg, arg, None), False


def _range_ops(op: NumOp, case_steps: _ty.Tuple[bool, ...], out: _ty.Type[S],
               args: _ty.Tuple[RangeOrNum]) -> S:
    """Perform operation on eranges/numbers.

    Parameters
    ----------
    args
        [left, right]
    case_steps : tuple(bool, bool, bool)
        if (both, left, right) isinstance RangeIsh, that element passed to _ops
    out
        type to convert output to
    op
        operator to use
    """
    args, is_rng = zip(*[_conv_range_args(x) for x in args])
    if not any(is_rng):
        return op(*args)
    is_rng = (all(is_rng),) + is_rng
    case_steps = case_steps[is_rng.index(True)]
    _raise_if_none(case_steps)
    return out(*_wrap_op(*args, op, case_steps))


def arg_mul(
        out: _ty.Type[S], arg1: RangeOrNum, arg2: RangeOrNum, step: bool) -> S:
    """multiply range by a number.

    Parameters
    ----------
    arg1, arg2 : RangeIsh or Number
        Arguments to multiply
    step : bool
        Also multiply step?
    """
    return _range_ops(_op.mul, (None, step, step), out, [arg1, arg2])


def arg_div(out: _ty.Type[S], arg1: RangeOrNum, arg2: Number, step: bool) -> S:
    """divide range by a number.

    Parameters
    ----------
    arg1, arg2 : RangeIsh or Number
        Arguments to divide
    step : bool
        Also divide step?
    """
    return _range_ops(_op.floordiv, (None, step, None), out, [arg1, arg2])


def arg_add(out: _ty.Type[S], arg1: RangeOrNum, arg2: RangeOrNum) -> S:
    """add ranges / numbers.

    Parameters
    ----------
    arg1, arg2 : RangeIsh or Number
        Arguments to add
    """
    return _range_ops(_op.add, (False, False, False), out, [arg1, arg2])


def arg_sub(out: _ty.Type[S], arg1: RangeOrNum, arg2: RangeOrNum) -> S:
    """subtract ranges / numbers.

    Parameters
    ----------
    arg1, arg2 : RangeIsh or Number
        Arguments to subtract
    """
    try:
        return arg_add(out, arg1, arg_mul(out, arg2, -1, True))
    except ValueError:
        return arg_add(out, arg1, arg_mul(out, arg2, -1, False))
