# -*- coding: utf-8 -*-
"""Tricks for manipulating slices and ranges
"""
from __future__ import annotations
from abc import abstractmethod
from numbers import Number
from typing import Optional, Tuple, Union, Callable

from . import _iter_base as _ib
from . import range_tricks as _rt
from . import arg_tricks as _ag
from . import integer_tricks as _ig

SliceArg = Optional[int]
SliceArgs = Tuple[SliceArg, SliceArg, SliceArg]

# =============================================================================
# %%* ABCs & mixins
# =============================================================================


class SliceIsh(_rt.RangeIsh, typecheckonly=True):
    """ABC for slice-ish objects - those with start, stop, step attributes.

    Intended for instance/subclass checks only.
    """


class SliceLike(SliceIsh, typecheckonly=True):
    """ABC fror slice-like objects: slice-ish objects with an indices method.

    Intended for instance/subclass checks only.
    """

    @abstractmethod
    def indices(self, length: int) -> SliceArgs:
        pass


class ContainerMixin(_rt.ContainerMixin):
    """Mixin class to add extra Collection methods to SliceIsh classes

    Should be used with `SliceCollectionMixin` or `RangeCollectionMixin`
    """

    def indices(self, length: int = None) -> SliceArgs:
        """Start, stop, step of equivalent slice

        This method takes a single integer argument length and computes
        information about the slice that the object would describe if
        applied to a sequence of `length` items.

        Parameters
        ----------
        length : int or None
            Length of sequence that equivalent `slice` is applied to.

        Returns
        -------
        (start, stop, step) : int
            a tuple of three integers; the start and stop indices and the step
            or stride length of the slice. Missing or out-of-bounds indices are
            handled in a manner consistent with regular slices.
        """
        if length is None:
            return tuple(_nonify_args(self))
        return range_to_slice(self).indices(length)


class SliceCollectionMixin(ContainerMixin):
    """Mixin class to add slice-container methods to SliceIsh classes"""

    def __len__(self) -> int:
        # slice behaviour
        std = _std_slice(self)
        if _unbounded(std):
            return _ig.inf
        return len(range(*slice_args(std)))

    def __contains__(self, arg: _ig.Eint) -> bool:
        # slice behaviour
        return in_slice(arg, self)


# =============================================================================
# %%* Displaying slices
# =============================================================================


def slice_str(*sliceobjs: SliceIsh, bracket: bool = True) -> str:
    """String representation of slice(s)

    Converts `slice(a, b, c)` to `'[a:b:c]'`, `np.s_[a:b, c:]` to `'[a:b,c:]'`,
    `*np.s_[::c, :4]` to `'[::c,:4]'`, `np.s_[:]` to `'[:]'`, etc.

    Parameters
    ----------
    sliceobj : SliceIsh
        Instance(s) to represent (or int, Ellipsis).
    bracket : bool, optional
        Do we enclose result in []? default: True

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    def func(sliceobj):
        """Format a single slice
        """
        return (_ag.default_non_eval(sliceobj.start, str, '') + ':'
                + _ag.default_non_eval(sliceobj.stop, str, '')
                + _ag.default_non_eval(sliceobj.step, lambda x: f':{x}', ''))
    if bracket:
        return _slice_disp(func, sliceobjs, '[{}]')
    return _slice_disp(func, sliceobjs)


def slice_repr(*sliceobjs: SliceIsh, bracket: bool = True) -> str:
    """Faithful string representation of slice(s)

    Minimal string such that 'slice(' slice_repr(input) ')' evaluates to input.

    Parameters
    ----------
    sliceobj : slice or range
        Instance(s) to represent (or int, Ellipsis, hasattr{start,stop,step}).
    bracket : bool, optional
        Do we enclose result in ()? default: True

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    def func(sliceobj):
        """Format a single slice
        """
        return _rt.range_repr(sliceobj, False)
    if bracket:
        return _slice_disp(func, sliceobjs, '({})')
    return _slice_disp(func, sliceobjs)

# =============================================================================
# %%* Range conversion
# =============================================================================


def range_to_slice(the_range: SliceIsh) -> slice:
    """Convert a range object to a slice.

    Parameters
    ----------
    the_range
        The `range` to convert, or any object that has integer/None attributes
        named `start`, `stop` and `step`, e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    sliceobj
        `slice` object with `start`, `stop` and `step` taken from `the_range`.
    """
    return slice(*_nonify_args(the_range))


def slice_to_range(the_slice: SliceIsh, length: int = None) -> _rt.erange:
    """Convert a slice object to an erange.

    Parameters
    ----------
    the_slice : slice
        The `slice` to convert. or any object that has integer attributes named
        `start`, `stop` and `step` and an `indices` method that takes a single
        integer argument and returns (`start`, `stop`, `step`).
        e.g. `slice`, `erange`, `DisplayCount`
    length : int or None
        Replaces upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    rangeobj : erange
        `erange` object with `start`, `stop` and `step` taken from `the_slice`.

    Notes
    -----
    Negative `start`, `stop` interpreted relative to `length` if it is not
    `None` and relative to `0` otherwise.
    """
    if isinstance(the_slice, int):
        return _rt.erange(the_slice, the_slice + 1)
    # if length is not None and hasattr(the_slice, 'indices'):
    if length is not None and isinstance(the_slice, SliceLike):
        return _rt.erange(*the_slice.indices(length))
    # enforce slice defaults
    # return erange(*slice_args_def(the_slice))
    # enforce range defaults
    return _rt.erange(*slice_args(the_slice))


class SliceRange():
    """Class for converting a slice to a range.

    You can build a `range` for iteration by calling `srange[start:stop:step]`,
    where `srange` is an instance of `SliceRange`.

    Parameters
    ----------
    length : int or None
        Replaces slice upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    the_range : erange
        `erange` object with `start`, `stop` and `step` taken from `the_slice`.

    Notes
    -----
    Negative `start`, `stop` interpreted relative to `length` if it is not
    `None` and relative to `0` otherwise.
    """
    length: Optional[int]

    def __init__(self, length: int = None):
        """
        Parameters
        ----------
        length : int or None
            Replaces slice upper bound if upper bound is `None` or `> length`.
            Upper bound is `stop` if `step > 0` and `start+1` otherwise.
        """
        self.length = length

    def __getitem__(self, arg) -> _rt.erange:
        """
        Parameters
        ----------
        the_slice
            The `slice` to convert.

        Returns
        -------
        the_range : range
            `range` object with `start`, `stop` and `step` from `the_slice`.
        """
        if isinstance(arg, tuple):
            return slice_to_range(*arg)
        return slice_to_range(arg, self.length)


srange = SliceRange()
# =============================================================================
# %%* Slice properties
# =============================================================================


def slice_args(the_slice: SliceIsh) -> SliceArgs:
    """Extract start, stop, step from slice
    """
    return _rt.range_args(the_slice)


def slice_args_def(the_slice: SliceIsh) -> SliceArgs:
    """Extract start, stop, step from slice, using defaults where possible

    Parameters
    ----------
    the_slice : slice
        An object that has integer attributes named `start`, `stop`, `step`,
        e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    start : int or None
        Start of slice, with default 0 if `step` > 0.
    stop : int or None
        Past end of slice, with default -1 if `step` < 0.
    step : int
        Increment of slice, with default 1.
    """
    start, stop, step = slice_args(the_slice)
    step = _ag.default(step, 1)
    if step > 0:
        start = _ag.default(start, 0)
    elif step < 0:
        stop = _ag.default(stop, -1)
    else:
        raise ValueError('slice step cannot be zero')
    return start, stop, step


def last_value(obj: SliceIsh, length: int = None) -> int:
    """Last value in range

    Parameters
    ----------
    obj
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`.
    length : int or None
        Replaces upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    val : int or None
        Last entry in iterable or slice, taking `start` and `step` into account
        or `None` if `step > 0` and `length, stop` are `None`.

    Raises
    ------
    ValueError
        If `length, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    obj = _std_slice(obj, length)
    _raise_non_determinable(obj)
    return obj.stop - obj.step


def stop_step(obj: SliceLike, length: int = None) -> int:
    """One step beyond last value in range

    Parameters
    ----------
    obj
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`.
    length : int or None
        Replaces upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    stop : int or None
        Sets `stop = start + integer * step` without changing last value,
        or `None` if `step > 0` and `length, stop` are `None`.

    Raises
    ------
    ValueError
        If `length, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    obj = _std_slice(obj, length)
    _raise_non_determinable(obj)
    return obj.stop

# =============================================================================
# %%* Slice tests
# =============================================================================


def in_slice(val: SliceArg, the_slice: SliceLike) -> bool:
    """Does slice contain value?
    """
    if val is None:
        return _unbounded(the_slice)
    _raise_non_determinable(the_slice)
    return val in slice_to_range(_std_slice(the_slice))


def is_subslice(subslice: SliceLike, the_slice: SliceLike) -> bool:
    """Does slice contain subslice?
    """
    subslice, the_slice = _rectify(subslice), _rectify(the_slice)
    return all(subslice.step % the_slice.step == 0,
               in_slice(subslice.start, the_slice),
               in_slice(subslice.stop, the_slice))


def disjoint_slice(slc1: SliceLike, slc2: SliceLike) -> bool:
    """Do slices fail to overlap?
    """
    from .modular_arithmetic import and_
    slc1, slc2 = _rectify(slc1), _rectify(slc2)
    overlap, step = and_(slc1.start, slc1.step, slc2.start, slc2.step)
    return not (in_slice(overlap, slc1) and in_slice(overlap, slc2))


# =============================================================================
# %%* Slice arithmetic
# =============================================================================
SliceOrNum = Union[SliceIsh, Number]


def slice_add(left: SliceOrNum, right: SliceOrNum) -> slice:
    """Add slices / numbers.

    Parameters
    ----------
    left, right : SliceIsh or Number
        Arguments to add.

    Raises
    ------
    ValueError
     If `step`s are incompatible.
    """
    return _ib.arg_add(slice_args, slice, left, right)


def slice_sub(left: SliceOrNum, right: SliceOrNum) -> slice:
    """Subtract slices / numbers.

    Parameters
    ----------
    left, right : RangeIsh or Number
        Arguments to subtract.

    Raises
    ------
    ValueError
     If `step`s are incompatible
    """
    return _ib.arg_sub(slice_args, slice, left, right)


def slice_mul(left: SliceOrNum, right: SliceOrNum, step: bool = True) -> slice:
    """Multiply slice by a number.

    Parameters
    ----------
    left, right : SliceIsh or Number
        Arguments to multiply. Cannot both be `SliceIsh`.
    step : bool
        Also multiply step?

    Raises
    ------
    TypeError
        If neither `left` nor `right is a number.`
    """
    return _ib.arg_mul(slice_args, slice, left, right, step)


def slice_div(left: SliceIsh, right: Number, step: bool = True) -> slice:
    """divide slice by a number.

    Parameters
    ----------
    left : SliceIsh or Number
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
    return _ib.arg_div(slice_args, slice, left, right, step)

# =============================================================================
# %%* Utilities
# =============================================================================

# -----------------------------------------------------------------------------
# %%* Tests
# -----------------------------------------------------------------------------


def _unbounded(the_slice: SliceIsh) -> bool:
    """Could slice include +infinity?"""
    if the_slice.step is None or the_slice.step > 0:
        return _ig.isinfnone(the_slice.stop)
    if the_slice.step == 0:
        return False
    return _ig.isinfnone(the_slice.start)


def _determinable(the_slice: SliceLike, length: int = None) -> bool:
    """Is lowest value in slice determinable?

    Parameters
    ----------
    the_slice : slice
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`
    length : int or None
        Replaces upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    determinable : bool
        True if lowest value in slice is determined.
    """
    return not (_unbounded(the_slice) and _ag.default(the_slice.step, 1) < -1)

# -----------------------------------------------------------------------------
# %%* Exceptions
# -----------------------------------------------------------------------------


def _raise_non_determinable(the_slice: SliceIsh):
    """Raise if lowest value in slice is not determinable.

    Parameters
    ----------
    the_slice : slice
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`.

        Assume standardised:
            `start,stop` can only be `None` if _unbounded.
            `step` is not `None`.
            `stop = start + integer * step` unless _unbounded.

    Raises
    ------
    ValueError
        If lowest value in slice is not determined.
    """
    if not _determinable(the_slice):
        raise ValueError('Must specify length or start if step < -1')

# -----------------------------------------------------------------------------
# %%* Properties from args
# -----------------------------------------------------------------------------


def _last_val(start: int, stop: int, step: int) -> int:
    """Last value in range, assuming all of start, stop, step are given

    Parameters
    ----------
    start, stop, step : int
        Parameters & attributes of a `slice`, `range`, etc.

    Returns
    -------
    val : int
        Last entry in iterable or slice, taking `start` and `step` into
        account, guarantees `val = start + integer * step` without changing
        last value.
    """
    remainder = (stop - start) % step
    if remainder:
        return stop - remainder
    return stop - step


def _stop_bound(start: int, stop: int, step: int) -> int:
    """First value after range, assuming all of start, stop, step are given

    Parameters
    ----------
    start, stop, step : int
        Parameters & attributes of a `slice`, `range`, etc.

    Returns
    -------
    stop : int
        Last+1 entry in iterable or slice, taking `start` and `step` into
        account, guarantees `stop = start + integer * step` without changing
        last value.
    """
    return _last_val(start, stop, step) + step

# -----------------------------------------------------------------------------
# %%* Properties
# -----------------------------------------------------------------------------


def _slice_sup(the_slice: SliceIsh) -> SliceArg:
    """Smallest value one step after slice, or None

    Assume standardised:
        `start,stop` can only be `None` if _unbounded.
        `step` is not `None`.
        `stop = start + integer * step` unless _unbounded.
    """
    if _unbounded(the_slice):
        return None
    if the_slice.step > 0:
        return the_slice.stop
    return the_slice.start - the_slice.step


def _slice_max(the_slice: SliceIsh) -> SliceArg:
    """Largest value in slice, or None

    Assume standardised:
        `start,stop` can only be `None` if _unbounded.
        `step` is not `None`.
        `stop = start + integer * step` unless _unbounded.
    """
    if _unbounded(the_slice):
        return None
    if the_slice.step <= 0:
        return the_slice.start
    return _last_val(*slice_args(the_slice))


def _slice_min(the_slice: SliceIsh) -> SliceArg:
    """Upper bound on smallest value in slice

    Assume standardised:
        `start,stop` can only be `None` if _unbounded.
        `step` is not `None`.
        `stop = start + integer * step` unless _unbounded.
    Exact if _determinable.
    """
    if the_slice.step > 0:
        return the_slice.start
    if the_slice.start is None:
        # Lies between the following:
        return the_slice.stop - the_slice.step
        # return the_slice.stop + 1
    return _last_val(*slice_args(the_slice))


def _slice_inf(the_slice: SliceIsh) -> SliceArg:
    """Lower bound on Largest value one step before slice

    Assume standardised:
        `start,stop` can only be `None` if _unbounded.
        `step` is not `None`.
        `stop = start + integer * step` unless _unbounded.
    Exact if _determinable.
    """
    if the_slice.step > 0:
        return the_slice.start - the_slice.step
    if the_slice.start is None:
        # Lies between the following:
        # return the_slice.stop
        return the_slice.stop + the_slice.step + 1
    return _stop_bound(*slice_args(the_slice))

# -----------------------------------------------------------------------------
# %%* Standardising
# -----------------------------------------------------------------------------


def _std_slice(the_slice: SliceLike, length: int = None) -> slice:
    """Equivalent slice with default values where possible

    Also sets `stop = start + integer * step` without changing last value, if
    possible. Not possible when unbounded, unless `step == -1`.

    Parameters
    ----------
    the_slice : SliceLike
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`.
        If `length is None`, `SliceIsh` is ok.
    length : int or None
        Replaces upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    the_slice : slice
        Slice object with default values where possible.
    """
    if length is not None:
        the_slice = slice(*the_slice.indices(length))
    start, stop, step = slice_args_def(the_slice)
    if not _unbounded(the_slice):
        # step > 0 => start is not None, and not unbounded => stop  is not None
        # step < 0 => stop  is not None, and not unbounded => start is not None
        stop = _stop_bound(start, stop, step)
    return slice(start, stop, step)


def _rectify(the_slice: SliceLike, length: int = None) -> slice:
    """Equivalent slice with positive step

    Raises
    ------
    ValueError
        If `length, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    the_slice = _std_slice(the_slice, length)
    if the_slice.step > 0:
        return the_slice
    _raise_non_determinable(the_slice)
    the_slice = slice_sub(the_slice, the_slice.step)
    return slice(the_slice.stop, the_slice.start, -the_slice.step)


def _nonify_args(the_range: _rt.RangeIsh) -> SliceArgs:
    """Replace inf with None in range args
    """
    def _nonify(val: _rt.RangeArg) -> SliceArg:
        """Replace inf with None
        """
        if _ig.isinfnone(val):
            return None
        return int(val)
    return [_nonify(x) for x in slice_args(the_range)]

# -----------------------------------------------------------------------------
# %%* Displaying
# -----------------------------------------------------------------------------


def _slice_disp(func: Callable[[SliceIsh], str],
                sliceobjs: Tuple[SliceIsh, ...],
                bracket: str = '') -> str:
    """String representation of slice(s)

    Parameters
    ----------
    func : Callable[SliceIsh -> str]
        Function to convert a single slice to a string.
    sliceobjs : tuple(SliceIsh)
        Instance(s) to represent (or int, Ellipsis, hasattr{start,stop,step}).
    bracket : str, optional
        If nonempty, String whose `format` method to apply, e.g. '[{}]'.
        default: ''.

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    if bracket:
        return bracket.format(_slice_disp(func, sliceobjs))
    if len(sliceobjs) == 0:
        return ''
    if len(sliceobjs) > 1:
        return ','.join(_slice_disp(func, s) for s in sliceobjs)
    sliceobj = sliceobjs[0]
    if isinstance(sliceobj, tuple):
        # in case we forgot to unpack a tuple originally
        return _slice_disp(func, sliceobj)
    if isinstance(sliceobj, int):
        return str(sliceobj)
    if sliceobj is Ellipsis:
        return '...'
    return func(sliceobj)
