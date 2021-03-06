# -*- coding: utf-8 -*-
"""Tricks for manipulating slices and ranges
"""
from __future__ import annotations

from abc import abstractmethod
from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union

import sl_py_tools._iter_base as _ib
import sl_py_tools.integer_tricks as _ig
import sl_py_tools.range_tricks as _rt
from .arg_tricks import default as _default
from .arg_tricks import eval_or_default as _default_neval
from .containers import ZipSequences, tuplify
from .modular_arithmetic import and_
from .range_tricks import RangeIsh as SliceIsh

SliceArg = Optional[int]
SliceArgs = Tuple[SliceArg, SliceArg, SliceArg]

# =============================================================================
# ABCs & mixins
# =============================================================================


class SliceLike(SliceIsh, typecheckonly=True):
    """ABC fror slice-like objects: slice-ish objects with an indices method.

    Intended for instance/subclass checks only.
    """

    @abstractmethod
    def indices(self, length: int) -> SliceArgs:
        """Start, stop, step of equivalent slice.
        """


class ContainerMixin(_rt.ContainerMixin):
    """Mixin class to add extra Collection methods to SliceIsh classes

    Should be used with `SliceCollectionMixin` or `RangeCollectionMixin`
    """

    @abstractmethod
    def __contains__(self, arg: _ig.Eint) -> bool:
        pass

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
            return tuple(_rt.nonify_args(self))
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


SliceAble = Union[SliceIsh, SliceArgs]
# =============================================================================
# Displaying slices
# =============================================================================


def slice_str(*sliceobjs: SliceIsh, bracket: bool = True) -> str:
    """String representation of slice(s)

    Converts `slice(a, b, c)` to `'[a:b:c]'`, `np.s_[a:b, c:]` to `'[a:b,c:]'`,
    `*np.s_[::c, :4]` to `'[::c,:4]'`, `np.s_[:]` to `'[:]'`, etc.
    Also accepts `int`s and `Ellipses` as parameters. Slice arguments can be
    `int`s, `None`, or anything that can be converted to `str`.

    Parameters
    ----------
    sliceobj : SliceIsh
        Instance(s) to represent (or int, Ellipsis).
    bracket : bool, optional, keyword only
        Do we enclose result in []? default: True

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    def func(sliceobj: SliceIsh) -> str:
        """Format a single slice
        """
        return (_default_neval(sliceobj.start, str, '') + ':'
                + _default_neval(sliceobj.stop, str, '')
                + _default_neval(sliceobj.step, lambda x: f':{x}', ''))
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
    bracket : bool, optional, keyword only
        Do we enclose result in ()? default: True

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    def func(sliceobj: SliceIsh) -> str:
        """Format a single slice
        """
        return _rt.range_repr(sliceobj, False)
    if bracket:
        return _slice_disp(func, sliceobjs, '({})')
    return _slice_disp(func, sliceobjs)

# =============================================================================
# Range conversion
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
    return slice(*_rt.nonify_args(the_range))


def slice_to_range(the_slice: SliceIsh, length: int = None,
                   default_slice: bool = True) -> _rt.erange:
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
    default_slice : bool
        Use slice conventions for default arguments? Default: True

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
    if default_slice:
        # enforce slice defaults
        return _rt.erange(*slice_args_def(the_slice))
    # enforce range defaults
    return _rt.erange(*slice_args(the_slice))


class SliceRange(_ib.SliceToIter):
    """Class for converting a slice to a range.

    You can build a `range` for iteration by calling `srange[start:stop:step]`,
    where `srange` is an instance of `SliceRange`.

    Parameters
    ----------
    length : int or None
        Replaces slice upper bound if upper bound is `None` or `> length`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.
    default_slice : bool
        Use slice conventions for default arguments? Default: True

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
    default_slice: bool

    def __init__(self, length: int = None, default_slice: bool = True):
        """
        Parameters
        ----------
        length : int or None
            Replaces slice upper bound if upper bound is `None` or `> length`.
            Upper bound is `stop` if `step > 0` and `start+1` otherwise.
        default_slice : bool
            Use slice conventions for default arguments? Default: True
        """
        super().__init__(slice_to_range, 0, tuple)
        self.length = length
        self.default_slice = default_slice

    def __getitem__(self, arg) -> _rt.erange:
        """
        Parameters
        ----------
        the_slice
            The `slice` to convert.
        length : int or None, Optional
            Replaces slice upper bound if upper bound is `None` or `> length`.
            Upper bound is `stop` if `step > 0` and `start+1` otherwise.
        default_slice : bool, Optional
            Use slice conventions for default arguments? Default: True

        Returns
        -------
        the_range : erange
            `erange` object with `start`, `stop` and `step` from `the_slice`.
        """
        arg = _ib.tuplify(arg)
        if len(arg) < 1:
            arg += (slice(None),)
        if len(arg) < 2:
            arg += (self.length,)
        if len(arg) < 3:
            arg += (self.default_slice,)
        return super().__getitem__(arg)


srange = SliceRange()


class Batch(_rt.ExtendedRange):
    """Iterator that yields slices covering each step

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
    # step of yielded slices
    _slice_step: int

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._slice_step = self.step // abs(self.step)

    def __iter__(self) -> Batch:
        self._iter = iter(self._iter)
        return self

    def __next__(self) -> slice:
        count = next(self._iter)
        return slice(count, count + self.step, self._slice_step)

    def __reversed__(self) -> Batch:
        obj = super.__reversed__()
        obj.start += self.step - self._slice_step
        obj.stop += self.step - self._slice_step
        return obj

    def __repr__(self) -> str:
        return type(self).__name__ + slice_repr(self)

    def __contains__(self, arg: slice) -> bool:
        start, stop, step = slice_args_def(arg)
        if step * self._slice_step < 0:
            start, stop, step = stop - step, start - step, -step
        if step == self._slice_step:
            return (in_slice(start, self) and in_slice(start, self)
                    and stop - start == self.step)
        return False


class Batched:
    """Iterator that yields slices of sequences

    Similar to `zip` object, except at each iteration it yields a slice of the
    sequences covering that step.

    Parameters
    ----------
    step : int
        increment of counter after each loop.
    sequence1, sequence2, ...
        sequences to iterate over
    usemax : bool, keyword only, default=True
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.

    Yields
    ------
    sequence1[i:i+step], sequence2[i:i+step], ...
        slice(s) of the sequence(s) that starts at current counter and stops at
        the next value with step size 1.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for xx, yy in Batched(10, x, y):
    >>>     yy[...] = np.linalg.eigvals(xx)
    """
    _seqs: ZipSequences
    _counter: Batch

    def __init__(self, step: int, *args, **kwds):
        self._seqs = ZipSequences(*args, **kwds)
        self._counter = Batch(0, len(self._seqs), step)

    def __iter__(self) -> Batched:
        self._counter = iter(self._counter)
        return self

    def __next__(self) -> Sequence:
        count = next(self._counter)
        return self._seqs[count]

    def __reversed__(self) -> Batched:
        self._counter = reversed(self._counter)
        return self

    def __len__(self) -> int:
        return len(self._counter)

    def __repr__(self) -> str:
        return type(self).__name__ + repr(self._seqs)[12:]

    def __str__(self) -> str:
        seqs = ','.join(type(s).__name__ for s in self._seqs)
        return type(self).__name__ + f'({seqs})'


class BatchEnum(Batched):
    """Iterator that yields slices covering each step and slices of sequences

    Similar to `enumerate` object, except at each iteration it yields a slice
    of the sequences covering that step and the corresponding slice(s) of the
    sequence(s).

    Parameters
    ----------
    step : int
        increment of counter after each loop.
    sequence1, sequence2, ...
        sequences to iterate over
    usemax : bool, keyword only, default=True
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.

    Yields
    ------
    batch_slice
        slice object that starts at current counter and stops at the next value
        with step size 1.
    sequence1[i:i+step], sequence2[i:i+step], ...
        slice(s) of the sequence(s) that starts at current counter and stops at
        the next value with step size 1.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for ss, xx in batchenum(10, x):
    >>>     y[ss] = np.linalg.eigvals(xx)
    """
    def __next__(self) -> Sequence:
        count = next(self._counter)
        return (count,) + tuplify(self._seqs[count])

    def __contains__(self, arg: slice) -> bool:
        return arg in self._counter


# =============================================================================
# Slice properties
# =============================================================================


def slicify(*args: SliceAble) -> SliceIsh:
    """Convert start, stop, step to slice, if necessary
    """
    if len(args) == 1 and isinstance(args[0], SliceIsh):
        return args[0]
    if len(args) > 3:
        raise TypeError(f"slices only take 3 arguments. Got {len(args)}.")
    return slice(*args)


def unslicify(*args: SliceAble, nones: bool = False) -> SliceArgs:
    """Convert slice to start, stop, step, if necessary
    """
    if len(args) == 1 and isinstance(args[0], SliceIsh):
        return slice_args_undef(args[0]) if nones else slice_args_def(args[0])
    if len(args) > 3:
        raise TypeError(f"slices only take 3 arguments. Got {len(args)}.")
    return args


def slice_args(the_slice: SliceIsh) -> SliceArgs:
    """Extract start, stop, step from slice
    """
    return the_slice.start, the_slice.stop, the_slice.step


def slice_args_def(the_slice: SliceIsh) -> SliceArgs:
    """Extract start, stop, step from slice, using defaults for None

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
    step = _default(step, 1)
    if step > 0:
        start = _default(start, 0)
    elif step < 0:
        stop = _default(stop, -1)
    else:
        raise ValueError('slice step cannot be zero')
    return start, stop, step


def slice_args_undef(the_slice: SliceIsh) -> SliceArgs:
    """Extract start, stop, step from slice, using None for defaults

    Parameters
    ----------
    the_slice : slice
        An object that has integer attributes named `start`, `stop`, `step`,
        e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    start : int or None
        Start of slice, with 0 -> None if `step` > 0.
    stop : int or None
        Past end of slice, with -1 -> None if `step` < 0.
    step : int
        Increment of slice, with 1 -> None.
    """
    start, stop, step = slice_args(the_slice)
    if step is None or step > 0:
        start = None if start in {None, 0} else start
    elif step is not None and step < 0:
        stop = None if stop in {None, -1} else stop
    step = None if step in {None, 1} else step
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
# Slice tests
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
    slc1, slc2 = _rectify(slc1), _rectify(slc2)
    overlap, _ = and_(slc1.start, slc1.step, slc2.start, slc2.step)
    return not (in_slice(overlap, slc1) and in_slice(overlap, slc2))


# =============================================================================
# Slice arithmetic
# =============================================================================
SliceOrNum = Union[SliceIsh, Number]
slice_add, slice_sub, slice_mul, slice_div = _ib.arg_ops(slice_args, slice)


def intersect(slc1: SliceLike, slc2: SliceLike) -> slice:
    """Do slices fail to overlap?
    """
    slc1, slc2 = _rectify(slc1), _rectify(slc2)
    overlap, step = and_(slc1.start, slc1.step, slc2.start, slc2.step)
    if _ig.isnan(overlap):
        return None
    return _rectify(slice(overlap, _min(slc1.stop, slc2.stop), step))


# =============================================================================
# Utilities
# =============================================================================


def _max(left: SliceArg, right: SliceArg) -> SliceArg:
    """Max of two slice args"""
    if left is None or right is None:
        return None
    return max(left, right)


def _min(left: SliceArg, right: SliceArg) -> SliceArg:
    """Min of two slice args"""
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def _unbounded(the_slice: SliceIsh, length: int = None) -> bool:
    """Could slice include +infinity?"""
    if not _ig.isinfnone(length):
        return False
    if the_slice.step is None or the_slice.step > 0:
        return _ig.isinfnone(the_slice.stop)
    if the_slice.step == 0:
        return False
    return _ig.isinfnone(the_slice.start)


def _indeterminable(the_slice: SliceLike, length: int = None) -> bool:
    """Is lowest value in slice indeterminable?

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
        False if lowest value in slice is determined.
    """
    return _unbounded(the_slice, length) and _default(the_slice.step, 1) < -1

# -----------------------------------------------------------------------------
# Exceptions
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
    if _indeterminable(the_slice):
        raise ValueError('Must specify length or start if step < -1')

# -----------------------------------------------------------------------------
# Properties from args
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
# Properties
# -----------------------------------------------------------------------------


def _slice_sup(the_slice: SliceIsh) -> SliceArg:
    """Smallest value one step larger than slice, or None

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
    Exact unless _indeterminable.
    """
    if the_slice.step > 0:
        return the_slice.start
    if the_slice.start is None:
        # Lies between the following:
        return the_slice.stop - the_slice.step
        # return the_slice.stop + 1
    return _last_val(*slice_args(the_slice))


def _slice_inf(the_slice: SliceIsh) -> SliceArg:
    """Lower bound on Largest value one step smaller than slice

    Assume standardised:
        `start,stop` can only be `None` if _unbounded.
        `step` is not `None`.
        `stop = start + integer * step` unless _unbounded.
    Exact unless _indeterminable.
    """
    if the_slice.step > 0:
        return the_slice.start - the_slice.step
    if the_slice.start is None:
        # Lies between the following:
        # return the_slice.stop
        return the_slice.stop + the_slice.step + 1
    return _stop_bound(*slice_args(the_slice))

# -----------------------------------------------------------------------------
# Standardising
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
    start, stop, step = slice_args(the_slice)
    return slice(stop - step, start - step, -step)

# -----------------------------------------------------------------------------
# Displaying
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
    if len(sliceobjs) != 1:
        return ','.join(_slice_disp(func, s) for s in sliceobjs)
    sliceobj = sliceobjs[0]
    if isinstance(sliceobj, (tuple, list)):
        # in case we forgot to unpack a tuple originally
        return _slice_disp(func, sliceobj)
    if sliceobj is Ellipsis:
        return '...'
    if isinstance(sliceobj, SliceIsh):
        return func(sliceobj)
    # if isinstance(sliceobj, int):
    #     return str(sliceobj)
    return str(sliceobj)
