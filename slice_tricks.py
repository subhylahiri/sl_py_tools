# -*- coding: utf-8 -*-
"""Tricks for manipulating slices and ranges
"""
import typing as _ty
import itertools as _it
import operator as _op
import collections.abc as _abc
from numbers import Number
from . import arg_tricks as _ag
from . import integer_tricks as _ig
from . import _iter_base as _ib

S = _ty.TypeVar('S')
NumOp = _ty.Callable[[Number, Number], Number]
SliceOrNum = _ty.Union[slice, Number]


def slice_str(*sliceobjs: slice, bracket: bool = True) -> str:
    """String representation of slice

    Converts `slice(a, b, c)` to `'a:b:c'`, `np.s_[a:b, c:]` to `'a:b,c:'`,
    `*np.s_[::c, :4]` to `'::c,:4'`, `np.s_[:]` to `':'`, etc.

    Parameters
    ----------
    sliceobj : slice or range
        Instance(s) to represent (or int, Ellipsis, hasattr{start,stop,step}).

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    if bracket:
        return '[' + slice_str(sliceobjs, bracket=False) + ']'
    if len(sliceobjs) == 0:
        return ''
    if len(sliceobjs) > 1:
        return ','.join(slice_str(s, bracket=False) for s in sliceobjs)
    sliceobj = sliceobjs[0]
    if isinstance(sliceobj, tuple):
        return slice_str(*sliceobj, bracket=False)
    if isinstance(sliceobj, int):
        return str(sliceobj)
    if sliceobj is Ellipsis:
        return '...'
    return (_ag.default_non_eval(sliceobj.start, str, '') + ':'
            + _ag.default_non_eval(sliceobj.stop, str, '')
            + _ag.default_non_eval(sliceobj.step, lambda x: f':{x}', ''))

# =============================================================================
# %%* Extended range
# =============================================================================


class ExtendedRange(_abc.Collection, _abc.Iterator):
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
        if self.step < 0:
            self.stop = _ag.default(self.stop, -1)
        else:
            self.stop = _ag.default(self.stop, _ig.inf)
        if _isinf(self):
            self._iter = _it.count(self.start, self.step)
        else:
            self.stop = _stop_bound(self.start, self.stop, self.step)
            self._iter = range(self.start, self.stop, self.step)

    def count(self, value) -> int:
        """return number of occurences of value"""
        if not _isinf(self):
            return self._iter.count(value)
        return int(value in self)

    def index(self, value):
        """return index of value.
        Raise ValueError if the value is not present.
        """
        if not _isinf(self):
            return self._iter.index(value)
        if value not in self:
            raise ValueError(f"{value} is not in range")
        return (value - self.stop) // self.step

    def indices(self, length: int = None) -> _ty.Tuple[int, ...]:
        """Start, stop, step of equivalent slice

        Parameters
        ----------
        length : int or None
            This method takes a single integer argument length and computes
            information about the slice that the object would describe if
            applied to a sequence of `length` items.

        Returns
        -------
        (start, stop, step)
            a tuple of three integers; respectively these are the start and
            stop indices and the step or stride length of the slice. Missing or
            out-of-bounds indices are handled in a manner consistent with
            regular slices.
        """
        if length is None:
            return tuple(_nonify(val) for val in slice_args(self))
        if _isinf(self):
            return slice(*self.indices()).indices(length)
        return range_to_slice(self).indices(length)

    def __iter__(self):
        return iter(self._iter)

    def __next__(self):
        return next(self._iter)

    def __len__(self):
        if _isinf(self):
            return _ig.inf
        return len(self._iter)

    def __contains__(self, arg):
        if not _isinf(self):
            return (arg in self._iter)
        return ((arg - self.start) * self.step >= 0
                and (arg - self.start) % self.step == 0)

    def __reversed__(self):
        _raise_if_no_stop(self)
        args = self.stop - self.step, self.start - self.step, -self.step
        return type(self)(*args)

    def __repr__(self):
        rpr = slice_str(self, bracket=False).replace(':', ', ')
        return f"erange({rpr})"


erange = ExtendedRange
# =============================================================================
# %%* Range conversion
# =============================================================================


def range_to_slice(the_range: range) -> slice:
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
    return slice(*slice_args(the_range))


def slice_to_range(the_slice: slice, length: int = None) -> erange:
    """Convert a slice object to an erange.

    Parameters
    ----------
    the_slice
        The `slice` to convert. or any object that has integer attributes named
        `start`, `stop` and `step` e.g. `slice`, `range`, `DisplayCount`
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
        return erange(the_slice, the_slice + 1)
    if length is not None and hasattr(the_slice, 'indices'):
        return erange(*the_slice.indices(length))
    return erange(*slice_args(the_slice))


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
    length: _ty.Optional[int]

    def __init__(self, length: int = None):
        """
        Parameters
        ----------
        length : int or None
            Replaces slice upper bound if upper bound is `None` or `> length`.
            Upper bound is `stop` if `step > 0` and `start+1` otherwise.
        """
        self.length = length

    def __getitem__(self, arg) -> erange:
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
        return slice_to_range(arg, self.length)


srange = SliceRange()
# =============================================================================
# %%* Slice properties
# =============================================================================


def slice_args(the_slice: slice) -> _ty.Tuple[_ty.Optional[int], ...]:
    """Extract start, stop, step from slice
    """
    return the_slice.start, the_slice.stop, the_slice.step


def slice_args_def(the_slice: slice) -> _ty.Tuple[_ty.Optional[int], ...]:
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


def last_value(obj, length: int = None) -> int:
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
    if obj.step > 0:
        return _slice_max(obj)
    _raise_non_determinable(obj)
    return _slice_min(obj)


def stop_step(obj, length: int = None) -> int:
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
    if obj.step > 0:
        return _slice_sup(obj)
    _raise_non_determinable(obj)
    return _slice_inf(obj)

# =============================================================================
# %%* Slice tests
# =============================================================================


def in_slice(val: int, the_slice: slice):
    """Does slice contain value?
    """
    if val is None:
        return _unbounded(the_slice)
    return val in slice_to_range(the_slice)


def is_subslice(subslice: slice, the_slice: slice):
    """Does slice contain subslice?
    """
    subslice, the_slice = _rectify(subslice), _rectify(the_slice)
    return all(subslice.step % the_slice.step == 0,
               in_slice(subslice.start, the_slice),
               in_slice(subslice.stop, the_slice))


def disjoint_slice(slc1: slice, slc2: slice):
    """Do slices fail to overlap?
    """
    from .modular_arithmetic import and_
    slc1, slc2 = _rectify(slc1), _rectify(slc2)
    overlap, step = and_(slc1.start, slc1.step, slc2.start, slc2.step)
    return not (in_slice(overlap, slc1) and in_slice(overlap, slc2))


# =============================================================================
# %%* Slice arithmetic
# =============================================================================


def slice_mul(left: SliceOrNum, right: SliceOrNum, step: bool = True) -> slice:
    """Multiply slice by a number.

    Parameters
    ----------
    step : bool
        Also multiply step?
    """
    return _all_slice_op(left, right, _op.mul, (None, step, step))


def slice_div(sliceobj: slice, other: Number, step: bool = True) -> slice:
    """divide slice by a number.

    Parameters
    ----------
    step : bool
        Also divide step?
    """
    return _all_slice_op(sliceobj, other, _op.floordiv, (None, step, None))


def slice_add(left: SliceOrNum, right: SliceOrNum) -> slice:
    """Add slices / numbers.
    """
    return _all_slice_op(left, right, _op.add, (False,)*3)


def slice_sub(left: SliceOrNum, right: SliceOrNum) -> slice:
    """Subtract slices / numbers.
    """
    result = _all_slice_op(left, right, _op.sub, (False,)*3)
    if any((isinstance(left, slice),
            not isinstance(right, slice),
            result.step is None)):
        return result
    return slice(result.start, result.stop, -result.step)

# =============================================================================
# %%* Utilities
# =============================================================================

# -----------------------------------------------------------------------------
# %%* Exceptions
# -----------------------------------------------------------------------------


def _raise_if_no_stop(obj):
    """raise ValueError if obj.stop is None/inf"""
    if _isinf(obj):
        raise ValueError("Need a finite value for stop")


def _raise_non_determinable(the_slice: slice):
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
    if _unbounded(the_slice) and the_slice.step < -1:
        raise ValueError('Must specify length or start if step < -1')


def _raise_if_none(obj):
    """raise TypeError if obj is None"""
    if obj is None:
        raise TypeError("Unsupported operation")

# -----------------------------------------------------------------------------
# %%* Tests
# -----------------------------------------------------------------------------


def _isinf(obj):
    """is obj.stop None/inf?"""
    return obj.stop is None or _ig.isinf(obj.stop)


def _unbounded(the_slice: slice) -> bool:
    """Could slice include infinity?"""
    if the_slice.step is None or the_slice.step > 0:
        return the_slice.stop is None
    if the_slice.step == 0:
        return False
    return the_slice.start is None


def _determinable(the_slice: slice, length: int = None) -> bool:
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
    the_slice = _std_slice(the_slice, length)
    return not (_unbounded(the_slice) and the_slice.step < -1)

# -----------------------------------------------------------------------------
# %%* Properties
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


def _slice_sup(the_slice: slice) -> _ty.Optional[int]:
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


def _slice_max(the_slice: slice) -> _ty.Optional[int]:
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
    return _last_val(the_slice.start, the_slice.stop, the_slice.step)


def _slice_min(the_slice: slice) -> _ty.Optional[int]:
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
    return _last_val(the_slice.start, the_slice.stop, the_slice.step)


def _slice_inf(the_slice: slice) -> _ty.Optional[int]:
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
    return _stop_bound(the_slice.start, the_slice.stop, the_slice.step)

# -----------------------------------------------------------------------------
# %%* Standardising
# -----------------------------------------------------------------------------


def _std_slice(the_slice: slice, length: int = None) -> slice:
    """Equivalent slice with default values where possible

    Also sets `stop = start + integer * step` without changing last value, if
    possible. Not possible when unbounded, unless `step == -1`.

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
    the_slice : slice
        Slice object with default values where possible.
    """
    if length is not None:
        the_slice = slice(*the_slice.indices(length))
    start, stop, step = slice_args_def(the_slice)
    if not _unbounded(the_slice):
        # if not _unbounded and step < 0, => start is not None
        # if not _unbounded and step > 0, => stop is not None
        stop = _stop_bound(start, stop, step)
    return slice(start, stop, step)


def _rectify(the_slice: slice, length: int = None) -> slice:
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
    return slice(_slice_min(the_slice), _slice_sup(the_slice), -the_slice.step)


def _nonify(val: _ty.Optional[_ig.Eint]) -> _ty.Optional[int]:
    """Replace inf with None
    """
    if val is None:
        return val
    if _ig.isfinite(val):
        return int(val)
    return None

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


def _slices_op(left: slice, right: slice, op: NumOp, step: bool) -> slice:
    """Perform operation on two slices."""
    if step:
        flex_op = _num_only(op)
        lslc_args, rslc_args = slice_args(left), slice_args(right)
        return slice(*[flex_op(s, t) for s, t in zip(lslc_args, rslc_args)])
    lstep, rstep = left.step, right.step
    if not ((lstep is None) or (rstep is None) or (lstep == rstep)):
        raise ValueError(f"incompatible steps: {lstep} and {rstep}")
    flex_op = _num_only(op)
    lslc_args, rslc_args = slice_args(left)[:2], slice_args(right)[:2]
    new_args = [flex_op(s, t) for s, t in zip(lslc_args, rslc_args)]
    new_args.append(_ag.default(lstep, rstep))
    return slice(*new_args)


def _lslice_op(sliceobj: slice, other: Number, op: NumOp, step: bool) -> slice:
    """Perform operation on slice & number."""
    flex_op = _num_only(op)
    slc_args = slice_args(sliceobj)
    new_args = [flex_op(s, other) for s in slc_args]
    if not step:
        new_args[-1] = slc_args[-1]
    return slice(*new_args)


def _rslice_op(other: Number, sliceobj: slice, op: NumOp, step: bool) -> slice:
    """Perform operation on number & slice."""
    flex_op = _num_only(op)
    slc_args = slice_args(sliceobj)
    new_args = [flex_op(other, s) for s in slc_args]
    if not step:
        new_args[-1] = slc_args[-1]
    return slice(*new_args)


def _all_slice_op(left: SliceOrNum, right: SliceOrNum, op: NumOp,
                  step: _ty.Tuple[bool, ...]) -> slice:
    """Perform operation on slices/numbers."""
    if isinstance(left, slice) and isinstance(right, slice):
        _raise_if_none(step[0])
        return _slices_op(left, right, op, step[0])
    if isinstance(left, slice):
        _raise_if_none(step[1])
        return _lslice_op(left, right, op, step[1])
    if isinstance(right, slice):
        _raise_if_none(step[2])
        return _rslice_op(left, right, op, step[2])
    return op(left, right)
