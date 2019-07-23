# -*- coding: utf-8 -*-
"""Tricks for manipulating slices
"""
import typing as _ty
from . import arg_tricks as _ag


def slice_str(*sliceobjs: slice) -> str:
    """String representation of slice

    Converts `slice(a, b, c)` to `'a:b:c'`, `np.s_[a:b, c:]` to `'a:b, c:'`,
    `*np.s_[::c, :4]` to `'::c, :4'`, `np.s_[:]` to `':'`, etc.

    Parameters
    ----------
    sliceobj : slice or range
        Slice instance(s) to represent.

    Returns
    -------
    slc_str : str
        String representing slice.
    """
    if len(sliceobjs) == 0:
        return ''
    if len(sliceobjs) > 1:
        return ', '.join(slice_str(s) for s in sliceobjs)
    sliceobj = sliceobjs[0]
    if isinstance(sliceobj, tuple):
        return slice_str(*sliceobj)
    if isinstance(sliceobj, int):
        return str(sliceobj)
    if sliceobj is Ellipsis:
        return '...'
    return (_ag.default_non_eval(sliceobj.start, str, '') + ':'
            + _ag.default_non_eval(sliceobj.stop, str, '')
            + _ag.default_non_eval(sliceobj.step, lambda x: f':{x}', ''))


# =============================================================================
# %%* Range conversion
# =============================================================================


def range_to_slice(the_range: range) -> slice:
    """Convert a slice object to a range.

    Parameters
    ----------
    the_range
        The `range` to convert, or any object that has integer/None attributes
        named `start`, `stop` and `step`, e.g. `slice`, `range`, `DisplayCount`

    Returns
    -------
    the_slice
        `slice` object with `start`, `stop` and `step` taken from `the_range`.
    """
    return slice(the_range.start, the_range.stop, the_range.step)


def slice_to_range(the_slice: slice, size: int = None) -> range:
    """Convert a slice object to a range.

    Parameters
    ----------
    the_slice
        The `slice` to convert. or any object that has integer attributes named
        `start`, `stop` and `step` e.g. `slice`, `range`, `DisplayCount`
    size : int or None
        Replaces upper bound if upper bound is `None` or `> size`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    the_range : range
        `range` object with `start`, `stop` and `step` taken from `the_slice`.

    Notes
    -----
    Negative `start`, `stop` interpreted relative to `size` if it is not `None`
    and relative to `0` otherwise.

    Raises
    ------
    ValueError
        If neither `size` nor upper bound are specified, where upper bound is
        `stop` if `step > 0` and `start` otherwise.
        Or if `step == 0`.
    """
    if isinstance(the_slice, int):
        return range(the_slice, the_slice + 1)
    if size is not None:
        return range(*the_slice.indices(size))
    if _unbounded(the_slice):
        raise ValueError('Must specify size or upper bound')
    step = _ag.default(the_slice.step, 1)
    if step < 0:
        start = the_slice.start
        stop = _ag.default(the_slice.stop, -1)
    else:
        start = _ag.default(the_slice.start, 0)
        stop = the_slice.stop
    return range(start, stop, step)


class SliceRange():
    """Class for converting a slice to a range.

    You can build a `range` for iteration by calling `srange[start:stop:step]`,
    where `srange` is an instance of `SliceRange`.

    Parameters
    ----------
    size : int or None
        Replaces slice upper bound if upper bound is `None` or `> size`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    the_range : range
        `range` object with `start`, `stop` and `step` taken from `the_slice`.

    Notes
    -----
    Negative `start`, `stop` interpreted relative to `size` if it is not `None`
    and relative to `0` otherwise.

    Raises
    ------
    ValueError
        If `size` is `None` and upper bound is not specified, where upper bound
        is `stop` if `step > 0` and `start` otherwise.
    """
    size: int

    def __init__(self, size: int = None):
        """
        Parameters
        ----------
        size : int or None
            Replaces slice upper bound if upper bound is `None` or `> size`.
            Upper bound is `stop` if `step > 0` and `start+1` otherwise.
        """
        self.size = size

    def __getitem__(self, arg) -> range:
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
        return slice_to_range(arg, self.size)


srange = SliceRange()
# =============================================================================
# %%* Utilities
# =============================================================================


def _unbounded(the_slice: slice) -> bool:
    """Could slice include infinity?"""
    if the_slice.step is None or the_slice.step > 0:
        return the_slice.stop is None
    if the_slice.step == 0:
        return False
    return the_slice.start is None


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


def _std_slice(the_slice: slice, size: int = None) -> slice:
    """Equivalent slice with default values where possible

    Also sets `stop = start + integer * step` without changing last value, if
    possible. Not possible when unbounded, unless `step == -1`.

    Parameters
    ----------
    the_slice : slice
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`
    size : int or None
        Replaces upper bound if upper bound is `None` or `> size`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    the_slice : slice
        Slice object with default values where possible.
    """
    if size is not None:
        the_slice = slice(*the_slice.indices(size))
    step = _ag.default(the_slice.step, 1)
    start, stop = the_slice.start, the_slice.stop
    if step > 0:
        start = _ag.default(start, 0)
    elif step < 0:
        stop = _ag.default(stop, -1)
    else:
        raise ValueError('slice step cannot be zero')
    if not _unbounded(the_slice):
        # if not _unbounded and step < 0, => start is not None
        # if not _unbounded and step > 0, => stop is not None
        stop = _stop_bound(start, stop, step)
    return slice(start, stop, step)


def _determinable(the_slice: slice, size: int = None) -> bool:
    """Is lowest value in slice determinable?

    Parameters
    ----------
    the_slice : slice
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`
    size : int or None
        Replaces upper bound if upper bound is `None` or `> size`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    determinable : bool
        True if lowest value in slice is determined.
    """
    the_slice = _std_slice(the_slice, size)
    return not (_unbounded(the_slice) and the_slice.step < -1)


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
        raise ValueError('Must specify size or start if step < -1')


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
    """Upper bound on smallest value slice

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


def _rectify(the_slice: slice, size: int = None) -> slice:
    """Equivalent slice with positive step

    Raises
    ------
    ValueError
        If `size, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    the_slice = _std_slice(the_slice, size)
    if the_slice.step > 0:
        return the_slice
    _raise_non_determinable(the_slice)
    return slice(_slice_min(the_slice), _slice_sup(the_slice), -the_slice.step)

# =============================================================================
# %%* Slice properties
# =============================================================================


def last_value(obj, size: int = None) -> int:
    """Last value in range

    Parameters
    ----------
    obj
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`.
    size : int or None
        Replaces upper bound if upper bound is `None` or `> size`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    val : int or None
        Last entry in iterable or slice, taking `start` and `step` into account
        or `None` if `step > 0` and `size, stop` are `None`.

    Raises
    ------
    ValueError
        If `size, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    obj = _std_slice(obj, size)
    if obj.step > 0:
        return _slice_max(obj)
    _raise_non_determinable(obj)
    return _slice_min(obj)


def stop_step(obj, size: int = None) -> int:
    """One step beyond last value in range

    Parameters
    ----------
    obj
        An object that has integer attributes named `start`, `stop` and `step`
        e.g. `slice`, `range`, `DisplayCount`.
    size : int or None
        Replaces upper bound if upper bound is `None` or `> size`.
        Upper bound is `stop` if `step > 0` and `start+1` otherwise.

    Returns
    -------
    stop : int or None
        Sets `stop = start + integer * step` without changing last value,
        or `None` if `step > 0` and `size, stop` are `None`.

    Raises
    ------
    ValueError
        If `size, start` are `None` and `step < -1`. Or if `step == 0`.
    """
    obj = _std_slice(obj, size)
    if obj.step > 0:
        return _slice_sup(obj)
    _raise_non_determinable(obj)
    return _slice_inf(obj)


def in_slice(val: int, the_slice: slice):
    """Does slice contain value?
    """
    if val is None:
        return _unbounded(the_slice)
    if not _unbounded(the_slice):
        return val in slice_to_range(the_slice)
    start = _rectify(the_slice).start
    return (val >= start) and ((val - start) % the_slice.step == 0)


def is_subslice(subslice: slice, the_slice: slice):
    """Does slice contain subslice?
    """
    subslice, the_slice = _rectify(subslice), _rectify(the_slice)
    return all(subslice.step % the_slice.step == 0,
               in_slice(subslice.start, the_slice),
               in_slice(subslice.stop, the_slice))
