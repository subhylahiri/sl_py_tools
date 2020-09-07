"""Displaying iterator classes
"""
from __future__ import annotations
import re

import sys
from itertools import chain
from numbers import Number
from typing import (ContextManager, Iterable, Iterator, Optional, Sequence,
                    Tuple)

from . import _iter_base as _it
from .containers import ZipSequences, tuplify
from .range_tricks import RangeIsh as _RangeIsh
from .range_tricks import RangeCollectionMixin as _RangeCollectionMixin
from .slice_tricks import ContainerMixin as _ContainerMixin

# =============================================================================
_BATCH_FIND = re.compile(r'(\{:>?\d*d\})')
_BATCH_REP = r"\1-\1"


def _raise_if_no_stop(obj):
    """raise ValueError if obj.stop is None"""
    if obj.stop is None:
        raise ValueError("Need a value for stop")


def _batch_format(num: Optional[int]) -> str:
    """Format string for counters that run up to num

    Produces '2-4/10', etc.
    """
    return _BATCH_FIND.sub(_BATCH_REP, _it.counter_format(num))


def _max_val(obj: _RangeIsh, offset: int = 0) -> Optional[int]:
    """Maximum value of RangeIsh"""
    start, stop, step = _it.st_args(obj)
    if step > 0:
        return None if stop is None else stop - step + offset
    return None if start is None else start + offset


def _min_val(obj: _RangeIsh, offset: int = 0) -> Optional[int]:
    """Minimum value of RangeIsh"""
    start, stop, step = _it.st_args(obj)
    if step > 0:
        return None if start is None else start + offset
    return None if stop is None else stop - step + offset


# =============================================================================
# Displaying iterator classes
# =============================================================================


class DisplayCount(_it.DisplayMixin, _RangeCollectionMixin, _ContainerMixin):
    """Iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    iterator([name: str,] [start: int=0, stop: int[, step: int=1]])

    DisplayCount([name: str,] [stop: int])

    Parameters
    ----------
    name : str or None, optional
        name of counter used for prefix.
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.
    Or use keyword arguments.

    Attributes
    ----------
    output : bool, default : True
        Class attribute. Set it to `False` to suppress display.
    debug : bool, default : False
        Class attribute. Set it to `True` to check counter range and nesting.
    counter : int or None
        Instance attribute. Current value of the counter.

    Methods
    -------
    None of the following methods are ordinarily needed:

    begin()
        to initialise counter and display.
    update()
        to display current counter.
    end()
        to erase display after loops.

    Examples
    --------
    Triple nested loops:

    >>> import time
    >>> from iter_tricks import DisplayCount
    >>> for i in DisplayCount('i', 5):
    >>>     for j in DisplayCount('j', 6):
    >>>         for k in DisplayCount('k', 7):
    >>>             time.sleep(0.1)
    >>> print('done')

    Using `zip` and omitting `stop`::

    >>> for i in DisplayCount('i', 5):
    >>>     for j, k in zip(DisplayCount('j'), [1, 7, 13]):
    >>>         time.sleep(1)
    >>>     time.sleep(3)
    >>> print('done')

    Raises
    ------
    IndexError
        If `DisplayCount.debug` is `True` and an instance is called with a
        counter value that is out of range, or if instances are improperly
        nested, e.g. if an outer DisplayCount is updated before an inner one
        has finished.

    See Also
    --------
    denumerate, DisplayZip, itertools.count
    """
    start: int
    stop: Optional[int]
    step: int
    disp_step: int

    def __init__(self, *args: _it.DSliceArg, **kwargs):
        name, sliceargs = _it.extract_name(args, kwargs)
        self.start, self.stop, self.step = _it.extract_slice(sliceargs, kwargs)
        self.disp_step = kwargs.pop('disp_step', 1)
        # offset for display of counter, default: 1 if start==0, 0 otherwise
        kwargs.setdefault('offset', int(_min_val(self) == 0))

        prefix = ''
        if name:
            prefix += name + ':'
        if self.step < 0:
            prefix += '-'
        kwargs.setdefault('prefix', prefix)

        if None not in {self.stop, self.start}:
            self.stop = self.start + self.step * len(self)
        maxv = _max_val(self, kwargs.get('offset', 0))
        kwargs.setdefault('formatter', _it.counter_format(maxv))

        super().__init__(**kwargs)

    def __iter__(self) -> DisplayCount:
        """Display initial counter with prefix."""
        self.counter = self.start
        self.begin()
        self.counter -= self.step
        return self

    def __reversed__(self) -> DisplayCount:
        """Prepare to display final counter with prefix.
        Calling iter and then next will count down.
        """
        _raise_if_no_stop(self)
        args = self.stop - self.step, self.start - self.step, -self.step
        name = self.prefix.template.rstrip(':-')
        kwds = {'offset': self.offset, 'disp_step': self.disp_step,
                'formatter': self.formatter}
        return type(self)(name, *args, **kwds)

    def __next__(self) -> int:
        """Increment counter, erase previous counter and display new one."""
        self.counter += self.step
        if self.counter in self:
            self.update()
            return self.counter
        self.end()
        raise StopIteration()

    def _check(self, msg: str = '') -> str:
        """Ensure that DisplayCount's are properly used"""
        # raise error if ctr is outside range
        if self.counter not in self:
            msg += type(self).__name__
            msg += f'({self.prefix.template}): has value {self.counter} '
            msg += f'when range is [{self.start}:{self.stop}:{self.step}].'
        super()._check(msg)

    def update(self):  # , msg: str = ''):
        """Erase previous counter and display new one."""
        if self.count_steps() % self.disp_step == 0:
            super().update()

    def count_steps(self) -> int:
        """How many steps have been taken?
        """
        if isinstance(self.counter, Number):
            return self.index(self.counter)
        return self.index(self.counter[0])


class DisplayBatch(DisplayCount):
    """Iterate over batches, with counter display

    Similar to `DisplayCount` object, except at each iteration it yields a
    `slice` covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8(/2), k:  7/10(/5),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.
    Or use keyword arguments.

    Yields
    ------
    batch_slice
        `slice` object that starts at current counter and stops at the next
        value with step size `sgn(step)`.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s in dbatch('s', 0, len(x), 10):
    >>>     y[s] = np.linalg.eigvals(x[s])
    """
    # step of yielded slices
    _slice_step: int

    def __init__(self, *args: _it.DSliceArg, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter.template = _batch_format(self.stop)
        self._slice_step = self.step // abs(self.step)

    def __next__(self) -> slice:
        """Increment counter, erase previous counter and display new one."""
        counter = super().__next__()
        return slice(counter, counter + self.step, self._slice_step)

    def __reversed__(self) -> DisplayBatch:
        obj = super.__reversed__()
        obj.start += self.step - self._slice_step
        obj.stop += self.step - self._slice_step
        return obj

    def update(self):
        """Erase previous counter and display new one."""
        if self.count_steps() % self.disp_step == 0:
            ctrs = [(n, n + abs(self.step) - 1) for n in tuplify(self.counter)]
            ctrs, self.counter = self.counter, tuple(chain.from_iterable(ctrs))
            super().update()
            self.counter = ctrs


class DisplayEnumerate(_it.AddDisplayToIterables, displayer=DisplayCount):
    """Wraps iterator to display progress.

    Like ``zenumerate``, but using a ``DisplayCount``.
    Reads maximum counter value from min length of Sized ``sequences``.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    sequences : Tuple[Iterable]
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.
        Note: argument is unpacked.
    **kwds
        Passed to `DisplayCount`

    Examples
    --------
    >>> import time
    >>> from iter_tricks import DisplayEnumerate
    >>> keys = 'xyz'
    >>> values = [1, 7, 13]
    >>> assoc = {}
    >>> for key, val in DisplayEnumerate('idx', keys, values):
    >>>     assoc[key] = val
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    DisplayCount, enumerate, zip
    """
    _iterator: Iterator

    def __iter__(self):
        """Display initial counter with prefix."""
        self._iterator = iter(ZipSequences(self.display, *self._iterables,
                                           usemax=self._max))
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        try:
            output = next(self._iterator)
        except StopIteration:
            self.end()
            raise
        else:
            return output


class DisplayZip(_it.AddDisplayToIterables, displayer=DisplayCount):
    """Wraps iterator to display progress.

    Like ``denumerate``, but doesn't return counter value.
    Reads maximum couter value from min length of Sized ``sequences``.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns sequence members in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    sequences : Tuple[Iterable]
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.
        Note: argument is unpacked.
    **kwds
        Passed to `DisplayCount`

    Examples
    --------
    >>> import time
    >>> from iter_tricks import DisplayZip
    >>> keys = 'xyz'
    >>> values = [1, 7, 13]
    >>> assoc = {}
    >>> for key, val in DisplayZip('idx', keys, values):
    >>>     assoc[key] = val
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    DisplayCount, zip
    """
    _iterator: Iterator

    def __iter__(self):
        """Display initial counter with prefix."""
        self.display = iter(self.display)
        if len(self._iterables) > 1:
            self._iterator = iter(ZipSequences(*self._iterables,
                                               usemax=self._max))
        else:
            self._iterator = iter(self._iterables[0])
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        try:
            next(self.display)
            output = next(self._iterator)
        except StopIteration:
            self.end()  # not needed
            raise
        else:
            return output


class DisplayBatched(_it.AddDisplayToIterables, displayer=DisplayBatch):
    """Iterate over batches, with counter display

    Similar to `DisplayZip` object, except at each iteration it yields a
    slice of the zipped sequences over that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8(/2), k:  7/10(/5),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Parameters
    ----------
    name : str or None
        name of counter used for prefix.
    step : int
        Size of steps.
    sequences : Tuple[Iterable]
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.
        Note: argument is unpacked.
    usemax : bool, keyword only, default=False
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.
    **kwds
        Other keywords are passed to the ``DisplayBatch`` constructor.

    Yields
    ------
    seq_slice : Tuple[Sequence]
        `tuple` of slices of sequences that starts at current counter and
        stops at the next value with step size `sgn(step)`.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for xx, yy in DisplayBatched('s', 10, x, y):
    >>>     yy[...] = np.linalg.eigvals(xx)
    >>> print(x[15], y[15])

    See Also
    --------
    DisplayBatch, zip
    """
    _iterator: Iterator

    def __init__(self, name: Optional[str], step: int, *args: _it.SliceArgs,
                 **kwds) -> None:
        """Construct the displayer"""
        super().__init__(name, *args, step=step, **kwds)
        self._iterator = None

    def __iter__(self) -> DisplayBatched:
        """Display initial counter with prefix."""
        self.display = iter(self.display)
        if len(self._iterables) > 1:
            self._iterator = ZipSequences(*self._iterables, usemax=self._max)
        else:
            self._iterator = self._iterables[0]
        return self

    def __next__(self) -> Tuple[Sequence, ...]:
        """Increment counter, erase previous counter and display new one."""
        try:
            slc = next(self.display)
            output = self._iterator[slc]
        except StopIteration:
            self.end()
            raise
        else:
            return output

    def __reversed__(self) -> DisplayBatched:
        self.display = reversed(self.display)
        return self


class DisplayBatchEnum(DisplayBatched, displayer=DisplayBatch):
    """Iterate over batches, with counter display

    Similar to `DisplayBatch` object, except at each iteration it yields a
    `slice` object covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8(/2), k:  7/10(/5),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Parameters
    ----------
    name : str or None
        name of counter used for prefix.
    step : int
        Size of steps.
    sequences : Tuple[Iterable]
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.
        Note: argument is unpacked.
    usemax : bool, keyword only, default=False
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.
    **kwds
        Other keywords are passed to the ``DisplayBatch`` constructor.

    Yields
    ------
    batch_slice : slice
        `slice` object that starts at current counter and stops at the next
        value with step size `sgn(step)`.
    seq_slice : Tuple[Sequence]
        `tuple` of slices of sequences corresponding to `batch_slice`.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s, xx in DisplayBatchEnum('s', 10, x):
    >>>     y[s] = np.linalg.eigvals(xx)
    >>> print(x[15], y[15])

    See Also
    --------
    DisplayBatch, DisplayBatched, DisplayEnumerate, enumerate
    """
    _iterator: Iterator

    def __next__(self) -> Tuple[slice, Sequence]:
        """Increment counter, erase previous counter and display new one."""
        try:
            slc = next(self.display)
            output = self._iterator[slc]
        except StopIteration:
            self.end()
            raise
        else:
            return (slc,) + tuplify(output)


# =============================================================================
# Wrapper
# =============================================================================


class ContextualIterator:
    """Wrap an iterable in a context

    Parameters
    ----------
    manager : ContextManager
        A for loop over the `ContextualIterator` is enclosed in `with manager:`
    iterable : Iterable
        The iterable being wrapped, its ouputs are used in a for loop over the
        `ContextualIterator`
    """
    manager: ContextManager
    iterator: Iterable

    def __init__(self, manager: ContextManager, iterable: Iterable):
        self.manager = manager
        self.iterator = iterable

    def __iter__(self):
        self.manager.__enter__()
        self.iterator = iter(self.iterator)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.manager.__exit__(None, None, None)
            raise
        finally:
            if self.manager.__exit__(*sys.exc_info()):
                raise StopIteration
