"""Displaying iterator classes
"""
import itertools
import sys
from typing import ContextManager, Iterable, Iterator, Optional

from . import _iter_base as _it
from .range_tricks import RangeCollectionMixin as _RangeCollectionMixin
from .slice_tricks import ContainerMixin as _ContainerMixin


def _raise_if_no_stop(obj):
    """raise ValueError if obj.stop is None"""
    if obj.stop is None:
        raise ValueError("Need a value for stop")


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
        # offset for display of counter, default: 1 if start==0, 0 otherwise
        self.offset = kwargs.pop('offset', int(self.start == 0))
        self.disp_step = kwargs.pop('disp_step', 1)

        super().__init__(**kwargs)

        if name:
            self._state.prefix += name + ':'
        if self.step < 0:
            self._state.prefix += '-'

        if self.stop:
            self.stop = self.start + self.step * len(self)
            self.formatter = _it.counter_format(self.stop)
        self.formatter += ','

    def __iter__(self):
        """Display initial counter with prefix."""
        self.counter = self.start
        self.begin()
        self.counter -= self.step
        return self

    def __reversed__(self):
        """Prepare to display final counter with prefix.
        Calling iter and then next will count down.
        """
        _raise_if_no_stop(self)
        self.start, self.stop = self.stop - self.step, self.start - self.step
        self.step *= -1
        if self.step < 0:
            self._state.prefix += '-'
        else:
            self._state.prefix.rstrip('-')
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        self.counter += self.step
        if self.counter in self:
            self.update()
            return self.counter
        self.end()
        raise StopIteration()

    def _check_ctr(self) -> str:
        """Ensure that DisplayCount's are properly used"""
        # raise error if ctr is outside range
        if self.counter not in self:
            msg = f'{self._state.name}: has value {self.counter} '
            msg += f'when range is ({self.start}:{self.stop}:{self.step}).'
        return msg

    def update(self, msg: str = ''):
        """Erase previous counter and display new one."""
        if self.count_steps() % self.disp_step == 0:
            super().update(msg)
        if self.debug:
            self._state.check(self._check_ctr())

    def count_steps(self) -> int:
        """How many steps have been taken?
        """
        return self.index(self.counter)


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
        value with step size 1.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s in dbatch('s', 0, len(x), 10):
    >>>     y[s] = np.linalg.eigvals(x[s])
    """

    def __init__(self, *args: _it.DSliceArg, **kwargs):
        super().__init__(*args, **kwargs)

        if self.stop is None:
            self.formatter = '{:d}-{:d}'
        else:
            frmt = _it.counter_format(self.stop)
            self.formatter = frmt[:frmt.find('/')] + '-' + frmt
        self.formatter += ','

    def format(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
#        return self._frmt.format(ctr)
        return super().format(*itertools.chain(*((n, n + abs(self.step) - 1)
                                                 for n in ctrs)))

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        counter = super().__next__()
        return slice(counter, counter + abs(self.step))


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
        Passed to `iterator`

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
        self._iterator = iter(zip(self.display, *self._iterables))
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
        Passed to `iterator`

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
    DisplayCount, enumerate, zip
    """
    _iterator: Iterator

    def __iter__(self):
        """Display initial counter with prefix."""
        self.display = iter(self.display)
        if len(self._iterables) > 1:
            self._iterator = iter(zip(*self._iterables))
        else:
            self._iterator = iter(self._iterables[0])
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        try:
            next(self.display)
            output = next(self._iterator)
        except StopIteration:
            self.end()
            raise
        else:
            return output


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
