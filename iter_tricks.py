# -*- coding: utf-8 -*-
# =============================================================================
# Created on Wed Aug  3 14:45:37 2016
#
# @author: Subhy
#
# Module: iter_tricks
# =============================================================================
"""
Iterators for convenience and displaying progress.

iterator : class
    Iterator for displaying loop counters.
DisplayBatch : class
    Iterator for displaying loop counters, returning slices.
DisplayEnumerate : class
    Like `zenumerate`, but using a `DisplayCount`.
DisplayZip : class
    Like `denumerate`, but doesn't return counter value.

Prints loop counter (plus 1), updates in place, and deletes at end.
Nested loops display on one line and update correctly if the inner
DisplayCount ends before the outer one is updated.
Displays look like:
    ' i: 3/5, j: 6/8, k:  7/10,'

.. warning:: Displays improperly in some clients.
See warning in `display_tricks` module.

For convenience:

dcount : function
    Creates a `iterator`.
denumerate : function
    Creates a `DisplayEnumerate`.
dzip: function
    Creates a `DisplayZip`.
dbatch
    Like `batch`, but uses `DisplayCount` to display counter.
The above have methods called rev that return reversed iterators.

zenumerate : function
    Combination of enumerate and unpacked zip.
batch
    Generator that yields `slice` objects covering batches of indices.
slice_to_range
    Convert a `slice` object to a `range`.
SliceRange
    Class whose instances can be subscripted to create `range`s.
srange
    Instance of `SliceRange`.
rdcount, rdbatch, rdenumerate, rdzip
    Aliases of `dcount.rev`, `dbatch.rev`, `denumerate.rev`, `dzip.rev`.
undcount, undbatch, undenumerate, undzip
    Wrappers around `range`, `batch`, `zenumerate`, `zip` that remove the
    `name` argument, to make it easier to switch between displaying/hiding.
You can set `start` and `stop` in `zenumerate`, `denumerate`, `dzip`, etc,
but only via keyword arguments.

Examples
--------
>>> import time
>>> for i in dcount('i', 5):
>>>     for j in dcount('j', 6):
>>>         for k in dcount('k', 4, 10):
>>>             time.sleep(0.1)
>>> print('done')

>>> for i in dcount('i', 5):
>>>     for j, k in zip(dcount('j', 8), [1, 7, 13]):
>>>         time.sleep(0.5)
>>>     time.sleep(1)
>>> print('done')

>>> words = [''] * 4
>>> letters = 'xyz'
>>> counts = [1, 7, 13]
>>> for idx, key, num in zenumerate(letters, counts):
>>>     words[idx] = key * num
>>>     time.sleep(0.1)
>>> print(words)

>>> words = [''] * 4
>>> letters = 'xyz'
>>> counts = [1, 7, 13]
>>> for idx, key, num in denumerate('idx', letters, counts):
>>>     words[idx] = key * num
>>>     time.sleep(0.1)
>>> print(words)

>>> keys = 'xyz'
>>> values = [1, 7, 13]
>>> assoc = {}
>>> for key, val in dzip('idx', keys, values):
>>>     assoc[key] = val
>>>     time.sleep(0.1)
>>> print(assoc)

>>> import numpy as np
>>> x = np.random.rand(1000, 3, 3)
>>> y = np.empty((1000, 3), dtype = complex)
>>> for s in batch(0, len(x), 10):
>>>     y[s] = np.linalg.eigvals(x[s])
>>> print(x[15], y[15])

>>> x = np.random.rand(1000, 3, 3)
>>> y = np.empty((1000, 3), dtype = complex)
>>> for s in dbatch('s', 0, len(x), 10):
>>>     y[s] = np.linalg.eigvals(x[s])
>>> print(x[15], y[15])
"""
# current_module = __import__(__name__)
__all__ = [
    'DisplayCount', 'DisplayBatch', 'DisplayEnumerate', 'DisplayZip',
    'zenumerate', 'batch', 'slice_to_range', 'SliceRange', 'srange',
    'dbatch', 'dcount', 'denumerate', 'dzip',
    'rdcount', 'rdbatch', 'rdenumerate', 'rdzip',
    'undcount', 'undbatch', 'undenumerate', 'undzip',
    'delay_warnings',
    ]
import itertools
# All of these imports could be removed:
from collections.abc import Iterator, Sized
from typing import Optional
import sys

from . import _iter_base as _it
from . import arg_tricks as _ag
from .display_tricks import delay_warnings

assert delay_warnings
assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# %%* Convenience functions
# =============================================================================


def zenumerate(*iterables: _it.Iterable, start=0, step=1) -> zip:
    """Combination of enumerate and unpacked zip.

    Behaves like `enumerate`, but accepts multiple iterables.
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)
    `start` and `step` can only be passed as keyword arguments.

    Example
    -------
    >>> words = [''] * 6
    >>> letters = 'xyz'
    >>> counts = [1, 7, 13]
    >>> for idx, key, num in zenumerate(letters, counts, start=1, step=2):
    >>>     words[idx] = key * num
    >>>     time.sleep(0.1)
    >>> print(words)
    """
    return zip(itertools.count(start, step), *iterables)


def batch(*sliceargs: _it.SliceArg, **kwargs: _it.SliceArg):
    """Iterate over batches

    Similar to `range` object, except at each iteration it yields a `slice`
    covering that step.

    Parameters
    ----------
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
        slice object that starts at current counter and stops at the next value
        with step size 1.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s in batch(0, len(x), 10):
    >>>     y[s] = np.linalg.eigvals(x[s])
    """
    start, stop, step = _it.extract_slice(sliceargs, kwargs)
    if stop is None:
        for i in itertools.count(start, step):
            yield slice(i, i+step, 1)
    else:
        for i in range(start, stop, step):
            yield slice(i, i+step, 1)


def slice_to_range(the_slice: slice, size: int = 0):
    """Convert a slice object to a range.

    Parameters
    ----------
    the_slice
        The `slice` to convert.
    size
        Upper limit of `range`s, used if `the_slice.stop` is `None`.

    Returns
    -------
    the_range
        `range` object with `start`, `stop` and `step` taken from `the_slice`.
    """
    if isinstance(the_slice, int):
        return the_slice
    return range(_ag.default(the_slice.start, 0),
                 _ag.default(the_slice.stop, size),
                 _ag.default(the_slice.step, 1))


class SliceRange():
    """Class for converting a slice to a range.

    You can build a `range` for iteration by calling `srange[start:stop:step]`,
    where `srange` is an instance of `SliceRange`.

    Parameters
    ----------
    size
        Upper limit of `range`s, used if `the_slice.stop` is `None`.
    """
    size: int

    def __init__(self, size: int = 0):
        """
        Parameters
        ----------
        size
            Upper limit of `range`s, used if `the_slice.stop` is `None`.
        """
        self.size = size

    def __getitem__(self, arg):
        """
        Parameters
        ----------
        the_slice
            The `slice` to convert.

        Returns
        -------
        the_range
            `range` object with `start`, `stop` and `step` from `the_slice`.
        """
        return slice_to_range(arg, self.size)


srange = SliceRange()


# =============================================================================
# %%* Displaying iterator classes
# =============================================================================


class DisplayCount(_it.DisplayMixin, Sized):
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
    iterator(name: str, low: int=0, high: int, step: int=1)

    DisplayCount(name: str, low: int=0, high: int)

    DisplayCount(name: str, high: int)

    DisplayCount(low: int=0, high: int, step: int=1)

    DisplayCount(low: int=0, high: int)

    DisplayCount(high: int)

    DisplayCount(name: str)

    DisplayCount()

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

    Using `zip` and omitting `high`::

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
        if self.stop is None:
            raise ValueError('Must specify stop to reverse')
        self.start, self.stop = self.stop - self.step, self.start - self.step
        self.step *= -1
        self._state.prefix += '-'
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        self.counter += self.step
        if (self.stop is None) or self.step*(self.stop - self.counter) > 0:
            self.update()
            return self.counter
        self.end()
        raise StopIteration()

    def __len__(self):
        """Number of entries"""
        if self.stop is None:
            raise ValueError('Must specify stop to define len')
        return (self.stop - self.start) // self.step

    def __contains__(self, val):
        """Is val a valid counter value?"""
        return (val - self.start) % self.step == 0

    def _check_ctr(self) -> str:
        """Ensure that DisplayCount's are properly used"""
        # raise error if ctr is outside range
        if self.counter > self.stop or self.counter < self.start:
            msg = f'{self._state.name}: has value {self.counter} '
            msg += f'when range is ({self.start}:{self.stop}:{self.step}).'
        return msg

    def update(self, msg: str = ''):
        """Erase previous counter and display new one."""
        if self.count_steps() % self.disp_step == 0:
            super().update(msg)
        if self.debug:
            self._state.check(self._check_ctr())

    def count_steps(self, counter=None):
        """How many steps have been taken?
        """
        if counter is None:
            counter = self.counter
        return (counter - self.start) // self.step


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
    Reads maximum couter value from min length of Sized ``sequences``.
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
        self._iterator = zip(self.display, *self._iterables)
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
            self._iterator = zip(*self._iterables)
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
# %%* Function interface
# - only saves ink
# =============================================================================


@_it.and_reverse
def dcount(*args: _it.DSliceArg, **kwargs) -> DisplayCount:
    """Produces iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

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

    Returns
    -------
    disp_counter : iterator
        An iterator that displays & returns counter value.

    Examples
    --------
    Triple nested loops:

    >>> import time
    >>> from iter_tricks import dcount
    >>> for i in dcount('i', 5):
    >>>     for j in dcount('j', 6):
    >>>         for k in dcount('k', 7):
    >>>             time.sleep(0.1)
    >>> print('done')

    Using `zip` and omitting `high`::

    >>> for i in dcount('i', 5):
    >>>     for j, k in zip(dcount('j'), [1, 7, 13]):
    >>>         time.sleep(1)
    >>>     time.sleep(3)
    >>> print('done')

    Raises
    ------
    IndexError
        If `DisplayCount.debug` is `True` and an instance is called with a
        counter value that is out of range, or if instances are improperly
        nested, e.g. if an outer DisplayCount is used before an inner one
        has finished.

    See Also
    --------
    DisplayCount, denumerate, dzip,
    DisplayEnumerate, DisplayZip,
    itertools.count
    """
    return DisplayCount(*args, **kwargs)


@_it.and_reverse
def denumerate(*args: _it.DZipArg, **kwds) -> DisplayEnumerate:
    """Like `zenumerate`, but using a `DisplayCount`.

    Reads maximum couter value from min length of Sized `sequences`.
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

    Returns
    -------
    disp_enum : DisplayEnumerate
        An iterator that displays & returns counter value & `sequences` entries

    Examples
    --------
    >>> import time
    >>> import numpy as np
    >>> from iter_tricks import denumerate
    >>> words = np.array([' ' * 13] * 3)
    >>> letters = 'xyz'
    >>> counts = [1, 7, 13]
    >>> for idx, key, num in denumerate('idx', letters, counts):
    >>>     words[idx] = key * num
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    DisplayEnumerate, dzip
    DisplayZip, DisplayCount,
    enumerate, zip
    """
    return DisplayEnumerate(*args, **kwds)


@_it.and_reverse
def dzip(*args: _it.DZipArg, **kwds) -> DisplayZip:
    """Like `enumerate` + `zip`, but using a `DisplayCount`.

    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
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

    Returns
    -------
    disp_zip : DisplayZip
        An iterator that displays counter value & returns `sequences` entries

    Examples
    --------
    >>> import time
    >>> from iter_tricks import dzip
    >>> keys = 'xyz'
    >>> values = [1, 7, 13]
    >>> assoc = {}
    >>> for key, val in dzip('idx', keys, values):
    >>>     assoc[key] = val
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    DisplayZip, denumerate,
    DisplayEnumerate, DisplayCount
    zip, enumerate
    """
    return DisplayZip(*args, **kwds)


@_it.and_reverse
def dbatch(*args: _it.DSliceArg, **kwargs) -> DisplayBatch:
    """Iterate over batches, with counter display

    Similar to `dcount`, except at each iteration it yields a
    `slice` covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 3-4/8k:  6-10/10,'

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


    Returns
    -------
    disp_counter : DisplayBatch
        An iterator that displays counter & returns:

        batch_slice:
            `slice` object that starts at current counter and stops at next.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s in dbatch('s', 0, len(x), 10):
    >>>     y[s] = np.linalg.eigvals(x[s])
    """
    return DisplayBatch(*args, **kwargs)


# =============================================================================
# %%* Reversed iterator factories
# =============================================================================
rdcount = dcount.rev
rdbatch = dbatch.rev
rdenumerate = denumerate.rev
rdzip = dzip.rev

# ==============================================================================
# %%* Non-displaying iterator functions
# ==============================================================================
undcount = _it.without_disp(range)
undbatch = _it.without_disp(batch)
undenumerate = _it.without_disp(zenumerate)
undzip = _it.without_disp(zip)
