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

DisplayCount : class
    Iterator for displaying loop counters.
DisplayBatch : class
    Iterator for displaying loop counters, returning slices.
DisplayEnumerate : class
    Like `zenumerate`, but using a `DisplayCount`.
DisplayZip : class
    Like `denumerate`, but doesn't return counter value.

For convenience:

zenumerate : function
    Combination of enumerate and unpacked zip.
batch
    Generator that yields `slice` objects covering batches of indices.
dcount : function
    Creates a `DisplayCount`.
denumerate : function
    Creates a `DisplayEnumerate`.
dzip: function
    Creates a `DisplayZip`.
dbatch
    Like `batch`, but uses `DisplayCount` to display counter.
rdcount, rdbatch, rdenumerate, rdzip
    Reversed versions of `dcount`, `dbatch`, `rdenumerate`, `dzip`.
You can set `start` and `stop` in `zenumerate`, `denumerate`, `dzip`, etc,
but only via keyword arguments.

.. warning:: Doesn't display properly on ``qtconsole``, and hence ``Spyder``.
Instead, use in a console connected to the same kernel:
``cd`` to the folder, then type: ``jupyter console --existing``, and run your
code there.

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
__all__ = [
    'DisplayCount',
    'DisplayBatch',
    'DisplayEnumerate',
    'DisplayZip',
    'zenumerate',
    'batch',
    'dbatch',
    'dcount',
    'denumerate',
    'dzip',
    'rdcount',
    'rdbatch',
    'rdenumerate',
    'rdzip',
    ]
import itertools
from functools import wraps
# All of these imports could be removed:
from collections.abc import Iterator, Sized
from typing import Optional, Iterable, Tuple, Callable, Union, Any, Dict
import sys

from .display_tricks import DisplayTemporary, _DisplayState

assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# %%* Convenience functions
# =============================================================================


def zenumerate(*iterables: Tuple[Iterable, ...], start=0, step=1) -> zip:
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


def batch(*sliceargs, **kwargs):
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
    inds = slice(*sliceargs)

    start = kwargs.get('start', inds.start)
    stop = kwargs.get('stop', inds.stop)
    step = kwargs.get('step', inds.step)

    if start is None:
        start = 0
    if step is None:
        step = 1

    if stop is None:
        for i in itertools.count(start, step):
            yield slice(i, i+step, 1)
    else:
        for i in range(start, stop, step):
            yield slice(i, i+step, 1)


# =============================================================================
# %%* Main display counter class
# =============================================================================


def _extract_name(args: Tuple[Any],
                  kwds: Dict[str, Any]) -> (Optional[str], Tuple[Any]):
    """Extract name from other args

    If name is in kwds, assume all of args is others, pop name from kwds.
    Else, if args[0] is a str or None, assume it's name & args[1:] is others.
    Else, name is None and all of args is others.
    """
    name = None
    others = args
    if 'name' not in kwds and isinstance(args[0], (str, type(None))):
        name = args[0]
        others = args[1:]
    name = kwds.pop('name', name)
    return name, others


class _DisplayCntState(_DisplayState):
    """Internal stae of a DisplayCount, etc."""
    prefix: Union[str, DisplayTemporary]
    formatter: str

    def __init__(self, prev_state: Optional[_DisplayState] = None):
        """Construct internal state"""
        super().__init__(prev_state)
        self.prefix = ' '
        self.formatter = ''


class _DisplayMixin(DisplayTemporary):
    """Mixin providing non-iterator machinery for DisplayCount etc.

    Does not define `__iter__` or `__next__` (or `__reversed__`)
    This is a mixin. Only implements `begin`, `disp`, `end` and private stuff.
    Subclasses must implement `iter` and `next`.
    """
    counter: Optional[int]
    start: int
    stop: Optional[int]
    step: int
    offset: int
    _state: _DisplayCntState

    def __init__(self, **kwds):
        """Construct non-iterator machinery"""
        super().__init__(**kwds)
        self._state = _DisplayCntState(self._state)
        self.counter = None
        self.rename(type(self).__name__)

    def begin(self, msg: str = ''):
        """Display initial counter with prefix."""
        self._state.format(self._state.prefix)
        self._state.prefix = DisplayTemporary.show(self._state.prefix)
        self.counter = self.start - self.step
        super().begin(self._str(self.start) + msg)

    def update(self, msg: str = ''):
        """Erase previous counter and display new one."""
        dsp = self._str(self.counter)
        super().update(dsp + msg)

    def end(self):
        """Erase previous counter and prefix."""
        super().end()
        self._state.prefix.end()

    def _str(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
#        return self._frmt.format(ctr)
        return self._state.formatter.format(*(n + self.offset for n in ctrs))

    def _check(self):
        """Ensure that DisplayCount's are properly used"""
        super()._check()
        # raise error if ctr is outside range
        if self.counter > self.stop or self.counter < self.start:
            msg1 = 'DisplayCount{}'.format(self._prefix)
            msg2 = 'has value {} '.format(self.counter)
            msg3 = 'when upper limit is {}.'.format(self.stop)
            raise IndexError(msg1 + msg2 + msg3)


class DisplayCount(_DisplayMixin, Iterator, Sized):
    """Iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Construction
    ------------
    DisplayCount(name: str, low: int=0, high: int, step: int=1)

    DisplayCount(name: str, low: int=0, high: int)

    DisplayCount(low: int=0, high: int, step: int=1)

    DisplayCount(name: str, high: int)

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
    def __init__(self, *args: Tuple[Union[str, int, None], ...],
                 **kwargs):
        super().__init__(**kwargs)

        name, sliceargs = _extract_name(args, kwargs)
        inds = slice(*sliceargs)
        if name:
            self._state.prefix += name + ':'

        self.start = kwargs.get('start', inds.start)
        self.stop = kwargs.get('stop', inds.stop)
        self.step = kwargs.get('step', inds.step)

        if self.start is None:
            self.start = 0
        if self.step is None:
            self.step = 1
        if self.stop is not None:
            self.stop = self.start + self.step * len(self)

        # offset for display of counter, default: 1 if start==0, 0 otherwise
        self.offset = kwargs.get('offset', int(self.start == 0))

        if self.stop is None:
            self._state.formatter = '{:d}'
        else:
            num_dig = len(str(self.stop))
            self._state.formatter = '{:>' + str(num_dig) + 'd}/'
            self._state.formatter += self._str(self.stop - self.offset)[:-1]
        self._state.formatter += ','

    def __iter__(self):
        """Display initial counter with prefix."""
        self.begin()
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

#    def __contains__(self, x):
#        """Is it an entry?"""
#        if not isinstance(x, int):
#            return False
#        out = ((x - self.start)*self.step >= 0
#               and ((x - self.stop) % self.step) == 0)
#        if self.stop is not None:
#            out &= (self.stop - x)*self.step > 0
#        return out


# =============================================================================
# %%* Display wrappers for enumerate/zip/batch
# =============================================================================


def min_len(sequences: Tuple[Iterable, ...]) -> Optional[int]:
    """Length of shortest sequence.
    """
    mlen = min((len(seq) for seq in
                filter(lambda x: isinstance(x, Sized), sequences)),
               default=None)
    return mlen


class DisplayBatch(DisplayCount):
    """Iterate over batches, with counter display

    Similar to `DisplayCount` object, except at each iteration it yields a
    `slice` covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8(/2), k:  7/10(/5),'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

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
    def __init__(self, *args: Tuple[Union[str, int, None], ...],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._state.name = 'DisplayBatch({})'

        if self.stop is None:
            self._state.formatter = '{:d}-{:d}'
        else:
            num_dig = len(str(self.stop))
            frmt = '{:>' + str(num_dig) + 'd}'
            self._state.formatter = frmt + '-' + frmt + '/'
            self._state.formatter += frmt.format(self.stop)
        self._state.formatter += ','

    def _str(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
#        return self._frmt.format(ctr)
        return super()._str(*itertools.chain(*((n, n + abs(self.step) - 1)
                                               for n in ctrs)))

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        counter = super().__next__()
        return slice(counter, counter + abs(self.step))


class _AddDisplayToIterables():
    """Wraps iterator to display progress.

    Does not define `__iter__` or `__next__`
    This is a mixin. Only implements constructor, `begin`, `disp` and `end`,
    as well as __init__ and __reversed__.
    Subclasses must implement `iter` and `next`.
    """
    _iterables: Tuple[Iterable, ...]
    display: DisplayCount

    def __init__(self, *args: Tuple[Union[str, Iterable, None], ...],
                 **kwds):
        """Construct non-iterator machinery"""
        name, self._iterables = _extract_name(args, kwds)
        self.display = DisplayCount(name, min_len(self._iterables), **kwds)
        self.display.rename(type(self).__name__)

    def __reversed__(self):
        """Prepare to display fina; counter with prefix."""
        self.display = reversed(self.display)
        self._iterables = tuple(reversed(seq) for seq in self._iterables)
        return self

    def begin(self, *args, **kwds):
        """Display initial counter with prefix."""
        self.display.begin(*args, **kwds)

    def update(self, *args, **kwds):
        """Erase previous counter and display new one."""
        self.display.update(*args, **kwds)

    def end(self):
        """Erase previous counter and prefix."""
        self.display.end()


class DisplayEnumerate(_AddDisplayToIterables):
    """Wraps iterator to display progress.

    Like `zenumerate`, but using a `DisplayCount`.
    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

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
        self._iterator = zip(self.display, *self._iterables)
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        try:
            output = next(self._iterator)
        except StopIteration:
            self.display = None
            raise
        else:
            return output


class DisplayZip(_AddDisplayToIterables):
    """Wraps iterator to display progress.

    Like `denumerate`, but doesn't return counter value.
    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns sequence members in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

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
            self.display = None
            raise
        else:
            return output

# =============================================================================
# %%* Function interface
# - only saves ink
# =============================================================================


def dcount(*args: Tuple[Union[str, int, None], ...],
           **kwargs)-> DisplayCount:
    """Produces iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

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
    disp_counter : DisplayCount
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


def denumerate(*args: Tuple[Union[str, Iterable, None], ...],
               **kwds)-> DisplayEnumerate:
    """Like `zenumerate`, but using a `DisplayCount`.

    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

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


def dzip(*args: Tuple[Union[str, Iterable, None], ...],
         **kwds)-> DisplayZip:
    """Like `enumerate` + `zip`, but using a `DisplayCount`.

    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.

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


def dbatch(*args: Tuple[Union[str, int, None], ...],
           **kwargs) -> DisplayBatch:
    """Iterate over batches, with counter display

    Similar to `dcount`, except at each iteration it yields a
    `slice` covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 3-4/8k:  6-10/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

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


def _reverse_iter(it_func: Callable):
    """Wrap iterator factory with reversed
    """
    @wraps(it_func)
    def rev_it_func(*args, **kwds):
        return reversed(it_func(*args, **kwds))
    return rev_it_func


rdcount = _reverse_iter(dcount)
rdbatch = _reverse_iter(dbatch)
rdenumerate = _reverse_iter(denumerate)
rdzip = _reverse_iter(dzip)
