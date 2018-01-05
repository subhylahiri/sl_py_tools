# -*- coding: utf-8 -*-
# =============================================================================
# Created on Wed Aug  3 14:45:37 2016
#
# @author: Subhy
#
# Module: iter_tricks
# =============================================================================
"""
DisplayCount : class
    Iterator for displaying loop counters.
DisplayEnumerate : class
    Like `zenumerate`, but using a `DisplayCount`.
DisplayZip : class
    Like `denumerate`, but doesn't return counter value.

For convenience:

zenumerate : function
    Combination of enumerate and unpacked zip.
dcount : function
    Creates a `DisplayCount`.
denumerate : function
    Creates a `DisplayEnumerate`.
dzip: function
    Creates a `DisplayZip`.
You can set `start` and `stop` in `zenumerate`, `denumerate` and `dzip`,
but only via keyword arguments.

.. warning:: Doesn't display properly on ``qtconsole``, and hence ``Spyder``.

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
"""
__all__ = [
           'DisplayCount',
           'DisplayEnumerate',
           'DisplayZip',
           'zenumerate',
           'dcount',
           'denumerate',
           'dzip',
           ]
import itertools
# All of these imports could be removed:
from collections.abc import Iterator, Sized
from typing import Optional, ClassVar, Iterable, Tuple, Dict, Any
import sys
assert sys.version_info.major >= 3

# =============================================================================
# %%* Convenience function
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

# =============================================================================
# %%* Main display counter class
# =============================================================================


class _DisplayMixin(object):
    """Mixin providing machinery for DisplayCount etc.

    Does not define `__iter__` or `__next__` (or `__reversed__`)
    This is a mixin. Only implements `begin`, `disp`, `end` and private stuff.
    Subclasses must implement `iter` and `next`.
    """
    counter: Optional[int]
    start: int
    stop: Optional[int]
    step: int
    offset: int
    _state: Dict[str, Any]
#    _clean: bool
#    _prefix: str
#    _frmt: str
#   _nest_level: Optional[int]

    # set output to False to suppress counter display
    output: ClassVar[bool] = True
    # set debug to True to check that counter is in range and properly nested
    debug: ClassVar[bool] = False
    _nactive: ClassVar[int] = 0

    def __init__(self):
        self._state = dict(clean=True, prefix=' ', frmt='', nestlevel=None)
        self.counter = None

    def __del__(self):
        """Clean up, if necessary"""
        if not self._state['clean']:
            self.end()

    def begin(self):
        """Display initial counter with prefix."""
        self.counter = self.start - self.step
        self._print(self._state['prefix'] + self._str(self.start))
        self._state['clean'] = False
        if self.debug:
            self._nactive += 1
            self._state['nest_level'] = self.nactive
            self._check()

    def disp(self):
        """Erase previous counter and display new one."""
        dsp = self._str(self.counter)

        if self.stop is None:
            num_digits = len(self._str(self.counter - self.step))
        else:
            num_digits = len(dsp)

#        self._print('\b' * num_digits)
        # hack for jupyter's problem with multiple backspaces
        for i in '\b' * num_digits:
            self._print(i)
        self._print(dsp)
        if self.debug:
            self._check()

    def end(self):
        """Erase previous counter and prefix."""
        ndig = len(self._state['prefix'] + self._str(self.counter - self.step))
#        self._print('\b \b' * ndig)
        # hack for jupyter's problem with multiple backspaces
        for i in '\b \b' * ndig:
            self._print(i)
        self._state['clean'] = True
        if self.debug:
            self._nactive -= 1

    def _str(self, ctr: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
#        return self._frmt.format(ctr)
        return self._state['frmt'].format(ctr + self.offset)

    def _print(self, text: str):
        """Print with customisations: same line and immediate output"""
        if self.output:
            print(text, end='', flush=True)

    def _check(self):
        """Ensure that DisplayCount's are properly used"""
        # raise error if ctr is outside range
        if self.counter > self.stop or self.counter < self.start:
            msg1 = 'DisplayCount{}'.format(self._prefix)
            msg2 = 'has value {} '.format(self.counter)
            msg3 = 'when upper limit is {}.'.format(self.stop)
            raise IndexError(msg1 + msg2 + msg3)
        # raise error if ctr_dsp's are nested incorrectly
        if self._state['nest_level'] != self._nactive:
            msg1 = 'DisplayCount{}'.format(self._prefix)
            msg2 = 'used at level {} '.format(self._nactive)
            msg3 = 'instead of level {}.'.format(self._state['nest_level'])
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
    output : bool
        Class attribute. Set it to `False` to suppress display.
        Default: `True`.
    debug : bool
        Class attribute. Set it to `True` to check counter range and nesting.
        Default: `False`
    counter : int or None
        Instance attribute. Current value of the counter.

    Methods
    -------
    None of the following methods are ordinarily needed:

    begin()
        to initialise counter and display.
    disp()
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
    def __init__(self, name: Optional[str] = None,
                 *sliceargs: Tuple[Optional[int], ...],
                 **kwargs):
        super().__init__()

        if name is None:
            inds = slice(*sliceargs)
        elif isinstance(name, str):
            inds = slice(*sliceargs)
            self._state['prefix'] += name + ': '
        else:
            inds = slice(name, *sliceargs)

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
            self._state['frmt'] = '{:d}'
        else:
            num_dig = len(str(self.stop))
            self._state['frmt'] = '{:>' + str(num_dig) + 'd}'
            self._state['frmt'] += '/' + self._state['frmt'].format(self.stop)
        self._state['frmt'] += ','

    def __iter__(self):
        """Display initial counter with prefix."""
        self.begin()
        return self

    def __reversed__(self):
        """Display final counter with prefix.
        Calling next will count down.
        """
        if self.stop is None:
            raise ValueError('Must specify stop to reverse')
        self.start, self.stop = self.stop - self.step, self.start - self.step
        self.step *= -1
        self.begin()
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        self.counter += self.step
        if (self.stop is None) or self.step*(self.stop - self.counter) > 0:
            self.disp()
            return self.counter
        else:
            self.end()
            raise StopIteration()

    def __len__(self):
        """Number of entries"""
        if self.stop is None:
            return None
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
# %%* Display wrappers for enumerate/zip
# =============================================================================


def _min_len(sequences: Tuple[Iterable, ...]) -> Optional[int]:
    """Minimum length of sequences
    """
    min_len = min((len(seq) for seq in
                   filter(lambda x: isinstance(x, Sized), sequences)),
                  default=None)
    return min_len


class _AddDisplayToIterables(object):
    """Wraps iterator to display progress.

    Does not define `__iter__` or `__next__` (or `__reversed__`)
    This is a mixin. Only implements constructor, `begin`, `disp` and `end`.
    Subclasses must implement `iter` and `next`.
    """
    _iterables: Tuple[Iterable, ...]
    display: DisplayCount

    def __init__(self, name: Optional[str] = None,
                 *sequences: Tuple[Iterable, ...],
                 **kwds):
        if name is None or isinstance(name, str):
            self._iterables = sequences
        else:
            self._iterables = (name,) + sequences
            name = None
        self.display = DisplayCount(name, _min_len(sequences), **kwds)

    def begin(self):
        """Display initial counter with prefix."""
        self.display.begin()

    def disp(self):
        """Erase previous counter and display new one."""
        self.display.disp()

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


def dcount(name: Optional[str] = None,
           *sliceargs: Tuple[Optional[int], ...],
           **kwargs)-> DisplayCount:
    """Produces iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.

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
    return DisplayCount(name, *sliceargs, **kwargs)


def denumerate(name: Optional[str] = None,
               *sequences: Tuple[Iterable, ...],
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
    return DisplayEnumerate(name, *sequences, **kwds)


def dzip(name: Optional[str] = None,
         *sequences: Tuple[Iterable, ...],
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
    return DisplayZip(name, *sequences, **kwds)
