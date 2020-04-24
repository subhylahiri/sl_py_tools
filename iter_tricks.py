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

Requires
--------
gmpy2
"""
# current_module = __import__(__name__)
__all__ = [
    'DisplayCount', 'DisplayBatch', 'DisplayEnumerate', 'DisplayZip',
    'zenumerate', 'batch',
    'range_to_slice', 'slice_to_range', 'SliceRange', 'srange',
    'dbatch', 'dcount', 'denumerate', 'dzip',
    'rdcount', 'rdbatch', 'rdenumerate', 'rdzip',
    'undcount', 'undbatch', 'undenumerate', 'undzip',
    'delay_warnings', 'erange', 'sr_',
    ]
import itertools
# All of these imports could be removed:
from typing import Sequence
import sys

from . import _iter_base as _it
from .arg_tricks import Export as _Export
from .display_tricks import delay_warnings
from .containers import ZipSequences, tuplify
from .range_tricks import erange, sr_
from .slice_tricks import range_to_slice, slice_to_range, SliceRange, srange
from .iter_class import DisplayCount, DisplayBatch, DisplayEnumerate, DisplayZip

_Export[delay_warnings, erange, sr_]
_Export[range_to_slice, slice_to_range, SliceRange, srange]
assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# Convenience functions
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
    for i in erange(start, stop, step):
        yield slice(i, i+step, 1)


def batched(step: int, *sequences: Sequence, usemax=True):
    """Iterate over chunks of sequence(s)

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
    >>> for xx, yy in batchenum(10, x, y):
    >>>     yy[...] = np.linalg.eigvals(xx)
    """
    seqs = ZipSequences(*sequences, usemax=usemax)
    for slc in batch(0, len(seqs), step):
        yield seqs[slc]


def batchenum(step: int, *sequences: Sequence, usemax=True):
    """Iterate over chunks of sequence(s)

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
    sequence1[s], sequence2[s], ...
        slice(s) of the sequence(s) that starts at current counter and stops at
        the next value with step size 1correspond to `batch_slice`.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for ss, xx in batchenum(10, x):
    >>>     y[ss] = np.linalg.eigvals(xx)
    """
    seqs = ZipSequences(*sequences, usemax=usemax)
    for slc in batch(0, len(seqs), step):
        yield (slc,) + tuplify(seqs[slc])


# =============================================================================
# Function interface
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

    Using `zip` and omitting `stop`::

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

    Reads maximum counter value from min length of Sized `sequences`.
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

    Reads maximum counter value from min length of Sized `sequences`.
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


def dbatched(name: str, step: int, *sequences: Sequence, usemax=True):
    """Iterate over chunks of sequence(s), with counter display

    Similar to `zip` object, except at each iteration it yields a slice of the
    sequences covering that step.

    Parameters
    ----------
    name : str
        name of counter used for prefix.
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
    >>> for xx, yy in batchenum('xy', 10, x, y):
    >>>     yy[...] = np.linalg.eigvals(xx)
    """
    seqs = ZipSequences(*sequences, usemax=usemax)
    for slc in dbatch(name, 0, len(seqs), step):
        yield seqs[slc]


def dbatchenum(name: str, step: int, *sequences: Sequence, usemax=True):
    """Iterate over chunks of sequence(s), with counter display

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
    name : str
        name of counter used for prefix.
    batch_slice
        slice object that starts at current counter and stops at the next value
        with step size 1.
    sequence1[s], sequence2[s], ...
        slice(s) of the sequence(s) that starts at current counter and stops at
        the next value with step size 1correspond to `batch_slice`.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for ss, xx in batchenum('ss', 10, x):
    >>>     y[ss] = np.linalg.eigvals(xx)
    """
    seqs = ZipSequences(*sequences, usemax=usemax)
    for slc in dbatch(name, 0, len(seqs), step):
        yield (slc,) + tuplify(seqs[slc])


# pylint: disable=invalid-name
# =============================================================================
# Reversed iterator factories
# =============================================================================
rdcount = dcount.rev
rdbatch = dbatch.rev
rdenumerate = denumerate.rev
rdzip = dzip.rev

# ==============================================================================
# Non-displaying iterator functions
# ==============================================================================
undcount = _it.without_disp(erange)
undbatch = _it.without_disp(batch)
undenumerate = _it.without_disp(zenumerate)
undzip = _it.without_disp(zip)

unrdcount = undcount.rev
unrdbatch = undbatch.rev
unrdenumerate = undenumerate.rev
unrdzip = undzip.rev

# =============================================================================
# Slice iterator factories
# =============================================================================
b_ = _it.SliceToIter(batch)
dc_ = _it.SliceToIter(dcount, 1)
db_ = _it.SliceToIter(dbatch, 1)
rdc_ = _it.SliceToIter(rdcount, 1)
rdb_ = _it.SliceToIter(rdbatch, 1)
udc_ = _it.SliceToIter(undcount, 1)
udb_ = _it.SliceToIter(undbatch, 1)
urdc_ = _it.SliceToIter(unrdcount, 1)
urdb_ = _it.SliceToIter(unrdbatch, 1)
