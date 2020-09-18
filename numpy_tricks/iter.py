# -*- coding: utf-8 -*-
"""
Iterating over multidimensional ndarrays
"""
from typing import Optional, Tuple
import numpy as np
import sl_py_tools._iter_base as _ib

# =============================================================================
# Classes
# =============================================================================


class DisplayNDIndex(_ib.DisplayMixin):
    """Numpy ndindex with progress display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    DisplayNDIndex(name:str, d0: int, d1:int, ...)
    DisplayNDIndex(d0: int, d1: int, ...)

    Parameters
    ----------
    name : str, optional
        Name to display before indices.
    d0, d1, ... : int
        Shape of the array(s) iterated over.

    Yields
    ------
    multi-index : Tuple[int]
        A tuple of integers describing the indices along each axis.

    See Also
    --------
    numpy.ndindex
    """
    counter: Optional[Tuple[int, ...]]
    shape: Tuple[int]
    ndim: int
    _it: np.ndindex

    def __init__(self, *args: _ib.DSliceArgs, **kwds: _ib.DSliceKeys):
        name, shape = _ib.extract_name(args, kwds)
        if isinstance(shape[0], tuple):
            shape = shape[0]
        kwds.setdefault('offset', 1)
        if name:
            kwds.setdefault('prefix', name + ':')
        self.shape = shape
        self.ndim = len(shape)
        self._it = np.ndindex(*shape)
        frmt = ' '.join([_ib.counter_format(n) for n in shape])
        kwds.setdefault('formatter', '(' + frmt.rstrip(',') + '),')
        super().__init__(**kwds)

    def __iter__(self):
        self._it = iter(self._it)
        self.counter = (0,) * self.ndim
        self.begin()
        return self

    def __next__(self):
        try:
            self.counter = next(self._it)
            self.update()
        except StopIteration:
            self.end()
            raise
        else:
            return self.counter


class _DispNDIterBase(_ib.AddDisplayToIterables, displayer=DisplayNDIndex):
    """Base class for numpy nditer with display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    _DispNDIterBase(...)

    Parameters
    ----------
    ...
        Passed to ``_iter_base.AddDisplayToIterables``.

    Yields
    ------
    Whatever the underlying ``nditer`` yields.

    See Also
    --------
    numpy.nditer
    """

    def __init_subclass__(cls, **kwds):  # pylint: disable=arguments-differ
        pass

    def __len__(self):
        return self._iterables[0].shape

    def __iter__(self):
        self.display = iter(self.display)
        self._iterables = (iter(self._iterables[0]),)
        return self

    def __next__(self):
        try:
            next(self.display)
            output = next(self._iterables[0])
        except StopIteration:
            self.end()
            raise
        else:
            return output

    def __enter__(self):
        self._iterables = (self._iterables[0].__enter__(),)
        return self

    def __exit__(self, *exc):
        return self._iterables[0].__exit__(*exc)


class DisplayNDIter(_DispNDIterBase):
    """Numpy nditer with display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    DisplayNDIter(name:str, ...)
    DisplayNDIter(...)

    Parameters
    ----------
    name : str, optional
        Name to display before indices.
    ...
        Passed to ``numpy.nditer``.

    Yields
    ------
    Whatever the underlying ``nditer`` yields.

    See Also
    --------
    numpy.nditer
    """

    def __init__(self, *args: _ib.DArgs, **kwds: _ib.DKeys):
        name, args = _ib.extract_name(args, kwds)
        my_iter = np.nditer(*args, **kwds)
        super().__init__(name, my_iter)


class DisplayNDEnumerate(_DispNDIterBase):
    """Numpy ndenumerate with display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    DisplayNDEnumerate(name:str, ...)
    DisplayNDEnumerate(...)

    Parameters
    ----------
    name : str, optional
        Name to display before indices.
    ...
        Passed to ``numpy.ndenumerate``.

    Yields
    ------
    Whatever the underlying ``ndenumerate`` yields.

    See Also
    --------
    numpy.ndenumerate
    """

    def __init__(self, *args: _ib.DArgs, **kwds: _ib.DKeys):
        name, args = _ib.extract_name(args, kwds)
        my_iter = np.ndenumerate(*args, **kwds)
        super().__init__(name, my_iter)


# =============================================================================
# Functions
# =============================================================================


def dndindex(*args: _ib.DSliceArgs, **kwds: _ib.DSliceKeys) -> DisplayNDIndex:
    """Numpy ndindex with progress display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    DisplayNDIndex(name:str, d0: int, d1:int, ...)
    DisplayNDIndex(d0: int, d1: int, ...)

    Parameters
    ----------
    name : str, optional
        Name to display before indices.
    d0, d1, ... : int
        Shape of the array(s) iterated over.

    Yields
    ------
    multi-index : Tuple[int]
        A tuple of integers describing the indices along each axis.

    See Also
    --------
    numpy.ndindex
    """
    return DisplayNDIndex(*args, **kwds)


def dnditer(*args: _ib.DArgs, **kwds: _ib.DKeys) -> DisplayNDIter:
    """Numpy nditer with display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    DisplayNDIter(name:str, ...)
    DisplayNDIter(...)

    Parameters
    ----------
    name : str, optional
        Name to display before indices.
    ...
        Passed to ``numpy.nditer``.

    Yields
    ------
    Whatever the underlying ``nditer`` yields.

    See Also
    --------
    numpy.nditer
    """
    return DisplayNDIter(*args, **kwds)


def dndenumerate(*args: _ib.DArgs, **kwds: _ib.DKeys) -> DisplayNDEnumerate:
    """Numpy ndenumerate with display

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Nested loops display on one line and update correctly if the inner
    iterator ends before the outer one is updated.
    Displays look like:
        ' ii: (3/5, 6/8,  7/10),'

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Construction
    ------------
    DisplayNDEnumerate(name: str, ...)
    DisplayNDEnumerate(...)

    Parameters
    ----------
    name : str, optional
        Name to display before indices.
    ...
        Passed to ``numpy.ndenumerate``.

    Yields
    ------
    Whatever the underlying ``ndenumerate`` yields.

    See Also
    --------
    numpy.ndenumerate
    """
    return DisplayNDEnumerate(*args, **kwds)
