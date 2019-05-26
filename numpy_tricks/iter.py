# -*- coding: utf-8 -*-
"""
Iterating over multidimensional ndarrays
"""
from typing import Optional, Tuple
import numpy as np
from .. import _iter_base as _it

# =============================================================================
# %%* Classes
# =============================================================================


class DisplayNDIndex(_it.DisplayMixin):
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

    def __init__(self, *args: _it.DSliceArgs, **kwds: _it.KeyWords):
        name, shape = _it.extract_name(args, kwds)
        if isinstance(shape[0], tuple):
            shape = shape[0]
        self.offset = 1
        super().__init__(**kwds)
        if name:
            self._state.prefix += name + ':'
        self.shape = shape
        self.ndim = len(shape)
        self._it = np.ndindex(*shape)
        frmt = ', '.join([_it.counter_format(n) for n in shape])
        self._state.formatter = '(' + frmt + '),'

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

    def _str(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
        return super()._str(*ctrs[0])


class _DispNDIterBase(_it.AddDisplayToIterables, displayer=DisplayNDIndex):
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

    def __init_subclass__(cls):
        pass

    def __init__(self, *args: _it.Args, **kwds: _it.KeyWords):
        super().__init__(*args, **kwds)

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

    def __init__(self, *args: _it.Args, **kwds: _it.KeyWords):
        name, args = _it.extract_name(args, kwds)
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

    def __init__(self, *args: _it.Args, **kwds: _it.KeyWords):
        name, args = _it.extract_name(args, kwds)
        my_iter = np.ndenumerate(*args, **kwds)
        super().__init__(name, my_iter)


# =============================================================================
# %%* Functions
# =============================================================================


def dndindex(*args: _it.DSliceArgs, **kwds: _it.KeyWords) -> DisplayNDIndex:
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


def dnditer(*args: _it.Args, **kwds: _it.KeyWords) -> DisplayNDIter:
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


def dndenumerate(*args: _it.Args, **kwds: _it.KeyWords) -> DisplayNDEnumerate:
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
