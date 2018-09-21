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
    """Numpy ndindex with display
    """
    counter: Optional[Tuple[int, ...]]
    shape: Tuple[int]
    ndim: int
    _it: np.ndindex

    def __init__(self, *args: _it.DSliceArgs, **kwds: _it.KeyWords):
        name, shape = _it.extract_name(args, kwds)
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


class DisplayNDIter(_it.AddDisplayToIterables, displayer=DisplayNDIndex):
    """Numpy nditer with display
    """
    def __init__(self, *args: _it.Args, **kwds: _it.KeyWords):
        name, args = _it.extract_name(args, kwds)
        my_iter = np.nditer(*args, **kwds)
        super().__init__(name, my_iter)

    def _get_len(self):
        return self._iterables[0].shape

    def __iter__(self):
        self.display = iter(self.display)
        self._iterables = iter(self._iterables)
        return self

    def __next__(self):
        try:
            next(self.display)
            output = next(self._iterables)
        except StopIteration:
            self.end()
            raise
        else:
            return output


class DisplayNDEnumerate(_it.AddDisplayToIterables, displayer=DisplayNDIndex):
    """Numpy nditer with display
    """
    def __init__(self, *args: _it.Args, **kwds: _it.KeyWords):
        name, args = _it.extract_name(args, kwds)
        my_iter = np.ndenumerate(*args, **kwds)
        super().__init__(name, my_iter)

    def _get_len(self):
        return self._iterables[0].shape

    def __iter__(self):
        self.display = iter(self.display)
        self._iterables = iter(self._iterables)
        return self

    def __next__(self):
        try:
            next(self.display)
            output = next(self._iterables)
        except StopIteration:
            self.end()
            raise
        else:
            return output


# =============================================================================
# %%* Functions
# =============================================================================


def dndindex(*args: _it.DSliceArgs, **kwds: _it.KeyWords):
    """
    """
    return DisplayNDIndex(*args, **kwds)


def dnditer(*args: _it.Args, **kwds: _it.KeyWords):
    """
    """
    return DisplayNDIter(*args, **kwds)


def dndenumerate(*args: _it.Args, **kwds: _it.KeyWords):
    """
    """
    return DisplayNDEnumerate(*args, **kwds)
