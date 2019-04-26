# -*- coding: utf-8 -*-
"""Custom containers
"""
from __future__ import annotations
import collections as cn
import itertools as _it
import operator as _op
import typing as _ty
import numbers as _num
A = _ty.TypeVar('A')


def default(optional: _ty.Optional[A], default_value: A):
    """Replace with default if None
    """
    return default_value if optional is None else optional


class Interval(cn.abc.Container):
    """An interval of the real line.

    For testing upper and lower bounds with ``in``.
    """
    start: _num.Real
    stop: _num.Real
    inclusive: _ty.Tuple[bool, bool]

    def __init__(self, start: _num.Real, stop: _num.Real,
                 inclusive: _ty.Union[bool, _ty.Tuple[bool, bool]] = True):
        if start > stop:
            raise ValueError(f"start={start} > stop={stop}")
        self.start = start
        self.stop = stop
        if isinstance(inclusive, bool):
            inclusive = (inclusive,) * 2
        self.inclusive = inclusive

    def __contains__(self, x):
        return ((self.start < x and x < self)
                or (self.inclusive[0] and x == self.start)
                or (self.inclusive[1] and x == self.stop))


# ------------------------------------------------------------------------------
# %% Shapes ans Tuples
# ------------------------------------------------------------------------------

def _eq_broadcast(siz0: int, siz1: int):
    """Would axes of this length be considered broadcastable?
    """
    return (siz0 == siz1) or (siz0 == 1) or (siz1 == 1)


def same_shape(shape0: _ty.Tuple[int, ...], shape1: _ty.Tuple[int, ...],
               compr: _ty.Callable[[int, int], bool] = _op.eq):
    """Are the two array shapes equivalent, ignoring leading singleton axes?
    """
    if isinstance(shape0, ShapeTuple) and isinstance(shape1, ShapeTuple):
        return same_shape(tuple(shape0), shape1, compr)
    diff = len(shape0) - len(shape1)
    pad0 = _it.chain(reversed(shape0), (-1,) * diff)
    pad1 = _it.chain(reversed(shape1), (-1,) * -diff)
    return all(compr(x, y) for x, y in zip(pad0, pad1))


def broadcastable(shape0: _ty.Tuple[int, ...], shape1: _ty.Tuple[int, ...]):
    """Are the two array shapes broadcastable?
    """
    return same_shape(shape0, shape1, _eq_broadcast)


def identical_shape(shape0: _ty.Tuple[int, ...], shape1: _ty.Tuple[int, ...]):
    """Are the two array shapes eaxctly the same, considering all axes?
    """
    return (len(shape0) == len(shape1)) and same_shape(shape0, shape1)


class ShapeTuple(tuple):
    """Stores the shapes of array types that implement broadcasting.

    Gives 1 if you ask for elements before the start, either via negative
    indexing, negative slicing or the reversed iterator.
    The reversed iterator will never stop iteration by itself.
    """

    def __getitem__(self, ind: _ty.Union[slice, int]):
        if isinstance(ind, slice):
            out = super().__getitem__(ind)
            m, n = len(out), len(self)
            start, stop, step = (default(ind.start, 0),
                                 default(ind.stop, n),
                                 default(ind.step, 1))
            if step < 0:
                start, stop = default(ind.stop, 0), default(ind.start, n)
            if start > n or start > stop:
                return out
            if start > 0:
                start -= n
            if stop > 0:
                stop -= n
            if step < 0:
                start += (stop - start) % -step
            num = (min(stop, 0) - start) // step
            return (1,)*(num-m) + out + (1,)*(-num-m)
        try:
            return super().__getitem__(ind)
        except IndexError:
            if isinstance(ind, int) and ind < -len(self):
                return 1
            raise

    def __reversed__(self):
        return _it.chain(reversed(tuple(self)), _it.repeat(1))


# ------------------------------------------------------------------------------
# %% Dictionaries
# ------------------------------------------------------------------------------


def dictify(**kwds):
    """Build dict from key names
    """
    return kwds


def update_existing(to_update: dict,
                    update_from: _ty.Union[dict, _ty.Iterable]):
    """Update existing keys only
    """
    if isinstance(update_from, cn.abc.Mapping):
        for k in to_update.keys():
            to_update[k] = update_from.get(k, to_update[k])
    else:
        for k, v in update_from:
            if k in to_update.keys():
                to_update[k] = v


def pop_existing(to_update: dict, pop_from: dict):
    """Pop to update existing keys only
    """
    for k in to_update.keys():
        to_update[k] = pop_from.pop(k, to_update[k])


def invert_dict(to_invert: dict) -> dict:
    """Swap keys and values.

    Assumes values are distinct."""
    return {v: k for k, v in to_invert.items()}


class PairedDict(dict):
    """One direction of bidirectional mapping

    Stores a reference to its inverse mapping in pdict.`inverse`. If the
    inverse is provided in the constructor, no effort is made to ensure that
    they are inverses of each other. Instead, the instances should be built
    using the classmethod `PairedDict.make_pairs`.

    Deleting an item also deletes the reversed item from `pdict.inverse`.
    Setting an item with ``pdict[key1] = key2``, deletes `key1` from `pdict` as
    above, deletes `key2` from `pdict.inverse`, adds item `(key1,key2)` to
    `pdict`, and adds item `(key2,key1)` to `pdict.inverse`.
    """
    inverse: PairedDict

    def __init__(self, *args, inverse=None, **kwds):
        super().__init__(*args, **kwds)
        self.inverse = inverse

    def __delitem__(self, key):
        """Delete inverse map as well as forward map"""
        super(PairedDict, self.inverse).__delitem__(self[key])
        super().__delitem__(key)

    def __setitem__(self, key, value):
        """Delete inverse & forward maps, then create new foward & inverse map
        """
        if key in self.keys():
            del self[key]
        if value in self.inverse.keys():
            del self.inverse[value]
        super().__setitem__(key, value)
        super(PairedDict, self.inverse).__setitem__(value, key)

    @classmethod
    def make_pairs(cls, *args, **kwds):
        """Create a pair of dicts that are inverses of each other"""
        fwd = cls(*args, **kwds)
        bwd = cls(invert_dict(fwd), inverse=fwd)
        fwd.inverse = bwd
        if len(fwd) != len(bwd):
            raise ValueError("Repeated keys/values")
        return [fwd, bwd]


class AssociativeMap(cn.ChainMap):
    """Bidirectional mapping

    Similar to a ``dict``, except the statement ``amap.fwd[key1] == key2`` is
    equivalent to ``amap.bwd[key2] == key1``. Both of these statements imply
    that ``amap[key1] == key2`` and ``amap[key2] == key1``. Both keys must be
    unique and hashable.

    An unordered associative map arises when subscripting the object itself.
    An ordered associative map arises when subscripting the ``fwd`` and ``bwd``
    properties, which are both Mappings and inverses of each other.

    If an association is modified, in either direction, both mappings that the
    association comprises are deleted and a new association is created. (see
    documentation for `PairedDict`). Setting ``amap[key1] = key2`` is always
    applied to ``amap.fwd``, with ``amap.bwd`` modified appropriately.

    See Also
    --------
    dict
    collections.ChainMap
    PairedDict
    """
    maps: _ty.List[PairedDict]

    def __init__(self, *args, **kwds):
        self.maps = PairedDict.make_pairs(*args, **kwds)

    def __delitem__(self, key):
        if key in self.fwd.keys():
            del self.fwd[key]
        elif key in self.bwd.keys():
            del self.bwd[key]
        else:
            raise KeyError(f"Key ''{key}' not found in either direction.")

    @property
    def fwd(self) -> PairedDict:
        return self.maps[0]

    @property
    def bwd(self) -> PairedDict:
        return self.maps[1]
