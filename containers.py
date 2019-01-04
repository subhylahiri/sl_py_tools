# -*- coding: utf-8 -*-
"""Custom containers
"""
import collections as cn
import itertools as _it
import operator as _op
import typing as _ty
A = _ty.TypeVar('A')


def default(optional: _ty.Optional[A], default_value: A):
    """Replace with default if None"""
    return default_value if optional is None else optional


def _eq_broadcast(siz0: int, siz1: int):
    """Would axes of this length be considered broadcastable?"""
    return (siz0 == siz1) or (siz0 == 1) or (siz1 == 1)


def same_shape(shape0: _ty.Tuple[int, ...], shape1: _ty.Tuple[int, ...],
               compr: _ty.Callable[[int, int], bool] = _op.eq):
    """Are the two array shapes equivalent, ignoring leading singleton axes?
    """
    if isinstance(shape0, ShapeTuple) and isinstance(shape1, ShapeTuple):
        return same_shape(tuple(shape0), shape1, compr)
    return all(compr(x, y) for x, y in zip(reversed(shape0), reversed(shape1)))


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
            if ind.start is not None and ind.start < -len(self):
                num = default(ind.stop, len(self))
                if num >= 0:
                    num = min(num - len(self), 0)
                num -= ind.start
                num //= default(ind.step, 1)
                out = (1,) * (num - len(out)) + out
            return out
        try:
            return super().__getitem__(ind)
        except IndexError:
            if isinstance(ind, int) and ind < -len(self):
                return 1
            raise

    def __reversed__(self):
        return _it.chain(reversed(tuple(self)), _it.repeat(1))


def invert_dict(to_invert: dict) -> dict:
    """Swap keys and values.

    Assumes values are distinct."""
    return {v: k for k, v in to_invert.items()}


class _PairedDict(dict):
    """One direction of bidirectional mapping"""

    def __init__(self, *args, inverse=None, **kwds):
        super().__init__(*args, **kwds)
        self._inverse = inverse

    def __delitem__(self, key):
        """Delete inverse map as well as forward map"""
        super(_PairedDict, self._inverse).__delitem__(self[key])
        super().__delitem__(key)

    def __setitem__(self, key, value):
        """Delete inverse & forward maps, then create new foward & inverse map
        """
        if key in self.keys():
            del self[key]
        if value in self._inverse.keys():
            del self._inverse[value]
        super().__setitem__(key, value)
        super(_PairedDict, self._inverse).__setitem__(value, key)

    @classmethod
    def make_pairs(cls, *args, **kwds):
        """Create a pair of dicts that are inverses of each other"""
        fwd = cls(*args, **kwds)
        bwd = cls(invert_dict(fwd), inverse=fwd)
        fwd._inverse = bwd
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
    properties.

    If an association is modified, in either direction, both mappings are
    deleted and a new association is created. Setting ``amap[key1] = key2`` is
    always stored in ``amap.fwd``, with ``amap.bwd`` modified appropriately.

    See Also
    --------
    dict
    collections.ChainMap
    """

    def __init__(self, *args, **kwds):
        self.maps = _PairedDict.make_pairs(*args, **kwds)

    def __delitem__(self, key):
        if key in self.bwd.keys():
            del self.bwd[key]
        else:
            del self.fwd[key]

    @property
    def fwd(self):
        return self.maps[0]

    @property
    def bwd(self):
        return self.maps[1]
