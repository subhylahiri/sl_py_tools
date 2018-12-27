# -*- coding: utf-8 -*-
"""Custom containers
"""
# import collections.abc as abc
import itertools as _it
import operator as _op
import typing as _ty


def default(optional: _ty.Optional[_ty.Any], default_value: _ty.Any):
    """Replace with default if None"""
    return default_value if optional is None else optional


def _eq_broadcast(siz0: int, siz1: int):
    return (siz0 == siz1) or (siz0 == 1) or (siz1 == 1)


def same_shape(shape0: tuple, shape1: tuple, compr: _ty.Callable = _op.eq):
    """Are the two array shapes equivalent, ignoring leading singleton axes?
    """
    if isinstance(shape0, ShapeTuple) and isinstance(shape1, ShapeTuple):
        return same_shape(tuple(shape0), shape1, compr)
    return all(compr(x, y) for x, y in zip(reversed(shape0), reversed(shape1)))


def broadcastable(shape0: tuple, shape1: tuple):
    """Are the two array shapes broadcastable?
    """
    return same_shape(shape0, shape1, _eq_broadcast)


def identical_shape(shape0: tuple, shape1: tuple):
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
                num = default(ind.stop, 0)
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
