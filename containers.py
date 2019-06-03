# -*- coding: utf-8 -*-
"""Custom containers
"""
from __future__ import annotations
import collections as cn
import itertools as _it
import operator as _op
import typing as _ty
import numbers as _num
from .arg_tricks import defaults

A = _ty.TypeVar('A')
B = _ty.TypeVar('B')
InstanceOrIterable = _ty.Union[A, _ty.Iterable[A]]
Dictable = _ty.Union[_ty.Mapping[A, B], _ty.Iterable[_ty.Tuple[A, B]]]


def tuplify(arg: InstanceOrIterable, num: int = 1) -> _ty.Tuple[A, ...]:
    """Make argument a tuple.

    If it is an iterable, it is converted to a tuple.
    Otherwise, it is placed in a tuple.

    Parameters
    ----------
    arg
        Thing to be turned / put into a tuple.
    num : int, optional
        Number of times to put `arg` in tuple, default: 1. Not used for
        conversion of iterables.
    """
    if isinstance(arg, cn.abc.Iterable):
        return tuple(arg)
    return (arg,) * num


def listify(arg: InstanceOrIterable, num: int = 1) -> _ty.List[A]:
    """Make argument a list.

    If it is an iterable, it is converted to a list.
    Otherwise, it is placed in a list.

    Parameters
    ----------
    arg
        Thing to be turned / put into a list.
    num : int, optional
        Number of times to put `arg` in tuple, default: 1. Not used for
        conversion of iterables.
    """
    if isinstance(arg, cn.abc.Iterable):
        return list(arg)
    return [arg] * num


class Interval(cn.abc.Container):
    """An interval of the real line.

    For testing upper and lower bounds with ``x in Interval(a,b)``.

    Parameters
    ----------
    start, stop : Real
        Lower and upper bounds of the interval.
    inclusive : {bool, Sequence[bool,bool]}, optional, default: (True, False)
        Is the (lower,upper) bound inclusive? Scalars apply to both ends.
    """
    start: _num.Real
    stop: _num.Real
    inclusive: _ty.List[bool]

    def __init__(self, start: _num.Real, stop: _num.Real,
                 inclusive: InstanceOrIterable[bool] = (True, False)):
        if start > stop:
            raise ValueError(f"start={start} > stop={stop}")
        self.start = start
        self.stop = stop
        self.inclusive = listify(inclusive, 2)

    def __contains__(self, x: _num.Real) -> bool:
        return ((self.start < x and x < self)
                or (self.inclusive[0] and x == self.start)
                or (self.inclusive[1] and x == self.stop))

    def clip(self, val: _num.Real):
        """Clip value to lie in interval
        """
        return min(max(val, self.start), self.stop)


# =============================================================================
# %%* Shapes and Tuples for array broadcasting
# =============================================================================


def _eq_broadcast(siz0: int, siz1: int) -> bool:
    """Would axes of these lengths be considered broadcastable?
    """
    return (siz0 == siz1) or (siz0 == 1) or (siz1 == 1)


def same_shape(shape0: _ty.Tuple[int, ...], shape1: _ty.Tuple[int, ...],
               compr: _ty.Callable[[int, int], bool] = _op.eq) -> bool:
    """Are the two array shapes equivalent, ignoring leading singleton axes?

    Parameters
    ----------
    shape0, shape1 : Tuple[int,...]
        Shapes of arrays
    compr : Callable[(int, int) -> bool], optional, default: ==
        Function used to compare tuple elements.
    """
    if isinstance(shape0, ShapeTuple) and isinstance(shape1, ShapeTuple):
        return same_shape(tuple(shape0), tuple(shape1), compr)
    diff = len(shape0) - len(shape1)
    pad0 = _it.chain(reversed(shape0), (1,) * diff)
    pad1 = _it.chain(reversed(shape1), (1,) * -diff)
    return all(compr(x, y) for x, y in zip(pad0, pad1))


def broadcastable(shape0: _ty.Tuple[int, ...],
                  shape1: _ty.Tuple[int, ...]) -> bool:
    """Are the two array shapes broadcastable?

    Parameters
    ----------
    shape0, shape1 : Tuple[int,...]
        Shapes of arrays
    """
    return same_shape(shape0, shape1, _eq_broadcast)


def identical_shape(shape0: _ty.Tuple[int, ...],
                    shape1: _ty.Tuple[int, ...]) -> bool:
    """Are the two array shapes eaxctly the same, considering all axes?

    Parameters
    ----------
    shape0, shape1 : Tuple[int,...]
        Shapes of arrays
    """
    return (len(shape0) == len(shape1)) and same_shape(shape0, shape1)


class ShapeTuple(tuple):
    """Stores the shapes of array types that implement broadcasting.

    Gives 1 if you ask for elements before the start, either via negative
    indexing, negative slicing or the reversed iterator beyond its length.
    The reversed iterator will never stop iteration by itself.
    """

    def __getitem__(self, ind: _ty.Union[slice, int]):
        if isinstance(ind, slice):
            out = super().__getitem__(ind)
            m, n = len(out), len(self)
            start, stop, step = defaults((ind.start, ind.stop, ind.step),
                                         (0, n, 1))
            if step < 0:
                start, stop = defaults((ind.stop, ind.start), (0, n))
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

    def __eq__(self, other):
        return same_shape(self, other)


# =============================================================================
# %%* Dictionaries
# =============================================================================


def update_new(to_update: dict, update_from: Dictable):
    """Update new keys only
    """
    if isinstance(update_from, cn.abc.Mapping):
        for k in update_from.keys():
            to_update.setdefault(k, update_from[k])
    else:
        for k, v in update_from:
            if k not in to_update.keys():
                to_update[k] = v


def update_existing(to_update: dict, update_from: Dictable):
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

    Assumes values are distinct and hashable.
    """
    return {v: k for k, v in to_invert.items()}


class PairedDict(cn.UserDict):
    """One direction of bidirectional mapping

    Stores a reference to its inverse mapping in `self.inverse`. If the
    inverse is provided in the constructor and ``dict(*args,**kwds)`` is empty,
    `self` will be updated with ``invert_dict(self.inverse)``. If the
    inverse is not provided in the constructor and ``dict(*args,**kwds)`` is
    non-empty, `self.inverse` will be created with ``invert_dict(self)``.
    Otherwise, no effort is made to ensure that they are inverses of each
    other. Instead, the instances should be built using the class-method
    `PairedDict.make_pairs`.

    Deleting an item also deletes the reversed item from `self.inverse`.
    Setting an item with ``self[key1] = key2``, deletes `key1` from `self` as
    above, deletes `key2` from `self.inverse`, adds item `(key1,key2)` to
    `self`, and adds item `(key2,key1)` to `self.inverse`.

    Both keys and values must be unique and hashable.

    Parameters
    ----------
    Positional arguments
        Passed to the `UserDict` constructor.
    inverse
        The inverse dictionary. Can only be given as a keyword argument.
    Other keword arguments
        Passed to the `UserDict` constructor.

    See Also
    --------
    dict
    collections.UserDict
    collections.ChainMap
    AssociativeMap
    """
    inverse: _ty.Optional[PairedDict] = None
    _formed: bool = False

    def __init__(self, *args, inverse=None, **kwds):
        self._formed = False
        super().__init__(*args, **kwds)
        self.inverse = None
        if len(args) > 0 or len(kwds) > 0 or inverse is not None:
            self.inverse = type(self)()
            self.inverse.inverse = self
        if inverse is not None:
            super(PairedDict, self.inverse).update(inverse)
            if len(self) == 0 and len(self.inverse) > 0:
                super().update(invert_dict(self.inverse))
        elif len(self) > 0:
            super(PairedDict, self.inverse).update(invert_dict(self))
        self._formed = True

    def __delitem__(self, key):
        """Delete inverse map as well as forward map"""
        if self._formed and self.inverse is not None:
            super(PairedDict, self.inverse).__delitem__(self[key])
        super().__delitem__(key)

    def __setitem__(self, key, value):
        """Delete inverse & forward maps, then create new foward & inverse map
        """
        if self._formed:
            # not in constructor, assume self.inverse is good
            if key in self.keys():
                del self[key]
            if (self.inverse is not None) and (value in self.inverse.keys()):
                super(PairedDict, self.inverse).__delitem__(value)
        # maybe in constructor, make no assumptions
        super().__setitem__(key, value)
        if self.inverse is not None:
            super(PairedDict, self.inverse).__setitem__(value, key)

    @classmethod
    def make_pairs(cls, *args, **kwds) -> _ty.Tuple[PairedDict, PairedDict]:
        """Create a pair of dicts that are inverses of each other

        Returns
        -------
        [fwd, bwd]
            fwd : PairedDict
                Dictionary built with other parameters.
            bwd : PairedDict
                Inverse of `fwd`

        Raises
        ------
        ValueError
            If 'inverse' is supplied as a keyword argument, or values are not
            unique.
        """
        if 'inverse' in kwds.keys():
            raise ValueError("Cannot use 'inverse' as a keyword here")
        fwd = cls(*args, **kwds)
        bwd = fwd.inverse
        if len(fwd) != len(bwd):
            raise ValueError("Repeated keys/values")
        return [fwd, bwd]


class AssociativeMap(cn.ChainMap):
    """Bidirectional mapping

    Similar to a ``dict``, except the statement ``self.fwd[key1] == key2`` is
    equivalent to ``self.bwd[key2] == key1``. Both of these statements imply
    that ``self[key1] == key2`` and ``self[key2] == key1``. Both keys must be
    unique and hashable.

    An unordered associative map arises when subscripting the object itself.
    An ordered associative map arises when subscripting the ``fwd`` and ``bwd``
    properties, which are both `PairedDict`s and inverses of each other.

    If an association is modified, in either direction, both the forward and
    backward mappings are deleted and a new association is created. (see
    documentation for `PairedDict`). Setting ``self[key1] = key2`` is always
    applied to ``self.fwd``, with ``self.bwd`` modified appropriately.

    See Also
    --------
    dict
    collections.UserDict
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
            raise KeyError(f"Key '{key}' not found in either direction.")

    @property
    def fwd(self) -> PairedDict:
        return self.maps[0]

    @property
    def bwd(self) -> PairedDict:
        return self.maps[1]
