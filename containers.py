# -*- coding: utf-8 -*-
"""Custom containers
"""
from __future__ import annotations
import collections as cn
import itertools as _it
import functools as _ft
import operator as _op
import contextlib as _cx
import typing as _ty
import numbers as _num
from . import arg_tricks as _ag

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

    For testing upper and lower bounds with `x in Interval(a,b)`.

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
    """Are two array shapes the same after padding with leading singleton axes?

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
    The reversed iterator will never stop iteration by itself, must be zipped
    with something finite.
    """

    def __getitem__(self, ind: _ty.Union[slice, int]):
        if isinstance(ind, slice):
            out = super().__getitem__(ind)
            step = _ag.default(ind.step, 1)
            rev = step < 0
            # defaults:- not rev: [0, n], rev: [-1, -n-1]
            my_lims = [[0, len(self)], [-1, -1 - len(self)]]
            lims = [self._posify(x, rev)
                    for x in _ag.defaults([ind.start, ind.stop], my_lims[rev])]
            # number of missing singletons, if positive.
            num = len(range(*lims, step)) - len(out)
            # not rev: (1,)*num + out, rev: out + (1,)*num, num <= 0: out
            return (1,)*(num * (not rev)) + out + (1,)*(num * rev)
        try:
            return super().__getitem__(ind)
        except IndexError:
            if isinstance(ind, int) and ind < -len(self):
                return 1
            raise

    def _posify(self, ind, rev=False):
        """Remap slice indices

        set 0 at start of tuple =>  if ind < 0: ind + len(self)
        clip at end => if step > 0 and ind > len(self): len(self)      (stop)
                    => if step < 0 and ind >= len(self): len(self) - 1 (start)
        """
        if ind < 0:
            return ind + len(self)
        if ind > len(self) - rev:
            return len(self) - rev
        return ind

    def __reversed__(self):
        return _it.chain(reversed(tuple(self)), _it.repeat(1))

    def __eq__(self, other):
        return same_shape(self, other)


# =============================================================================
# %%* Dictionaries
# =============================================================================


def update_new(to_update: dict, update_from: Dictable):
    """Update new keys only

    If `key` in `update_from` but not `to_update`, set `to_update[key] =
    update_from[key]`.
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

    If `key` in `update_from` and `to_update`, set `to_update[key] =
    update_from[key]`.
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

    If `k` in `pop_from` and `to_update`, set `to_update[k] = pop_from[k]`
    and `del pop_from[k]`.
    """
    for k in to_update.keys():
        to_update[k] = pop_from.pop(k, to_update[k])


def pop_new(to_update: dict, pop_from: dict):
    """Pop to update new keys only

    If `k` in `pop_from` but not `to_update`, set `to_update[k] = pop_from[k]`
    and `del pop_from[k]`.
    """
    for k in pop_from.keys():
        if k not in to_update.keys():
            to_update[k] = pop_from.pop(k)


def _inv_dict_iter(to_invert: dict) -> _ty.Iterator:
    """Swap keys and values.

    Can be used to build/update another dict or other container.
    Can only be used once - best not to store in a variable.
    """
    return ((v, k) for k, v in to_invert.items())


def invert_dict(to_invert: dict) -> dict:
    """Swap keys and values.

    Assumes values are distinct and hashable.

    Raises
    ------
    TypeError
        If any of `to_invert.values()` are not hashable.
    ValueError
        If any of `to_invert.values()` are repeated.
    """
    inverted = dict(_inv_dict_iter(to_invert))
    if len(inverted) < len(to_invert):
        raise ValueError(f'Repeated values in {to_invert}?')
    return inverted


def is_inverse_dict(map1: _ty.Mapping, map2: _ty.Mapping) -> bool:
    """Test if two dicts are each others inverses.

    Checks `map2[map1[key]] == key` for every `key` in `map1.keys()` and every
    `key` in `map2.values()`. Does not check order of entries.
    """
    if len(map1) != len(map2):
        return False
    return all(map2[v] == k for k, v in map1.items())


class PairedDict(cn.UserDict):
    """One direction of bidirectional mapping

    Instances Store a reference to their inverse mapping in `self.inverse`.
    Both keys and values must be unique and hashable.

    Deleting an item also deletes the reversed item from `self.inverse`.
    Setting an item with `self[key1] = key2`, deletes `key1` from `self` as
    above, deletes `key2` from `self.inverse`, adds item `(key1, key2)` to
    `self`, and adds item `(key2, key1)` to `self.inverse`.

    Ideally, the instances should be built using the class-method
    `PairedDict.make_pairs`. This will raise a `ValueError` for repeated values

    If you do use the constructor directly: when `inverse` is not provided in
    the constructor, `self.inverse` will be created with `invert_dict(self)`
    without checking for repeated values. When `inverse` is provided in the
    constructor, it will be copied and updated with `invert_dict(self)` and
    `self` will be updated with `invert_dict(self.inverse)`, both without
    checking for repeated values. We recommend running `self.fix_inverse()`
    after construction, which will raise a `ValueError` if there were any
    repeated values, or at least calling `self.check_inverse()`.

    Under normal circumstances `self.inverse.inverse is self` should hold. This
    can break if `self.inverse` is replaced or private machinery is used.
    There is no guarantee that `self.inverse == invert_dict(self)` due to the
    possibility of repeated values, but if it holds after construction it
    should remain True thereafter.

    Parameters
    ----------
    Positional arguments
        Passed to the `UserDict` constructor.
    inverse
        The inverse dictionary. Can only be given as a keyword argument.
    Other keword arguments
        Passed to the `UserDict` constructor.

    Raises
    ------
    ValueError
        If any keys/values are not unique.
    TypeError
        If any keys/values are not hashable.

    See Also
    --------
    dict
    collections.UserDict
    BijectiveMap
    """
    inverse: _ty.Optional[PairedDict] = None
    _formed: bool = False

    def __init__(self, *args, inverse=None, **kwds):
        # ensure we only use super().__setitem__ to prevent infinite recursion
        self._formed = False
        # were we called by another object's __init__?
        secret = kwds.pop('__secret', False)
        # use this to construct self.inverse if inverse is not already ok
        init_fn = _ft.partial(type(self), inverse=self, __secret=True)
        super().__init__(*args, **kwds)

        # self.inverse.__init__ will not be callled if secret is True and
        # inverse is not None.
        if secret:
            self.inverse = _ag.default_eval(inverse, init_fn)
        else:
            self.inverse = _ag.non_default_eval(inverse, init_fn, init_fn)
        # In all calls of self.inverse.__init__ above, __secret is True and
        # inverse is self => no infinite recursion (at most one more call).

        # First object constructed is last to be updated by its inverse
        if secret or inverse is not None:
            self.update(_inv_dict_iter(self.inverse))
        # we can use our own __setitem__ now that self.update is done
        self._formed = True
        # if not (secret or self.check_inverse()):
        #     raise ValueError("Unable to form inverse. Repeated keys/values?")

    def __delitem__(self, key):
        """Delete inverse map as well as forward map"""
        if self._formed:
            # not in constructor/fix_inverse, assume self.inverse is good
            # use super().__delitem__ to avoid infinite recursion
            super(PairedDict, self.inverse).__delitem__(self[key])
        # maybe in constructor/fix_inverse, make no assumptions
        super().__delitem__(key)

    def __setitem__(self, key, value):
        """Delete inverse & forward maps, then create new foward & inverse map
        """
        if self._formed:
            # not in constructor/fix_inverse, assume self.inverse is good
            if key in self.keys():
                del self[key]
            if value in self.inverse.keys():
                del self.inverse[value]
            # use super().__setitem__ to avoid infinite recursion
            super(PairedDict, self.inverse).__setitem__(value, key)
        # maybe in constructor/fix_inverse, make no assumptions
        super().__setitem__(key, value)

    def check_inverse(self) -> bool:
        """Check that inverse has the correct value."""
        if self.inverse is None:
            return False
        if self.inverse.inverse != self:
            return False
        return is_inverse_dict(self, self.inverse)

    def check_inverse_strict(self) -> bool:
        """Check that inverse is correct and properly linked to self."""
        return self.check_inverse() and (self.inverse.inverse is self)

    @_cx.contextmanager
    def _unformed(self):
        try:
            self._formed = False
            yield
        finally:
            self._formed = True

    def fix_inverse(self):
        """Set inverse using self

        If `self.inverse` has not been set, it is created by inverting `self`.
        If they are not inverses of each other, first we try updating
        `self.inverse` with the inverse of `self`. Then we try updating `self`
        with the inverse of `self.inverse`. If they are still not inverses, we
        raise a `ValueError`.
        """
        if self.inverse is None:
            self.inverse = type(self)(inverse=self)
        self.inverse.inverse = self
        if not self.check_inverse():
            with self.inverse._unformed():
                self.inverse.update(_inv_dict_iter(self))
        if not self.check_inverse():
            with self._unformed():
                self.update(_inv_dict_iter(self.inverse))
        if not self.check_inverse():
            raise ValueError("Unable to fix inverse. Repeated keys/values?")

    @classmethod
    def make_pairs(cls, *args, **kwds) -> _ty.Tuple[PairedDict, PairedDict]:
        """Create a pair of dicts that are inverses of each other

        Parameters
        ----------
        All used to construct `fwd`.

        Returns
        -------
        fwd : PairedDict
            Dictionary built with parameters.
        bwd : PairedDict
            Inverse of `fwd`

        Raises
        ------
        ValueError
            If values are not unique or if 'inverse' is used as a keyword.
        TypeError
            If any values are not hashable.
        """
        if 'inverse' in kwds.keys():
            raise ValueError("Cannot use 'inverse' as a keyword here")
        fwd = cls(*args, **kwds)
        fwd.fix_inverse()
        bwd = fwd.inverse
        if not fwd.check_inverse_strict():
            raise ValueError("Repeated keys/values?")
        return fwd, bwd


class BijectiveMap(cn.ChainMap):
    """Bidirectional mapping

    Similar to a `dict`, except the statement `self.fwd[key1] == key2` is
    equivalent to `self.bwd[key2] == key1`. Both of these statements imply
    that `self[key1] == key2` and `self[key2] == key1`. Both keys must be
    unique and hashable.

    A symmetric bijective map arises when subscripting the object itself.
    An asymmetric bijective map arises when subscripting the `fwd` and `bwd`
    properties, which are both `PairedDict`s and inverses of each other.

    If an association is modified, in either direction, both the forward and
    backward mappings are deleted and a new association is created. (see
    documentation for `PairedDict`). Setting `self[key1] = key2` is always
    applied to `self.fwd`, with `self.bwd` modified appropriately. For more
    control, you can call `self.fwd[key1] = key2` or `self.bwd[key2] = key1`.

    See Also
    --------
    dict
    collections.UserDict
    collections.ChainMap
    PairedDict
    """
    maps: _ty.List[PairedDict]

    def __init__(self, *args, **kwds):
        super().__init__(*PairedDict.make_pairs(*args, **kwds))

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
