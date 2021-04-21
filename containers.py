# -*- coding: utf-8 -*-
"""Custom containers & container routines
"""
from __future__ import annotations

import collections as cn
import contextlib as _cx
import functools as _ft
import itertools as _it
import numbers as _num
import operator as _op
import typing as _ty

import sl_py_tools.arg_tricks as _ag

Var = _ty.TypeVar('Var')
Val = _ty.TypeVar('Val')
# =============================================================================
# Function parameter/return helpers
# =============================================================================
EXCLUDIFY = (str, dict, cn.UserDict)


def _is_iter(arg: _ty.Any, exclude: Excludable = ()) -> bool:
    """Is it a non-exluded iterable?"""
    return (isinstance(arg, cn.abc.Iterable)
            and not isinstance(arg, EXCLUDIFY + exclude))


@_ty.overload
def tuplify(arg: _ty.Iterable[Var], num: int = 1, exclude: Excludable = ()
            ) -> _ty.Tuple[Var, ...]:
    pass

@_ty.overload
def tuplify(arg: Var, num: int = 1, exclude: Excludable = ()
            ) -> _ty.Tuple[Var, ...]:
    pass

def tuplify(arg, num=1, exclude=()):
    """Make argument a tuple.

    If it is an iterable (except `str`, `dict`), it is converted to a `tuple`.
    Otherwise, it is placed in a `tuple`.

    Parameters
    ----------
    arg : Var|Iterable[Var]
        Thing to be turned / put into a `tuple`.
    num : int, optional
        Number of times to put `arg` in `tuple`, default: 1. Not used for
        conversion of iterables.
    exclude : Tuple[Type, ...], optional
        Additional iterable types to exclude from conversion, by default `()`.

    Returns
    -------
    tuplified : Tuple[Var, ...]
        Tuple containing `arg`.
    """
    return tuple(arg) if _is_iter(arg, exclude) else (arg,) * num


@_ty.overload
def listify(arg: _ty.Iterable[Var], num: int = 1, exclude: Excludable = ()
            ) -> _ty.List[Var]:
    pass

@_ty.overload
def listify(arg: Var, num: int = 1, exclude: Excludable = ()) -> _ty.List[Var]:
    pass

def listify(arg, num=1, exclude=()):
    """Make argument a list.

    If it is an iterable (except `str`, `dict`), it is converted to a `list`.
    Otherwise, it is placed in a `list`.

    Parameters
    ----------
    arg : Var|Iterable[Var]
        Thing to be turned / put into a `list`.
    num : int, optional
        Number of times to put `arg` in `list`, default: 1. Not used for
        conversion of iterables.
    exclude : Tuple[Type, ...], optional
        Additional iterable types to exclude from conversion, by default `()`.

    Returns
    -------
    listified : List[Var, ...]
        List containing `arg`.
    """
    return list(arg) if _is_iter(arg, exclude) else [arg] * num


@_ty.overload
def setify(arg: _ty.Iterable[Var], exclude: Excludable = ()) -> _ty.Set[Var]:
    pass

@_ty.overload
def setify(arg: Var, exclude: Excludable = ()) -> _ty.Set[Var]:
    pass

def setify(arg, exclude=()):
    """Make argument a set.

    If it is an iterable (except `str`, `dict`), it is converted to a `set`.
    Otherwise, it is placed in a `set`.

    Parameters
    ----------
    arg : Var|Iterable[Var]
        Thing to be turned / put into a `set`.
    exclude : Tuple[Type, ...], optional
        Additional iterable types to exclude from conversion, by default `()`.

    Returns
    -------
    setified : Set[Var, ...]
        Set containing `arg`.
    """
    return set(arg) if _is_iter(arg, exclude) else {arg}


@_ty.overload
def repeatify(arg: _ty.Iterable[Var], times: _ty.Optional[int] = 1,
              exclude: Excludable = ()) -> _ty.Iterable[Var]:
    pass

@_ty.overload
def repeatify(arg: Var, times: _ty.Optional[int] = 1, exclude: Excludable = ()
              ) -> _ty.Iterable[Var]:
    pass

def repeatify(arg, times=None, exclude=()):
    """Repeat argument if not iterable

    Parameters
    ----------
    arg : Var or Iterable[Var]
        Argument to repeat
    times : int, optional
        Maximum number of times to repeat, by default `None`.
    exclude : Tuple[Type, ...], optional
        Additional iterable types to exclude from conversion, by default `()`.

    Returns
    -------
    repeated : Iterable[Var]
        Iterable version of `arg`.
    """
    opt = _ag.eval_or_default(times, tuple, ())
    return arg if _is_iter(arg, exclude) else _it.repeat(arg, *opt)


def unseqify(arg: _ty.Sequence[Var]) -> _ty.Optional[InstanceOrSeq[Var]]:
    """Unpack sequence before returning, if not longer than 1.

    If a sequence has a single element, return that. If empty, return `None`.
    Otherwise return the sequence.

    Parameters
    ----------
    arg : Sequence[Var]
        Sequence to be unpacked.

    Returns
    -------
    val : Sequence[Var] or Var or None
        The sequence or its contents (if there are not more than one).
    """
    if len(arg) == 0:
        return None
    if len(arg) == 1:
        return arg[0]
    return arg


def unsetify(arg: _ty.Set[Var]) -> _ty.Optional[InstanceOrSet[Var]]:
    """Unpack set before returning, if not longer than 1.

    If `set` has a single element, return that. If empty, return `None`.
    Otherwise return the `set`.

    Parameters
    ----------
    arg : Set[Var]
        `set` to be unpacked.

    Returns
    -------
    val : Set[Var] or Var or None
        The set or its contents (if there are not more than one).
    """
    if len(arg) == 0:
        return None
    if len(arg) == 1:
        for val in arg:
            return val
    return arg


def seq_get(seq: _ty.Sequence[Val], ind: _ty.Union[int, slice],
            default: _ty.Optional[Val] = None) -> Val:
    """Get an element from a sequence, or default if index is out of range

    Parameters
    ----------
    seq : Sequence[Val]
        The sequence from which we get the element.
    ind : int or slice
        The index of the element we want from `seq`.
    default : Optional[Val], optional
        Value to return if `ind` is out of range for `seq`, by default `None`.

    Returns
    -------
    element : Val
        Element of the sequence, `seq[ind]`, or `default`.
    """
    try:
        return seq[ind]
    except IndexError:
        return default


def map_join(func: _ty.Callable[[Var], _ty.Iterable[Val]],
             iterable: _ty.Iterable[Var]) -> _ty.List[Val]:
    """Like map, but concatenates iterable outputs
    """
    return list(_it.chain.from_iterable(map(func, iterable)))


def unique_nosort(seq: _ty.Iterable[_ty.Hashable]) -> _ty.List[_ty.Hashable]:
    """Make a list of unique members, in the order of first appearance

    Parameters
    ----------
    seq : Iterable[Hashable]
        Sequence of items with repetition

    Returns
    -------
    uniqued : List[Hashable]
        List of items with repetitions removed
    """
    return list(dict.fromkeys(seq))


@_cx.contextmanager
def appended(seq: _ty.List[Val], *extra: Val) -> _ty.List[Val]:
    """Context manager where list has additional items

    Parameters
    ----------
    seq : List[Val]
        Base List
    *extra : Val
        Appended to `seq` in context, then removed after

    Yields
    -------
    extended : List[Val]
        `seq` with extra items appended.

    Notes
    -----
    It decides which elements to remove after the context based on length.
    If any elements are inserted into/removed from the original elements,
    this will be messed up.
    """
    before = len(seq)
    try:
        seq.extend(extra)
        yield seq
    finally:
        del seq[before:]


@_cx.contextmanager
def extended(seq: _ty.List[Val], extra: _ty.Iterable[Val]) -> _ty.List[Val]:
    """Context manager where list has additional items

    Parameters
    ----------
    seq : List[Val]
        Base List
    extra : Iterable[Val]
        Appended to `seq` in context, then removed after

    Yields
    -------
    extended : List[Val]
        `seq` with extra items appended.

    Notes
    -----
    It decides which elements to remove after the context based on length.
    If any elements are inserted into/removed from the original elements,
    this will be messed up.
    """
    before = len(seq)
    try:
        seq.extend(extra)
        yield seq
    finally:
        del seq[before:]


def rev_seq(seq: _ty.Reversible) -> _ty.Reversible:
    """reverse a sequence, leaving it a sequence if possible

    Parameters
    ----------
    seq : _ty.Reversible
        Either a sequence or a reversible iterator

    Returns
    -------
    rseq : _ty.Reversible
        If `seq` is a sequence, this is the sequence in reversed order,
        otherwise it is a reversed iterator over `seq`.
    """
    if isinstance(seq, cn.abc.Sequence):
        return seq[::-1]
    return reversed(seq)


# =============================================================================
# Classes
# =============================================================================


class ZipSequences(cn.abc.Sequence):
    """Like zip, but sized, subscriptable and reversible (if arguments are).

    Parameters
    ----------
    sequence1, sequence2, ...
        sequences to iterate over
    usemax : bool, keyword only, default=False
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.

    Raises
    ------
    TypeError
        When calling `len` if any memeber is is not `Sized`.
        When calling `reverse` if any memeber is is not `Reversible`.
        When subscripting if any memeber is is not subscriptable.

    Notes
    -----
    If sequences are not of equal length, the reversed iterator will not yield
    the same tuples as the original iterator. Each sequence is reversed as is,
    without omitting end-values or adding fill-values. Similar considerations
    apply to negative indices.

    Indexing with an integer returns a (tuple of) sequence content(s).
    Indexing with a slice returns a (tuple of) sub-sequence(s).
    """
    _seqs: _ty.Tuple[_ty.Sequence, ...]
    _max: bool

    def __init__(self, *sequences: _ty.Sequence, usemax: bool = False) -> None:
        self._seqs = sequences
        self._max = usemax

    def __len__(self) -> int:
        if self._max:
            return max(len(obj) for obj in self._seqs)
        return min(len(obj) for obj in self._seqs)

    def __iter__(self) -> _ty.Union[zip, _it.zip_longest]:
        if self._max:
            return iter(_it.zip_longest(*self._seqs))
        return iter(zip(*self._seqs))

    def __getitem__(self, index: _ty.Union[int, slice]):
        if self._max:
            return unseqify(tuple(seq_get(obj, index) for obj in self._seqs))
        return unseqify(tuple(obj[index] for obj in self._seqs))

    def __reversed__(self) -> ZipSequences:
        return ZipSequences(*(rev_seq(obj) for obj in self._seqs),
                            usemax=self._max)

    def __repr__(self) -> str:
        return type(self).__name__ + repr(self._seqs)

    def __str__(self) -> str:
        seqs = ','.join(type(s).__name__ for s in self._seqs)
        return type(self).__name__ + f'({seqs})'


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
                 inclusive: InstanceOrIter[bool] = (True, False)) -> None:
        if start > stop:
            raise ValueError(f"start={start} > stop={stop}")
        self.start = start
        self.stop = stop
        self.inclusive = listify(inclusive, 2)

    def __contains__(self, x: _num.Real) -> bool:
        return ((self.start < x < self.stop)
                or (self.inclusive[0] and x == self.start)
                or (self.inclusive[1] and x == self.stop))

    def clip(self, val: _num.Real) -> _num.Real:
        """Clip value to lie in interval
        """
        return min(max(val, self.start), self.stop)

# =============================================================================
# Shapes and Tuples for array broadcasting
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
# Subscriptable functions
# =============================================================================


class SubscriptFunction:
    """Decorate a function so that it can be subscripted to call.
    """
    func: _ty.Callable[..., _ty.Any]

    def __init__(self, func: _ty.Callable[..., _ty.Any]) -> None:
        self.func = func
        _ft.update_wrapper(self, func)

    def __getitem__(self, key: _ty.Any) -> _ty.Any:
        return self.func(*tuplify(key))

    def __call__(self, *args: _ty.Any, **kwds: _ty.Any) -> _ty.Any:
        return self.func(*args, **kwds)


def subscriptable(func: _ty.Callable[..., _ty.Any]) -> SubscriptFunction:
    """Decorate a function so that it can be subscripted to call.

    Parameters
    ----------
    func : Callable[..., Any]
        The function being wrapped. There are no restrictions on its signature.

    Returns
    -------
    obj : SubscriptFunction
        Subscriptable and callable object. `obj(...)` will return `func(...)`.
        `obj[a, b, c]` will return `func(a, b, c)` (so tuples are unpacked),
        but with slice and ellipsis notation (`a:b:c` and `...`) converted to
        `slice` and `Ellipsis` objects.

    Example
    -------
    ```
    @subscriptable
    def index_exp(*args):
        return args
    ```
    """
    return SubscriptFunction(func)


# =============================================================================
# Subscriptable properties
# =============================================================================


class SubscriptProxy:
    """Object that passes subscripts to functions

    Parameters
    ----------
    getit : Callable[Var->Val], optional
        Function that returns items, by default `None`
    setit : Callable[Var,Val->None], optional
        Function that sets items, by default `None`
    delit : Callable[Var->None], optional
        Function that deletes items, by default `None`
    doc : str|None, optional
        Docstring, by default `None`
    """
    getit: Getter
    setit: Setter
    delit: Deleter

    def __init__(self, getit: Getter = None, setit: Setter = None,
                 delit: Deleter = None, doc: _ty.Optional[str] = None) -> None:
        self.getit = getit
        self.setit = setit
        self.delit = delit
        self.__doc__ = doc

    def __getitem__(self, key: Var) -> Val:
        if self.getit is None:
            raise AttributeError("Cannot be read")
        return self.getit(key)

    def __setitem__(self, key: Var, value: Val) -> None:
        if self.setit is None:
            raise AttributeError("Cannot be set")
        self.setit(key, value)

    def __delitem__(self, key: Var) -> None:
        if self.delit is None:
            raise AttributeError("Cannot be deleted")
        self.delit(key)


class SubscriptProperty:
    """Subscriptable property.

    This can be used to decorate the methods of a class in a similar way to the
    `property` decorator. The signature of the methods should be the same as
    `__getitem__`, `__setitem__` and `__delitem__` methods. The resulting
    property is then used by subscripting rather than calling.

    Parameters
    ----------
    getit : Callable[Owner,Var->Val], optional
        Function that returns items, by default `None`
    setit : Callable[Owner,Var,Val->None], optional
        Function that sets items, by default `None`
    delit : Callable[Owner,Var->None], optional
        Function that deletes items, by default `None`
    doc : str|None, optional
        Docstring, by default `None`

    Examples
    --------
    ```
    class MyList:
        '''List access two ways
        '''
        def __init__(self, mylist) -> None:
            self.mylist = mylist

        def __getitem__(self, key):
            return self.mylist[key]

        def __setitem__(self, key, value):
            self.mylist[key] = value

        def __delitem__(self, key):
            del self.mylist[key]

        @SubscriptProperty
        def aslist(self, key):
            '''Treats int indices as length 1 slices. Always returns a list,
            always expects an iterable when setting.
            '''
            try:
                return listify(self.mylist[key])
            except IndexError:
                return []

        @aslist.setter
        def aslist(self, key, value):
            if isinstance(value, set):
                self.mylist[key] = unsetify(value)
            else:
                self.mylist[key] = unseqify(value)

        @aslist.deleter
        def aslist(self, key):
            del self.mylist[key]
    ```
    """
    getit: GetterMethod
    setit: SetterMethod
    delit: DeleterMethod
    name: str

    def __init__(self, getit: GetterMethod = None, setit: SetterMethod = None,
                 delit: DeleterMethod = None, doc: _ty.Optional[str] = None
                 ) -> None:
        self.getit = getit
        self.setit = setit
        self.delit = delit
        self.__doc__ = _ag.default(doc, self._get_fun_attr('__doc__'))
        self.name = self._get_fun_attr('__name__', '')
        for fun in self._get_funs():
            _set_doc_name(self, fun)

    def __set_name__(self, owner: _ty.Type[Owner], name: str) -> None:
        self.name = name

    def __get__(self, obj: Owner, objtype: OwnerType = None) -> SubscriptProxy:
        if obj is None:
            return self
        fns = [_make_fn(fun, obj) for fun in self._get_funs()]
        obj.__dict__[self.name] = SubscriptProxy(*fns, self.__doc__)
        return obj.__dict__[self.name]

    def getter(self, getit: GetterMethod) -> SubscriptProperty:
        """Decorate the method that implements __getitem__"""
        return type(self)(getit, self.setit, self.delit, self.__doc__)

    def setter(self, setit: SetterMethod) -> SubscriptProperty:
        """Decorate the method that implements __setitem__"""
        return type(self)(self.getit, setit, self.delit, self.__doc__)

    def deleter(self, delit: DeleterMethod) -> SubscriptProperty:
        """Decorate the method that implements __delitem__"""
        return type(self)(self.getit, self.setit, delit, self.__doc__)

    def _get_funs(self) -> _ty.Tuple[PropMethod, ...]:
        """Get `getit, setit, delit` in a tuple"""
        return self.getit, self.setit, self.delit

    def _get_fun_attr(self, attr: str, default: _ty.Any = None) -> _ty.Any:
        """Get attribute from first of `getit, setit, delit` that has it"""
        return getattr(self.getit, attr,
                       getattr(self.setit, attr,
                               getattr(self.delit, attr, default)))


def _make_fn(fun: PropMethod, obj: Owner) -> _ft.partial:
    """Make partial object from function attribute"""
    return None if fun is None else _ft.partial(fun, obj)


def _set_doc_name(prop: SubscriptProperty, fun: PropMethod) -> None:
    """Set docstring and name of function attribute"""
    if fun is not None:
        fun.__doc__, fun.__name__ = prop.__doc__, prop.name


# =============================================================================
# Hints, aliases
# =============================================================================
untuplify = unseqify
unlistify = unseqify
Owner = _ty.TypeVar('Owner')
InstanceOrIter = _ty.Union[Var, _ty.Iterable[Var]]
InstanceOrSeq = _ty.Union[Var, _ty.Sequence[Var]]
InstanceOrSet = _ty.Union[Var, _ty.Set[Var]]
Excludable = _ty.Tuple[_ty.Type[_ty.Iterable], ...]
Getter = _ty.Optional[_ty.Callable[[Var], Val]]
Setter = _ty.Optional[_ty.Callable[[Var, Val], None]]
Deleter = _ty.Optional[_ty.Callable[[Var], None]]
OwnerType = _ty.Optional[_ty.Type[Owner]]
GetterMethod = _ty.Optional[_ty.Callable[[Owner, Var], Val]]
SetterMethod = _ty.Optional[_ty.Callable[[Owner, Var, Val], None]]
DeleterMethod = _ty.Optional[_ty.Callable[[Owner, Var], None]]
PropMethod = _ty.Union[GetterMethod, SetterMethod, DeleterMethod]
