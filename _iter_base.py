# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Sep 20 13:32:47 2018
#
# @author: Subhy
# =============================================================================
"""
Base classes and behind the scenes work for `iter_tricks`, `range_tricks`
and `slice_tricks` modules.
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sized
from functools import wraps
from numbers import Number
from operator import add, floordiv, mul
from typing import (Callable, ClassVar, Dict, Iterable, Optional, Tuple,
                    TypeVar, Union)

from .arg_tricks import default
from .containers import rev_seq, tuplify
from .display_tricks import FormattedTempDisplay

# =============================================================================
# %* Utility functions
# =============================================================================


def extract_name(args: DArgs, kwds: DKeys) -> Tuple[NameArg, Args]:
    """Extract name from other args

    If name is in kwds, assume all of args is others, pop name from kwds.
    Else, if args[0] is a str or None, assume it's name & args[1:] is others.
    Else, name is None and all of args is others.

    Parameters
    ----------
    args
        `tuple` of arguments.
    kwargs
        `dict` of keyword arguments. Keyword `name` is popped if present.

    Returns
    -------
    name
        The name, from keyword or first argument if `str`.
    others
        `tuple` of other arguments.
    """
    name = None
    others = args
    if 'name' not in kwds and isinstance(args[0], (str, type(None))):
        name = args[0]
        others = args[1:]
    name = kwds.pop('name', name)
    return name, others


def extract_slice(args: SliceArgs, kwargs: SliceKeys) -> SliceArgs:
    """Extract slice indices from args/kwargs

    Parameters
    ----------
    args
        `tuple` of arguments.
    kwargs
        `dict` of keyword arguments. Keywords below are popped if present.

    Returns
    -------
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
    """
    if not args:
        inds = slice(None)
    elif len(args) == 1 and isinstance(args[0], slice):
        inds = args[0]
    else:
        inds = slice(*args)
    start = kwargs.pop('start', inds.start)
    stop = kwargs.pop('stop', inds.stop)
    step = kwargs.pop('step', inds.step)
    step = default(step, 1)
    if step > 0:
        start = default(start, 0)
    else:
        stop = default(stop, -1)
    return start, stop, step


def counter_format(num: Optional[int]) -> str:
    """Format string for counters that run up to num

    Produces ' 4/10', etc.
    """
    if num is None:
        return '{:d},'
    num_dig = str(len(str(num)))
    formatter = '{:>' + num_dig + 'd}'
    formatter += f'/{formatter.format(num)},'
    return formatter


# -----------------------------------------------------------------------------
# %* Decorators/wrappers
# -----------------------------------------------------------------------------


def and_reverse(it_func: Callable[..., Iterable]) -> Callable[..., Iterable]:
    """Wrap iterator factory with reversed
    """
    @wraps(it_func)
    def rev_it_func(*args, **kwds):
        return reversed(it_func(*args, **kwds))

    new_name = it_func.__name__ + '.rev'
    rev_it_func.__name__ = new_name
    it_func.rev = rev_it_func
    rev_it_func.rev = it_func
#    __all__.append(new_name)
#    setattr(current_module, new_name, rev_it_func)
    return it_func


def without_disp(it_func: Callable[..., Iterable]) -> Callable[..., Iterable]:
    """Create iterator factory with non displaying version

    Parameters
    ----------
    it_func : Callable
        The iterator creating function to use instead

    Returns
    -------
    no_disp_it_func : Callable
        wrapper for `it_func`. It eats the `name` argument and passes the rest.
        It also eats keywords `(offset, disp_step, addto, usemax)`.
    """
    @wraps(it_func)
    @and_reverse
    def no_disp_it_func(*args, **kwds):
        _, no_disp_args = extract_name(args, kwds)
        kwds.pop('offset', None)
        kwds.pop('disp_step', None)
        kwds.pop('addto', None)
        kwds.pop('usemax', None)
        return it_func(*no_disp_args, **kwds)

    return no_disp_it_func

# -----------------------------------------------------------------------------
# %* Slice based factories
# -----------------------------------------------------------------------------


def st_args(with_st: Union[slice, range]) -> SliceArgs:
    """Extract start, stop, step from slice/range-like object
    """
    return with_st.start, with_st.stop, with_st.step


class SliceToIter:
    """Use slice indexing to create iterators

    Parameters
    ----------
    iterfun : Callable[SliceArgs->Iterable], optional
        Function to convert `(start,stop,step)` to iterator. Default: `range`
    argnum : int, optional
        Which argument to `__getitem__` is the slice to convert? Default: 0
    slicefun : Callable[slice->SliceArgs], optional
        Function to convert slice to `(start,stop,step)`. Default: `st_args`
    """
    argnum: int
    iterfun: ConvOut[Iterable]
    slicefun: ConvIn[slice]

    def __init__(self,
                 iterfun: ConvOut[Iterable] = range,
                 argnum: int = 0,
                 slicefun: ConvIn[slice] = st_args,
                 ):
        super().__init__()
        self.iterfun = iterfun
        self.slicefun = slicefun
        self.argnum = argnum

    def __getitem__(self, arg) -> Iterable:
        arg = tuplify(arg)
        return self.convert(*arg)

    def convert(self, *args) -> Iterable:
        """Convert slice arguments to an iterator

        Returns
        -------
        the_iter : Iterable
            Resulting iterator
        """
        argspre = args[:self.argnum]
        argspost = args[self.argnum+1:]
        argsslc = self.slicefun(args[self.argnum])
        return self.iterfun(*argspre, *argsslc, *argspost)


def min_len(*iterables) -> int:
    """Length of shortest sequence.
    """
    return min((len(seq) for seq in
                filter(lambda x: isinstance(x, Sized), iterables)),
               default=None)


def max_len(*iterables) -> int:
    """Length of shortest sequence.
    """
    return max((len(seq) for seq in
                filter(lambda x: isinstance(x, Sized), iterables)),
               default=None)


# =============================================================================
# Mixins for defining displaying iterators
# =============================================================================


class DisplayMixin:
    """Mixin providing non-iterator machinery for DisplayCount etc.

    This is an ABC. Only implements `begin`, `disp`, `end` and private stuff.
    Subclasses must implement `iter` and `next`. They must set ``counter``.
    """
    counter: Optional[int]
    offset: int
    formatter: FormattedTempDisplay
    prefix: FormattedTempDisplay
    # set debug to True to check that displays are properly nested
    debug: ClassVar[bool] = False

    def __init__(self, **kwds):
        """Construct non-iterator machinery"""
        self.counter = None
        self.offset = kwds.pop('offset', 0)
        self.prefix = kwds.pop('prefix', '')
        if isinstance(self.prefix, str):
            self.prefix = FormattedTempDisplay(self.prefix)
        self.formatter = kwds.pop('formatter', '{:d},')
        if isinstance(self.formatter, str):
            self.formatter = FormattedTempDisplay(self.formatter)

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def begin(self):  # , msg: str = ''):
        """Display initial counter with prefix."""
        if self.prefix.numchar:
            raise RuntimeError(
                '''DisplayMixin.begin() was called a second time.
                It should only be called once.''') from None
        self.prefix()
        self.update()
        if self.debug:
            self._check()

    def update(self):  # , msg: str = ''):
        """Erase previous counter and display new one."""
        self.formatter(*(n + self.offset for n in tuplify(self.counter)))
        if self.debug:
            self._check()

    def end(self):
        """Erase previous counter and prefix."""
        # super().end()
        self.formatter.end()
        self.prefix.end()
        if self.debug:
            self._check()

    def _check(self, msg: str = '') -> str:
        """Ensure that DisplayTemporaries are properly used

        Can be overloaded in subclasses
        """
        # raise error if msg is non-empty
        if msg and self.debug:
            raise IndexError(msg)


class AddDisplayToIterables:
    """Wraps iterator to display progress.

    This is an ABC. Only implements `begin`, `update` and `end` methods, as
    well as `__init__`, `__reversed__` and `__len__`. Subclasses must implement
    `__iter__` and `__next__`.

    Specify ``displayer`` in keyword arguments of subclass definition to
    customise display. There is no default, but ``iter_tricks.DisplayCount`` is
    suggested. It must implement the ``DisplayMixin`` interface with
    constructor signature ``displayer(name, len(self), **kwds)``, with `self`
    an instance of the subclass being defined.

    Subclasses of subclasses will also have to specify a ``displayer`` unless
    an intermediate subclass redefines `__init_subclass__`.

    Parameters
    ----------
    name: str
        Name to be used for display. See `extract_name`.
    iterable1, ...
        The iterables being wrapped.
    usemax : bool, keyword only, default=False
        If True, we continue until all sequences are exhausted. If False, we
        stop when we reach the end of the shortest sequence.
    **kwds
        Keywords other than `name` are passed to the ``displayer`` constructor.
    """
    _iterables: Tuple[Iterable, ...]
    display: DisplayMixin
    _max: bool

    def __init_subclass__(cls, displayer, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.displayer = displayer

    def __init__(self, *args: DZipArg, **kwds):
        """Construct the displayer"""
        name, self._iterables = extract_name(args, kwds)
        addto = kwds.pop('addto', 0)
        self._max = kwds.pop('usemax', False)
        # self._iterables = ZipSequences(*iterables, usemax=self._max)
        stop = len(self)
        if stop is None:
            self.display = self.displayer(name, addto, stop, **kwds)
        else:
            self.display = self.displayer(name, addto, addto + stop, **kwds)

    def __reversed__(self):
        """Prepare to display final counter with prefix.

        Assumes `self.display` and all iterables have `__reversed__` methods.
        """
        try:
            self.display = reversed(self.display)
        except AttributeError as exc:
            raise AttributeError('The displayer is not reversible.') from exc
        try:
            self._iterables = tuple(rev_seq(seq) for seq in self._iterables)
        except AttributeError as exc:
            raise AttributeError('Some iterables are not reversible.') from exc
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        """Length of shortest sequence.
        """
        if self._max:
            return max_len(*self._iterables)
        return min_len(*self._iterables)

    def begin(self, *args, **kwds):
        """Display initial counter with prefix."""
        self.display.begin(*args, **kwds)

    def update(self, *args, **kwds):
        """Erase previous counter and display new one."""
        self.display.update(*args, **kwds)

    def end(self):
        """Erase previous counter and prefix."""
        # if self.__len__() is None:
        if self.display.formatter.numchar:
            self.display.end()


# -----------------------------------------------------------------------------
# Arithmetic operations
# -----------------------------------------------------------------------------


def _num_only_l(opr: NumOp) -> Callable[[NumOrNone, Number], NumOrNone]:
    """Wrap an operator to only act on numbers
    """
    def wrapper(left: SomeType, right: Number) -> SomeType:
        if isinstance(left, Number):
            return opr(left, right)
        return left
    return wrapper


def _num_only_r(opr: NumOp) -> Callable[[Number, NumOrNone], NumOrNone]:
    """Wrap an operator to only act on numbers
    """
    def wrapper(left: Number, right: SomeType) -> SomeType:
        if isinstance(right, Number):
            return opr(left, right)
        return right
    return wrapper


def num_only(opr: NumOp) -> Callable[[NumOrNone, NumOrNone], NumOrNone]:
    """Wrap an operator to only act on numbers
    """
    return _num_only_l(_num_only_r(opr))


def raise_if_steps(left: SliceArgs, right: SliceArgs):
    """raise ValueError if steps do not match"""
    lstep, rstep = left[2], right[2]
    if not ((lstep is None) or (rstep is None) or (lstep == rstep)):
        raise ValueError(f"Incompatible steps: {lstep} and {rstep}")


def wrap_op(args: Tuple[SArgsOrNum, ...], opr: NumOp, step: bool) -> SliceArgs:
    """Perform operation on range arguments."""
    flex_op = num_only(opr)
    if step:
        return [flex_op(s, t) for s, t in zip(*args)]
    raise_if_steps(*args)
    ops = (flex_op, flex_op, default)
    return [_op(s, t) for _op, s, t in zip(ops, *args)]


def conv_in_wrap(func: ConvIn) -> Callable[[SorNum], Tuple[SliceArgs, bool]]:
    """wrap a function that converts Iterator to (start, top, step)."""
    def conv_range_args(arg: SorNum) -> Tuple[SliceArgs, bool]:
        """return range (args), True or (arg,arg,arg), False."""
        if isinstance(arg, Number):
            return (arg, arg, None), False
        return func(arg), True
    return conv_range_args


def _range_ops(opr: NumOp, case_steps: Tuple[bool, ...],
               conv_in: ConvIn, conv_out: ConvOut, *args: SorNum) -> SomeType:
    """Perform operation on ranges/numbers.

    Parameters
    ----------
    opr
        operator to use
    case_steps : tuple(bool, bool, bool)
        if (both, left, right) argument(s) isinstance(SomeType), that element
        of `case_steps` will be passed to `wrap_op`.
    conv_in
        function to convert inputs to input argument
    conv_out
        function to convert output arguments to output
    args
        [left, right] argument to operator
    """
    conv_in = conv_in_wrap(conv_in)
    args, is_rng = zip(*[conv_in(x) for x in args])
    if not any(is_rng):
        return opr(*args)
    is_rng = (all(is_rng),) + is_rng
    case_steps = case_steps[is_rng.index(True)]
    if case_steps is None:
        raise TypeError("Unsupported operation")
    return conv_out(*wrap_op(*args, opr, case_steps))


def arg_ops(cin: ConvIn, cout: ConvOut) ->Tuple[Callable[[SorNum, SorNum], SomeType], ...]:
    """Create arithmetic operations for `RangeIsh/SliceIsh` objects.

    Parameters
    ----------
    cin : Callable[RangeIsh -> SliceArgs]
        Convert `RangeIsh/SliceIsh` object to `start, stop, step`.
    cout : Callable[[SliceArg, SliceArg, SliceArg] -> SomeType]
        Convert `start, stop, step` to `RangeIsh/SliceIsh` object.

    Returns
    -------
    adder: Callable[[SorNum, SorNum], SomeType]
        Operator that performs addition.
    subtractor: Callable[[SorNum, SorNum], SomeType]
        Operator that performs subtraction.
    multiplier: Callable[[SorNum, SorNum], SomeType]
        Operator that performs multiplication.
    divider: Callable[[SomeType, Number], SomeType]
        Operator that performs division.
    """

    def adder(left: SorNum, right: SorNum) -> slice:
        """Add slices / ranges / numbers.

        Parameters
        ----------
        left, right : RangeIsh|SliceIsh|Number
            Arguments to add.

        Raises
        ------
        ValueError
            If `step`s are incompatible.
        """
        return _range_ops(add, (False, False, False), cin, cout, left, right)

    def subtractor(left: SorNum, right: SorNum) -> slice:
        """Subtract slices / ranges / numbers.

        Parameters
        ----------
        left, right : RangeIsh|SliceIsh|Number
            Arguments to subtract.

        Raises
        ------
        ValueError
            If `step`s are incompatible
        """
        try:
            return adder(left, multiplier(right, -1, True))
        except ValueError:
            return adder(left, multiplier(right, -1, False))

    def multiplier(left: SorNum, right: SorNum, step: bool = True) -> slice:
        """Multiply slice / range by a number.

        Parameters
        ----------
        left, right : RangeIsh|SliceIsh|Number
            Arguments to multiply. Cannot both be `SliceIsh`.
        step : bool
            Also multiply step?

        Raises
        ------
        TypeError
            If neither `left` nor `right is a number.`
        """
        return _range_ops(mul, (None, step, step), cin, cout, left, right)

    def divider(left: SomeType, right: Number, step: bool = True) -> slice:
        """divide slice / range by a number.

        Parameters
        ----------
        left : RangeIsh|SliceIsh
            Argument to divide.
        right : Number
            Argument to divide by.
        step : bool
            Also divide step?

        Raises
        ------
        TypeError
            If `right` is not a number.
        """
        return _range_ops(floordiv, (None, step, None), cin, cout, left, right)

    return adder, subtractor, multiplier, divider


# =============================================================================
# Hinting
# =============================================================================
NameArg = Optional[str]
SliceArg = Optional[int]
DZipArg = Union[NameArg, Iterable]
DSliceArg = Union[NameArg, SliceArg]
Arg = Union[SliceArg, Iterable]
DArg = Union[NameArg, Arg]
SliceArgs = Tuple[SliceArg, ...]
DSliceArgs = Tuple[DSliceArg, ...]
Args = Tuple[Arg, ...]
DArgs = Tuple[DArg, ...]
SliceKeys = Dict[str, SliceArg]
DSliceKeys = Dict[str, DSliceArg]
Keys = Dict[str, Arg]
DKeys = Dict[str, DArg]

NumOrNone = TypeVar('NumOrNone', Number, None)
NumOp = Callable[[Number, Number], Number]
SomeType = TypeVar('SomeType')
SorNum = Union[SomeType, Number]
SArgsOrNum = Union[SliceArgs, Number]
ConvIn = Callable[[SomeType], SliceArgs]
ConvOut = Callable[[SliceArg, SliceArg, SliceArg], SomeType]
