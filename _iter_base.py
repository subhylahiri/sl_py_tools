# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Sep 20 13:32:47 2018
#
# @author: Subhy
# =============================================================================
"""
Base classes and behind the scenes work for modules ``display_tricks`` and
``iter_tricks``.
"""
from abc import abstractmethod
from collections.abc import Sized, Iterator
from typing import Optional, Union, Tuple, Iterable, Dict, Callable
from functools import wraps

from .display_tricks import _DisplayState as DisplayState
from .display_tricks import DisplayTemporary
from .arg_tricks import default

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
# =============================================================================
# %%* Utility functions
# =============================================================================


def extract_name(args: DArgs, kwds: DKeys) -> (NameArg, Args):
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
    start = default(start, 0)
    step = default(step, 1)
    return start, stop, step


def counter_format(num: int) -> str:
    """Format string for counters that run up to num

    Produces ' 4/10', etc.
    """
    num_dig = str(len(str(num)))
    formatter = '{:>' + num_dig + 'd}/'
    formatter += formatter.format(num)[:-1]
    return formatter


# -----------------------------------------------------------------------------
# %%* Decorators/wrappers
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
        wrapper for `it_func` that eats the name argument and passes the rest.
    """
    @wraps(it_func)
    def no_disp_it_func(*args, **kwds):
        _, no_disp_args = extract_name(args, kwds)
        return it_func(*no_disp_args, **kwds)

    return no_disp_it_func


# =============================================================================
# %%* Client class
# =============================================================================


class DisplayCntState(DisplayState):
    """Internal state of a DisplayCount, etc."""
    prefix: Union[str, DisplayTemporary]

    def __init__(self, prev_state: Optional[DisplayState] = None):
        """Construct internal state"""
        super().__init__(prev_state)
        self.prefix = ''

    def show(self, *args, **kwds):
        """Display prefix"""
        self.prefix = DisplayTemporary.show(self.prefix)

    def hide(self, *args, **kwds):
        """Delete prefix"""
        self.prefix.end()


# =============================================================================
# %%* Mixins for defining displaying iterators
# =============================================================================


class DisplayMixin(DisplayTemporary, Iterator):
    """Mixin providing non-iterator machinery for DisplayCount etc.

    This is an ABC. Only implements `begin`, `disp`, `end` and private stuff.
    Subclasses must implement `iter` and `next`. They must set ``counter``.
    """
    counter: Optional[int]
    offset: int
    formatter: str
    _state: DisplayCntState

    def __init__(self, **kwds):
        """Construct non-iterator machinery"""
        super().__init__(**kwds)
        self._state = DisplayCntState(self._state)
        self.counter = None
        self.formatter = '{:d}'

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def begin(self, msg: str = ''):
        """Display initial counter with prefix."""
        self._state.show()
        super().begin(self.format(self.counter) + msg)

    def update(self, msg: str = ''):
        """Erase previous counter and display new one."""
        dsp = self.format(self.counter)
        super().update(dsp + msg)

    def end(self):
        """Erase previous counter and prefix."""
        super().end()
        self._state.hide()

    def format(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
        return self.formatter.format(*(n + self.offset for n in ctrs))


class AddDisplayToIterables(Iterator):
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
    **kwds
        Keywords other than `name` are passed to the ``displayer`` constructor.
    """
    _iterables: Tuple[Iterable, ...]
    display: DisplayMixin

    def __init_subclass__(cls, displayer, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.displayer = displayer

    def __init__(self, *args: DZipArg, **kwds):
        """Construct the displayer"""
        name, self._iterables = extract_name(args, kwds)
        self.display = self.displayer(name, len(self), **kwds)

    def __reversed__(self):
        """Prepare to display final counter with prefix.

        Assumes `self.display` and all iterables have `__reversed__` methods.
        """
        try:
            self.display = reversed(self.display)
        except AttributeError as exc:
            raise AttributeError('The displayer is not reversible.') from exc
        try:
            self._iterables = tuple(reversed(seq) for seq in self._iterables)
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
        return min((len(seq) for seq in
                    filter(lambda x: isinstance(x, Sized), self._iterables)),
                   default=None)

    def begin(self, *args, **kwds):
        """Display initial counter with prefix."""
        self.display.begin(*args, **kwds)

    def update(self, *args, **kwds):
        """Erase previous counter and display new one."""
        self.display.update(*args, **kwds)

    def end(self):
        """Erase previous counter and prefix."""
        self.display.end()
