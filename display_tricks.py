# -*- coding: utf-8 -*-
# =============================================================================
# Created on Tue Jan  9 17:06:58 2018
#
# @author: Subhy
#
# Module: display_tricks
# =============================================================================
"""
Tools for displaying temporary messages.

DisplayTemporary : class
    Class for temporarily displaying a message.

dtemp : function
    Temporarily display a message.
dcontext
    Display message during context.
dexpr
    Display message during lambda execution.
dwrap : function
    Decorate a function with a temporary printed message.

.. warning:: Doesn't display properly on ``qtconsole``, and hence ``Spyder``.
Instead, use in a console connected to the same kernel:
``cd`` to the folder, then type: ``jupyter console --existing``, and run your
code there.

Examples
--------
>>> dtmp = DisplayTemporary.show('running...')
>>> execute_fn(param1, param2)
>>> dtmp.end()

>>> dtmp = dtemp('running...')
>>> execute_fn(param1, param2)
>>> dtmp.end()

>>> with dcontext('running...'):
>>>     execute_fn(param1, param2)

>>> dexpr('running...', lambda: execute_fn(param1, param2))

>>> @dcontext('running...')
>>> def myfunc(param1, param2):
>>>     smthng = do_something(param1, param2)
>>>     return smthng
"""
from typing import ClassVar, Callable, Optional
import io
from contextlib import contextmanager
import sys

assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# %%* Class
# =============================================================================


class _DisplayState():
    """Internal state of a DisplayTemporary"""
    numchar: int
    nest_level: Optional[int]
    name: str

    def __init__(self, prev_state: Optional['_DisplayState'] = None):
        """Construct internal state"""
        self.nest_level = None
        self.numchar = 0
        self.name = "DisplayTemporary({})"
        if prev_state is not None:
            self.numchar = prev_state.numchar
            self.nest_level = prev_state.nest_level

    def format(self, *args, **kwds):
        """Replace field(s) in name"""
        self.name = self.name.format(*args, **kwds)

    def rename(self, name: str):
        """Replace prefix in name"""
        self.name = name + "({})"


class DisplayTemporary():
    """Class for temporarily displaying a message.

    Message erases when `end()` is called, or object is deleted.

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Attributes
    ----------
    output : bool, default : True
        Class attribute. Set it to `False` to suppress display.
    file : Optional[io.TextIOBase], default : None
        Class attribute. Output printed to `file`. If None, use `sys.stdout`.
    debug : bool, default : False
        Class attribute. Set it to `True` to check nesting.

    Class method
    ------------
    show(msg: str) -> DisplayTemporary:
        display `msg` and return class instance (needed to erase message).

    Methods
    -------
    begin(msg: str)
        for initial display of `msg`.
    update(msg: str)
        to erase previous message and display `msg`.
    end()
        to erase display.

    Example
    -------
    >>> dtmp = DisplayTemporary.show('running...')
    >>> execute_fn(param1, param2)
    >>> dtmp.end()
    """
    _state: _DisplayState

    # set output to False to suppress display
    output: ClassVar[bool] = True
    # write output to file. If None, use sys.stdout
    file: ClassVar[Optional[io.TextIOBase]] = None
    # set debug to True to check that displays are properly nested
    debug: ClassVar[bool] = False
    # used for debug
    _nactive: ClassVar[int] = 0

    def __init__(self, **kwds):
        self._state = _DisplayState(**kwds)

    def __del__(self):
        """Clean up, if necessary, upon deletion."""
        if self._state.numchar:
            self.end()

    def begin(self, msg: str = ''):
        """Display message.

        Parameters
        ----------
        msg : str
            message to display
        """
        if self._state.numchar:
            raise AttributeError('begin() called more than once.')
        self._state.format(msg)
        self._state.numchar = len(msg) + 1
        self._print(' ' + msg)
#        self._state['clean'] = False
        if self.debug:
            self._nactive += 1
            self._state.nest_level = self._nactive
            self._check()

    def update(self, msg: str = ''):
        """Erase previous message and display new message.

        Parameters
        ----------
        msg : str
            message to display
        """
        self._bksp(self._state.numchar)
        self._state.numchar = len(msg) + 1
        self._print(' ' + msg)
        if self.debug:
            self._check()

    def end(self):
        """Erase message.
        """
        self._erase(self._state.numchar)
        self._state.numchar = 0
        if self.debug:
            self._nactive -= 1

    def _print(self, text: str):
        """Print with customisations: same line and immediate output

        Parameters
        ----------
        text : str
            string to display
        """
        if self.output:
            print(text, end='', flush=True, file=self.file)

    def _bksp(self, num: int = 1, bkc: str = '\b'):
        """Go back num characters
        """
        if self.file is None:  # self.file.isatty() or self.file is sys.stdout
            pass
        elif self.file.seekable():
            self.file.seek(self.file.tell() - num)
            return

        # hack for jupyter's problem with multiple backspaces
        for i in bkc * num:
            self._print(i)
        # self._print('\b' * num)

    def _erase(self, num: int = 1):
        """Go back num characters
        """
        self._bksp(num)
        self._bksp(num, ' ')
        self._bksp(num)

    def _check(self):
        """Ensure that DisplayTemporaries are properly used
        """
        # raise error if ctr_dsp's are nested incorrectly
        if self._state.nest_level != self._nactive:
            msg1 = 'used at level {} '.format(self._nactive)
            msg2 = 'instead of level {}.'.format(self._state.nest_level)
            raise IndexError(self._state.name + msg1 + msg2)

    def rename(self, name):
        """Change name in debug message"""
        self._state.rename(name)

    @classmethod
    def show(cls, msg: str) -> 'DisplayTemporary':
        """Show message and return class instance.
        Parameters
        ----------
        msg : str
            message to display

        Returns
        -------
        disp_temp : DisplayTemporary
            instance of `DisplayTemporary`. Call `disp_temp.end()` or
            `del disp_temp` to erase displayed message.
        """
        disp_temp = cls()
        disp_temp.begin(msg)
        return disp_temp


# Crappy way of checking if we're running in a qtconsole.
# if 'spyder' in sys.modules:
#     DisplayTemporary.output = False

# =============================================================================
# %%* Functions
# =============================================================================


def dtemp(msg: str = ''):
    """Temporarily display a message.

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Parameters
    ----------
    msg : str
        message to display

    Returns
    -------
    disp_temp : DisplayTemporary
        instance of `DisplayTemporary`. Call `disp_temp.end()` or
        `del disp_temp` to erase displayed message.

    Example
    -------
    >>> dtmp = dtemp('running...')
    >>> execute_fn(param1, param2)
    >>> dtmp.end()
    """
    return DisplayTemporary.show(msg)


@contextmanager
def dcontext(msg: str):
    """Display message during context.

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Prints message before entering context and deletes after.

    Parameters
    ----------
    msg : str
        message to display

    Example
    -------
    >>> with dcontext('running...'):
    >>>     execute_fn(param1, param2)

    >>> @dcontext('running...')
    >>> def myfunc(param1, param2):
    >>>     smthng = do_something(param1, param2)
    >>>     return smthng
    """
    dtmp = DisplayTemporary.show(msg)
    try:
        yield
    finally:
        dtmp.end()


def dexpr(msg: str, lambda_expr: Callable):
    """Display message during lambda execution.

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Prints message before running `lambda_expr` and deletes after.

    Parameters
    ----------
    msg : str
        message to display
    lambda_expr : Callable
        A `lambda` function with no parameters.
        Note that only the `lambda` has no prarmeters. One can pass parameters
        to the function executed in the `lambda`.

    Returns
    -------
    Whatever `lambda_expr` returns.

    Example
    -------
    >>> dexpr('running...', lambda: execute_fn(param1, param2))
    """
    with dcontext(msg):
        out = lambda_expr()
    return out
