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
get_display_options : function
set_display_options : function
display_options : context manager
    Control the behaviour of `DisplayTemporary`: whether/where output is
    displayed and safety checks.
delay_warnings : context manager
    Temporarily stop printing warnings, so they don't appear in the middle of
    `DisplayTemporary` output resulting in the wrong things being deleted.

.. warning:: Does not display properly on ``qtconsole``, ``Spyder``,
``jupyterlab``, ``nteract``.
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
import io
import logging
import sys
from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Dict, Optional, Union

assert sys.version_info[:2] >= (3, 6)


# =============================================================================
# %* Class
# =============================================================================


class _DisplayState():
    """Internal state of a DisplayTemporary"""
    numchar: int
    nest_level: Optional[int]
    name: str
    # used for debug
    nactive: ClassVar[int] = 0

    def __init__(self, prev_state: Optional['_DisplayState'] = None):
        """Construct internal state"""
        self.nest_level = None
        self.numchar = 0
        self.name = "Display({})"
        if prev_state is not None:
            self.numchar = prev_state.numchar
            self.nest_level = prev_state.nest_level

    def begin(self, identifier: str, *args, **kwds):
        """Setup for debugging
        """
        self.name = self.name.format(identifier)
        self.nactive += 1
        self.nest_level = self.nactive
        self.check()

    def check(self, msg=''):
        """Ensure that DisplayTemporaries used in correct order.
        """
        # raise error if ctr_dsp's are nested incorrectly
        if self.nest_level != self.nactive:
            msg += self.name
            msg += f": used at {self.nactive}, nested at {self.nest_level}. "
        if msg:
            raise IndexError(msg)

    def end(self, *args, **kwds):
        """Clean up debugging state
        """
        self.check()
        self.nactive -= 1


class DisplayTemporary():
    """Class for temporarily displaying a message.

    Message erases when `end()` is called, or object is deleted.

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

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
        self._print(' ' + msg)
        self._state.numchar = len(msg) + 1
        if self.debug:
            self._state.begin(msg)

    def update(self, msg: str = ''):
        """Erase previous message and display new message.

        Parameters
        ----------
        msg : str
            message to display
        """
        self._bksp(self._state.numchar)
        self._print(' ' + msg)
        self._state.numchar = len(msg) + 1
        if self.debug:
            self._state.check()

    def end(self):
        """Erase message.
        """
        self._erase(self._state.numchar)
        self._state.numchar = 0
        if self.debug:
            self._state.end()

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
        if self.file is not None and self.file.seekable():
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
        self._print(' ' * num)
        self._bksp(num)

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
# %* Functions
# =============================================================================


def dtemp(msg: str = ''):
    """Temporarily display a message.

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

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

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

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


def dexpr(msg: str, lambda_expr: Callable[[], Any]):
    """Display message during lambda execution.

    .. warning:: Displays improperly in some clients.
    See warning in `display_tricks` module.

    Prints message before running `lambda_expr` and deletes after.

    Parameters
    ----------
    msg : str
        message to display
    lambda_expr : Callable[[], Any]
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


def get_display_options() -> Dict[str, Union[bool, Optional[io.TextIOBase]]]:
    """Current behaviour of DisplayTemporary

    Returns
    -------
    options: dict
        output: bool
            Is the output displayed? default: True
        file: io.TextIOBase or None
            Where the output is displayed, default: None (i.e. sys.stdout)
        debug: bool
            Do we check that they are used in the correct order? default: False
    """
    options = {}
    for opt in ["output", "file", "debug"]:
        options[opt] = getattr(DisplayTemporary, opt)
    return options


def set_display_options(**new_options):
    """Set behaviour of DisplayTemporary

    Parameters
    ----------
    output: bool, optional
        Is the output displayed? default: True
    file: io.TextIOBase, optional
        Where the output is displayed, default: None (i.e. sys.stdout)
    debug: bool, optional
        Do we check that they are used in the correct order? default: False
    """
    old_options = get_display_options()
    for opt in old_options.keys() & new_options.keys():
        setattr(DisplayTemporary, opt, new_options[opt])
    return old_options


@contextmanager
def display_options(**new_options):
    """Control behaviour of DisplayTemporary
    """
    old_options = set_display_options(**new_options)
    try:
        yield
    finally:
        set_display_options(**old_options)


@contextmanager
def delay_warnings(collect=True, print_after=True):
    """Context manager to temporarily suppress warnings

    Parameters
    ----------
    collect : bool, optional, default: True
        When it is `True`, any warnings are collected in a `StringIO` stream.
        The stream can be accessed during the context as `warnlog` in:
            ``with delay_warnings(True) as warnlog:``
    print_after : bool, optional, default: True
        When it is `True`, the accumulated warnings are printed to `sys.stderr`
        when the context manager exits. Has no effect if `collect` is False.
    """
    logging.captureWarnings(True)
    warn_log_stream = None
    if collect:
        warn_log_stream = io.StringIO()
        warn_handler = logging.StreamHandler(warn_log_stream)
        warnlogger = logging.getLogger("py.warnings")
        warnlogger.addHandler(warn_handler)
    try:
        yield warn_log_stream
    finally:
        if collect:
            warn_string = warn_log_stream.getvalue()
            warn_log_stream.close()
            warnlogger.removeHandler(warn_handler)
            if print_after and warn_string:
                print(warn_string, file=sys.stderr)
        logging.captureWarnings(False)
