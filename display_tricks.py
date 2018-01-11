# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:06:58 2018

@author: Subhy
"""
from typing import ClassVar, Dict, Any, Callable
from functools import wraps
import sys

assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# %%* Class
# =============================================================================


class DisplayTemporary(object):
    """Class for temporarily displaying a message.

    Message erases when `end()` is called, or object is erased.

    Class method
    ------------
    show(msg: str) -> DisplayTemporary:
        display `msg` and return class instance (needed to erase message).

    Methods
    -------
    begin(msg: str)
        to initialise counter and display.
    update(msg: str)
        to erase previous message and display `msg`.
    end()
        to erase display.
    """
    _state: Dict[str, Any]

    # set output to False to suppress display
    output: ClassVar[bool] = True
    # set debug to True to check that displays are properly nested
    debug: ClassVar[bool] = False
    _nactive: ClassVar[int] = 0

    def __init__(self):
        self._state = dict(numchar=0)

    def __del__(self):
        """Clean up, if necessary"""
        if self._state['numchar']:
            self.end()

    def begin(self, msg: str = ''):
        """Display message.

        Parameters
        ----------
        msg : str
        """
        if self._state['numchar']:
            raise AttributeError('begin() called more than once.')
        self._state['numchar'] = len(msg) + 1
        self._print(' ' + msg)
        self._state['clean'] = False
        if self.debug:
            self._nactive += 1
            self._state['nest_level'] = self._nactive
            self._check()

    def update(self, msg: str = ''):
        """Erase previous message and display new message."""
#        self._print('\b \b' * self._state['numchar'])
        # hack for jupyter's problem with multiple backspaces
        for i in '\b' * self._state['numchar']:
            self._print(i)
        self._state['numchar'] = len(msg) + 1
        self._print(' ' + msg)
        if self.debug:
            self._check()

    def end(self):
        """Erase message."""
#        self._print('\b \b' * self._state['numchar'])
        # hack for jupyter's problem with multiple backspaces
        for i in '\b \b' * self._state['numchar']:
            self._print(i)
        self._state['numchar'] = 0
        if self.debug:
            self._nactive -= 1

    def _print(self, text: str):
        """Print with customisations: same line and immediate output"""
        if self.output:
            print(text, end='', flush=True)

    def _check(self):
        """Ensure that DisplayTemporaries are properly used
        """
        # raise error if ctr_dsp's are nested incorrectly
        if self._state['nest_level'] != self._nactive:
            msg1 = 'DisplayCount{}'.format(self._prefix)
            msg2 = 'used at level {} '.format(self._nactive)
            msg3 = 'instead of level {}.'.format(self._state['nest_level'])
            raise IndexError(msg1 + msg2 + msg3)

    @classmethod
    def show(cls, msg: str):
        """Show message and return class instance.
        """
        obj = cls()
        obj.begin(msg)
        return obj


def dtemp(msg: str = ''):
    """Temporarily display a message
    """
    return DisplayTemporary.show(msg)


def dexpr(msg: str, lambda_expr: Callable):
    """Print message during lambda execution.

    Prints message before running `lambda_expr` and deletes after.

    Parameters
    ----------
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
    d = dtemp(msg)
    out = lambda_expr()
    d.end()
    return out


def dwrap(msg: str):
    """Decorate a function with a temporary printed message.

    Prints message before running `func` and deletes after.


    Parameters
    ----------
    msg : str
        the message to display during function execution

    Returns
    -------
    decorator
        decorator that wraps a function, to displae `msg` during execution.

    Example
    -------
    >>> @dwrap('running...')
    >>> def myfunc(param1, param2):
    >>>     smthng = do_something(param1, param2)
    >>>     return smthng
    """
    def decorator(func):
        """Decorate a function with a temporary printed message.

        Prints message before running `func` and deletes after.

        Parameters
        ----------
        func
            the function you want to time

        Returns
        -------
        timed_func
            wrapped version of `func`, with same paramaters and returns.

        Example
        -------
        >>> decorator = dwrap('running...')
        >>> @decorator
        >>> def myfunc(param1, param2):
        >>>     smthng = do_something(param1, param2)
        >>>     return smthng
        """
        @wraps(func)
        def dfunc(*args, **kwds):
            """Wrapped function"""
            return dexpr(msg, lambda: func(*args, **kwds))
        return dfunc
    return decorator
