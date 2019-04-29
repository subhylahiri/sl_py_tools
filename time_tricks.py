# -*- coding: utf-8 -*-
# =============================================================================
# Created on Fri Sep  1 15:34:56 2017
#
# @author: Subhy
#
# Module: `time_tricks`.
# =============================================================================
"""
Produce formatted strings for displaying objects from datetime module.

Functions
=========
dt_format(d: Union[datetime.date, datetime.time]) -> str:
    Formatted string for date/time/datetime objects
    eg. 'Fri, 1 Sep 2017, 3:34:56 PM'
td_format(d: datetime.timedelta, subsec: bool=False) -> str:
    Formatted string for timedelta objects,
    e.g. '10w, 5d, 4h, 39m, 50s'
ampm_hr(d: datetime.time) -> (int, str):
    converts 24 hr clock to 12 hr.
time_with
    Prints date & time before & after context, and elapsed time.
time_expr(lambda_expr):
    Time a lambda expression.
time_wrap(func):
    Decorate a function with a timer.

Classes
=======
Timer:
    Class for displaying before/after time.

Examples
========
>>> import time
>>> d1 = datetime.datetime.now()
>>> print(dt_format(d1))
>>> time.sleep(100)
>>> d2 = datetime.datetime.now()
>>> print(td_format(d2 - d1, True))

>>> dt Timer.now()
>>> time.sleep(1000)
>>> dt.time()
>>> time.sleep(100)
>>> dt.time()

>>> with time_with():
>>>     time.sleep(100)

>>> time_expr(lambda: time.sleep(100))

>>> @time_with()
>>> def myfunc(param1, param2):
>>>     time.sleep(param1)
>>>     smthng = do_something(param2)
>>>     return smthng
>>>
>>> myfunc(20, 30)
"""

import datetime
from contextlib import contextmanager
from typing import Union, Optional, Callable, ClassVar
import io
import sys

assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# %%* Formatting functions
# =============================================================================


def dt_format(dtm: Union[datetime.date, datetime.time]) -> str:
    """
    Formatted string for date/time objects,
    eg. 'Fri, 1 Sep 2017, 3:34:56 PM'

    Parameters
    ==========
    d
        datetime.date/time object

    Returns
    =======
    out_str
        string representation of date/time

    Examples
    ========
    >>> d1 = datetime.datetime.now()
    >>> print(dt_format(d1))
    """
    out_str = ""
    if isinstance(dtm, datetime.date):
        out_str += "{0:%a}, {0.day} {0:%b} {0.year}".format(dtm)
    if isinstance(dtm, datetime.datetime):
        out_str += ", "
    if isinstance(dtm, (datetime.time, datetime.datetime)):
        out_str += "{1}:{0.minute:02}:{0.second:02} {2}".format(dtm,
                                                                *ampm_hr(dtm))
    return out_str


def td_format(tdl: datetime.timedelta, subsec: bool = False) -> str:
    """
    Formatted string for timedelta objects,
    e.g. '10w, 5d, 4h, 39m, 50s'

    Parameters
    ==========
    d
        datetime.timedelta object
    subsec
        include subsecond info? Default=False.

    Returns
    =======
    out_str
        string representation of date/time

    Examples
    ========
    >>> d1 = datetime.datetime.now()
    ... some stuff
    >>> d2 = datetime.datetime.now()
    >>> print(td_format(d2 - d1), True)
    """
    if tdl.days < 0:
        out_str = "-"
        cmpts = (tdl.days + 1, 86399 - tdl.seconds, 10**6 - tdl.microseconds)
    else:
        out_str = ""
        cmpts = (tdl.days, tdl.seconds, tdl.microseconds)

    unitss = (["y", "w", "d"], ["h", "m", "s"], ["ms", "us"])
    basess = ([365, 7, 1], [3600, 60, 1])
    if subsec:
        basess += ([1000, 1],)

    for cmpt, bases, units in zip(cmpts, basess, unitss):
        for base, unit in zip(bases, units):
            if cmpt >= base:
                out_str += str(cmpt // base) + unit + ", "
                cmpt %= base

    return out_str[:-2]


def ampm_hr(dtm: datetime.time) -> (int, str):
    """
    Convert 24 hour clock to 12 hour.

    Parameters
    ==========
    d
        datetime.time object.

    Returns
    =======
    new_hour
        d.hour mod 12
    ampm
        "AM" if d.hour < 12, or "PM" otherwise.
    """
    return dtm.hour % 12, ('AM', 'PM')[dtm.hour // 12]

# =============================================================================
# %%* Timer class
# =============================================================================


class Timer(object):
    """Class for displaying before/after time.

    Parameters
    ==========
    begin : Optional[datetime.datetime]=None
        `datetime` object for initial time.

    Examples
    ========
    >>> import time
    >>> dt = Timer()
    >>> dt.start()
    >>> time.sleep(1000)
    >>> dt.time()

    >>> dt = Timer.now()
    >>> time.sleep(1000)
    >>> dt.time()
    >>> time.sleep(100)
    >>> dt.time()
    """
    begin: datetime.datetime
    absolute: bool

    # write output to file. If None, use sys.stdout
    file: ClassVar[Optional[io.TextIOBase]] = None

    def __init__(self, begin: Optional[datetime.datetime] = None):
        self.begin = begin
        self.absolute = True

    def start(self, *args, absolute=True, **kwargs):
        """Call this before thing you are timing.

        Prints and stores current time.

        Parameters
        ==========
        tz : Optional[datetime.tzinfo] = None
            Time zone to use. Passed to `datetime.datetime.now`.
        absolute: bool, optional, default=True
            Print current time before & after (duration is printed either way).
        """
        self.absolute = absolute
        self.begin = datetime.datetime.now(*args, **kwargs)
        if self.absolute:
            print(dt_format(self.begin), file=self.file, flush=True)

    def time(self, subsec: bool = False, *args, **kwargs):
        """Call this after thing you are timing.

        Prints and stores current time.
        Prints time elapsed since previous call to `start` ot `time`.

        Parameters
        ==========
        subsec : bool = False
            include subsecond info? Passed to `td_format`.
        """
        if self.begin is None:
            self.start(*args, **kwargs)
            return
        end = datetime.datetime.now(self.begin.tzinfo)
        if self.absolute:
            print(dt_format(end), file=self.file, flush=True)
        print("That took: " + td_format(end - self.begin, subsec=subsec),
              file=self.file, flush=True)
        self.begin = end

    @classmethod
    def now(cls, *args, **kwargs):
        """Create and start factory method.

        Automatically calls `timeit.start` after creation, so that it prints
        and stores current time.

        Parameters
        ==========
        tz : Optional[datetime.tzinfo] = None
            Time zone to use. Passed to `timeit.start` which passes it to
            `datetime.datetime.now`.
        absolute: bool, optional, default=True
            Print current time before & after (duration is printed either way).
        """
        obj = cls()
        obj.start(*args, **kwargs)
        return obj

# =============================================================================
# %%* Wrappers
# =============================================================================


@contextmanager
def time_with(subsec: bool = False, *args, **kwargs):
    """Time a context, or decorate a function with a timer

    Prints date & time before & after context, and elapsed time.
    Can also be used as a function decorator.

    Parameters
    ----------
    subsec: bool, optional, default=False
        Print sub-second timing.
    absolute: bool, optional, default=True
        Print current time before & after (duration is printed either way).

    Examples
    --------
    >>> with time_with():
    >>>     execute_fn(param1, param2)

    >>> @time_with()
    >>> def myfunc(param1, param2):
    >>>     smthng = do_something(param1, param2)
    >>>     return smthng
    """
    dtmp = Timer.now(*args, **kwargs)
    try:
        yield
    finally:
        dtmp.time(subsec=subsec)


def time_expr(lambda_expr: Callable, subsec: bool = False, *args, **kwargs):
    """Time a lambda expression.

    Prints date & time before & after running `lambda_expr` and elapsed time.

    Parameters
    ----------
    lambda_expr : Callable
        A `lambda` function with no parameters.
        Note that only the `lambda` has no prarmeters. One can pass parameters
        to the function executed in the `lambda`.
    subsec: bool, optional, default=False
        Print sub-second timing.
    absolute: bool, optional, default=True
        Print current time before & after (duration is printed either way).

    Returns
    -------
    Whatever `lambda_expr` returns.

    Example
    -------
    >>> time_expr(lambda: execute_fn(param1, param2))
    """
    with time_with(subsec, *args, **kwargs):
        out = lambda_expr()
    return out
