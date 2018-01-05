# -*- coding: utf-8 -*-
# =============================================================================
# Created on Fri Sep  1 15:34:56 2017
#
# @author: Subhy
#
# Module: `sl_datetime`.
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

Classes
=======
timeit:
    Class for displaying before/after time.

Examples
========
>>> d1 = datetime.datetime.now()
>>> print(dt_format(d1))
>>> time.sleep(100)
>>> d2 = datetime.datetime.now()
>>> print(td_format(d2 - d1, True))

>>> dt timeit.now()
>>> time.sleep(1000)
>>> dt.time()
>>> time.sleep(100)
>>> dt.time()
"""

import datetime
from typing import Union, Optional

DateTime = Union[datetime.date, datetime.time]


def dt_format(d: DateTime) -> str:
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
    if isinstance(d, datetime.date):
        out_str += "{0:%a}, {0.day} {0:%b} {0.year}".format(d)
    if isinstance(d, datetime.datetime):
        out_str += ", "
    if isinstance(d, (datetime.time, datetime.datetime)):
        out_str += "{1}:{0.minute:02}:{0.second:02} {2}".format(d, *ampm_hr(d))
    return out_str


def td_format(d: datetime.timedelta, subsec: bool=False) -> str:
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
    if d.days < 0:
        out_str = "-"
        cmpts = (d.days + 1, 86399 - d.seconds, 10**6 - d.microseconds)
    else:
        out_str = ""
        cmpts = (d.days, d.seconds, d.microseconds)

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


def ampm_hr(d: datetime.time) -> (int, str):
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
    return d.hour % 12, ('AM', 'PM')[d.hour // 12]


class timeit(object):
    """Class for displaying before/after time.

    Parameters
    ==========
    begin : Optional[datetime.datetime]=None
        `datetime` object for initial time.

    Examples
    ========
    >>> dt = timeit()
    >>> import time
    >>> dt.start()
    >>> time.sleep(1000)
    >>> dt.time()

    >>> dt timeit.now()
    >>> time.sleep(1000)
    >>> dt.time()
    >>> time.sleep(100)
    >>> dt.time()
    """
    begin: datetime.datetime

    def __init__(self, begin: Optional[datetime.datetime]=None):
        self.begin = begin

    def start(self, *args, **kwargs):
        """Call this before thing you are timing.

        Prints and stores current time.

        Parameters
        ==========
        tz : Optional[datetime.tzinfo] = None
            Time zone to use. Passed to `datetime.datetime.now`.
        """
        self.begin = datetime.datetime.now(*args, **kwargs)
        print(dt_format(self.begin))

    def time(self, *args, **kwargs):
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
        print(dt_format(end))
        print("That took: " + td_format(end - self.begin, *args, **kwargs))
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
        """
        obj = cls()
        obj.start(*args, **kwargs)
        return obj
