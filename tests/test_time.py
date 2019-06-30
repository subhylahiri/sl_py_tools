# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:37:12 2018

@author: Subhy
"""
import time
from sl_py_tools.time_tricks import Timer, time_with, time_expr

dt = Timer.now()
time.sleep(42)
dt.time()
time.sleep(64)
dt.time()

with time_with():
    time.sleep(45)

time_expr(lambda: time.sleep(73))


@time_with()
def myfunc(param1, param2):
    time.sleep(param1)
    smthng = str(param2)
    return smthng


myfunc(23, 37)
