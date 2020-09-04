# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 20:06:44 2017

@author: subhy
"""
import time
import sys
import numpy as np
from sl_py_tools.iter_tricks import dcount, denumerate, dzip, zenumerate
from sl_py_tools.iter_tricks import batch, dbatch, rdenumerate, rdzip
from sl_py_tools.display_tricks import DisplayTemporary, dcontext, dexpr
from sl_py_tools.numpy_tricks.iter import dndindex
# pylint: disable=all
# =============================================================================
# Code running functions
# =============================================================================


def test_dndindex():
    for x in dndindex('a', 2, 3, 4):
        time.sleep(0.2)


def test_reversed():
    for i in reversed(denumerate('a', [1, 2, 3, 4, 5])):
        time.sleep(1)
    for i in reversed(dcount('b', 2, 9, 3)):
        time.sleep(1)


def test_nesting():
    for i in dcount('i', 5):
        for j in dcount('j', 6):
            for k in dcount('k', 4, 10):
                time.sleep(.1)
    print('done')


def test_zip():
    for i in dcount('i', 5):
        for j, k in zip(dcount('j', 8), [1, 7, 13]):
            time.sleep(.2)
        time.sleep(.2)
    print('done')


@dcontext('decorating denumerate')
def test_denumerate():
    for i in dcount('i', 5):
        for j, k in denumerate('j', [1, 7, 13]):
            time.sleep(.2)
        time.sleep(.2)
    # print('done')


def test_denumerate_zip():
    words = [''] * 4
    letters = 'xyz'
    counts = [5, 7, 13]
    for idx, key, num in denumerate('idx', letters, counts):
        words[idx] = key * num
        time.sleep(1)
    print(words)


def test_zenumerate():
    words = [''] * 4
    letters = 'xyz'
    counts = [5, 7, 13]
    for idx, key, num in zenumerate(letters, counts):
        words[idx] = key * num
    print(words)


def test_dzip():
    keys = 'xyz'
    values = [1, 7, 13]
    assoc = {}
    for key, val in dzip('idx', keys, values):
        assoc[key] = val
        time.sleep(1)
    print(assoc)


def test_rdenumerate():
    words = [''] * 4
    letters = 'xyz'
    counts = [5, 7, 13]
    for idx, key, num in rdenumerate('idx', letters, counts):
        words[idx] = key * num
        time.sleep(1)
    print(words)


def test_rdzip():
    keys = 'xyz'
    values = [1, 7, 13]
    assoc = {}
    for key, val in rdzip('idx', keys, values):
        assoc[key] = val
        time.sleep(1)
    print(assoc)


def test_batch():
    x = np.random.rand(1000, 3, 3)
    y = np.empty((1000, 3), dtype=complex)
    for s in batch(0, len(x), 10):
        y[s] = np.linalg.eigvals(x[s])
    print(x[15], '\n', y[15])


def test_dbatch():
    x = np.random.rand(1000, 3, 3)
    y = np.empty((1000, 3), dtype=complex)
    for s in dbatch('s', 0, len(x), 10):
        y[s] = np.linalg.eigvals(x[s])
        time.sleep(.1)
    print(x[15], '\n', y[15])

# =============================================================================
# Running code
# =============================================================================


if __name__ == "__main__":
    # DisplayTemporary.file = None
    # DisplayTemporary.file = open('test_iter.txt', 'w')
    DisplayTemporary.file = sys.stderr

    test_zip()
    with dcontext('reversed in context'):
        test_reversed()
    test_nesting()
    test_denumerate()
    test_zenumerate()
    test_denumerate_zip()
    test_dzip()
    test_rdenumerate()
    test_rdzip()
    test_batch()
    test_dbatch()
    dexpr('dndindex in lambda', test_dndindex)

    #    DisplayTemporary.file.close()
    DisplayTemporary.file = None
