# Numpy linear algebra helpers

This has been moved to [its own repo](https://github.com/subhylahiri/numpy_linalg_extras).
Therefore, this subpackage is no longer being updated. It does contain some
abandoned approaches that were not moved to the new repo.

This package contains classes and functions that make the syntax for linear
algebra in `numpy` cleaner, particularly with respect to broadcasting and
matrix division. The main way of using this is via the `lnarray` class
(the `qr` function is the only other thing I find useful here). All of the
functions will work with `numpy.ndarray` objects as well.

The `lnarray` class has properties `t` for transposing, `h` for
conjugate-transposing, `r` for row vectors, `c` for column vectors and `s` for
scalars in a way that fits with `numpy.linalg` broadcasting rules (`t` only
transposes the last two indices, `r,c,s` add singleton axes so that linear
algebra routines treat them as arrays of vectors/scalars rather than matrices,
and `uc,ur,us` undo the effects of `r,c,s`).

The `lnarray` class also has properties for delayed matrix division:
```python
>>> z = x.inv @ y
>>> z = x @ y.inv
>>> z = x.pinv @ y
>>> z = x @ y.pinv
```
None of the above actually invert the matrices. They return `invarray/pinvarray`
objects that call `solve/lstsq` behind the scenes, which is [faster and more
accurate](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/).
To get the actual inverse matrices you can call the objects:
```python
>>> x = y.inv()
>>> x = y.pinv()
```

## Classes

* `lnarray`:  
    Subclass of `numpy.ndarray` with properties such as `pinv/inv` for matrix
    division, `t` and `h` for transposing stacks of matrices, `c`, `r` and `s`
    for dealing with stacks of vectors and scalars.
* `lnmatrix`:  
    Subclass of `lnarray` which swaps matrix/elementwise multiplication and
    division from the right. Shouldn't be necessary given `lnarray`'s syntax.
* `ldarray`:  
    Subclass of `lnarray` which overloads bit-shift operators for matrix division.
    One of several reasons why this is a bad idea is that bit-shifting has lower
    operator priority than division, so you will have to use parentheses often.
    I think you're better off sticking with `lnarray`.

Examples
--------
```python
>>> import numpy as np
>>> import linalg as sp
>>> x = sp.lnarray(np.random.rand(2, 3, 4))
>>> y = sp.lnarray(np.random.rand(2, 3, 4))
>>> z = x.pinv @ y
>>> w = x @ y.pinv
>>> u = x @ y.t
>>> v = (x.r @ y[:, None, ...].t).ur
>>> a = sp.ldarray(np.random.rand(2, 3, 4))
>>> b = sp.ldarray(np.random.rand(2, 3, 4))
>>> c = (a << b)
>>> d = (a >> b)
```
