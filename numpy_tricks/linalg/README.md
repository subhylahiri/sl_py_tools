# Numpy linear algebra helpers

This package contains classes and functions that make the syntax for linear
algebra in `numpy` cleaner, particularly with respect to broadcasting.
The main way of using this is via the `lnarray` class (the `qr` function is the
only other thing I find useful here).

The `lnarray` class has properties `t` for transposing, `r` for row vectors,
`c` for column vectors and `s` for scalars in a way that fits with `numpy.linalg`
broadcasting rules (`t` only transposes the last two indices, `r,c,s` add
singleton axes so that linear algebra routines treat them as arrays of
vectors/scalars rather than matrices, and `uc,ur,us` undo the effects of `r,c,s`).

The `lnarray` class also has properties for delayed matrix division:
```python
>>> z = x.inv @ y
>>> z = x @ y.inv
>>> z = x.pinv @ y
>>> z = x @ y.inv
```
None of the above actually invert the matrices. They return `invarray/pinvarray`
objects that call `solve/lstsq` behind the scenes, which is [faster and more
accurate](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/). To get the actual inverse matrices:
```python
>>> x = y.inv()
>>> x = y.pinv()
```

## Classes

* `lnarray`:  
    Subclass of `numpy.ndarray` with properties such as `pinv/inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
* `pinvarray`:  
    Provides interface for matrix division when it is matrix multiplied (@).
    Returned by `lnarray.pinv`. It calls `np.linalg.lstsq` behind the scenes.
    Does not actually pseudoinvert the matrix unless it is explicitly called.
    I think it is best not to store these objects in variables, and call on
    `lnarray.pinv` on the rhs instead.
* `invarray`:  
    Provides interface for matrix division when it is matrix multiplied (@).
    Returned by `lnarray.inv`. It calls `np.linalg.solve` behind the scenes.
    Does not actually invert the matrix unless it is explicitly called.
    I think it is best not to store these objects in variables, and call on
    `lnarray.inv` on the rhs instead.
* `lnmatrix`:  
    Subclass of `lnarray` which swaps matrix/elementwise multiplication and
    division from the right. Shouldn't be necessary given `lnarray`'s syntax.
* `ldarray`:  
    Subclass of `lnarray` which overloads bitshift operators for matrix division.
    One of several reasons why this is a bad idea is that bitshifting has lower
    operator priority than division, so you will have to use parentheses often.
    I think you're better off sticking with `lnarray`.

## Functions

* `matmul`:  
    Broadcasting, BLAS accelerated matrix multiplication.
* `solve`:  
    Broadcasting, Lapack accelerated, linear equation solving (matrix
    left-division).
* `rsolve`:  
    Reversed, broadcasting, Lapack accelerated, linear equation solving (matrix
    right-division).
* `lstsq`:  
    Broadcasting, Lapack accelerated, linear least squares problems (matrix
    left-division). Unlike `numnpy.linalg.lstsq`, this does not take an `rcond`
    parameter, or return diagnostic information (which is better suited to binary
    operators). However, it does broadcast and pass through subclasses.
* `rlstsq`:  
    Reversed, broadcasting, Lapack accelerated, linear least squares problems
    (matrix right-division).
* `norm`
    Vector 2-norm. Broadcasts and passes through subclasses.
* `transpose`:  
    Transpose last two indices.  
* `col`:  
    Treat multi-dim array as a stack of column vectors.
* `row`:  
    Treat multi-dim array as a stack of row vectors.
* `scal`:  
    Treat multi-dim array as a stack of scalars.
* `matldiv`:  
    Matrix division from left (exact or least-squares).
* `matrdiv`:  
    Matrix division from right (exact or least-squares).
* `qr`:  
    QR decomposition with broadcasting and subclass passing. Does not implement the deprecated modes of `numpy.linalg.qr`.

## GUfuncs
These implement the functions above.
* `gufuncs.matmul`:  
* `gufuncs.solve`:  
* `gufuncs.rsolve`:  
* `gufuncs.lstsq`:  
* `gufuncs.rlstsq`:  
* `gufuncs.norm`:  
    These are literally the same as the functions above.
* `gufuncs.qr_m`:  
    Implements `qr` for wide matrices in `reduced` mode, and all matrices in
    `complete` mode.
* `gufuncs.qr_n`:  
    Implements `qr` for narrow matrices in `reduced` mode.
* `gufuncs.qr_rm`:  
* `gufuncs.qr_rn`:  
    Implement `qr` in `r` mode.
* `gufuncs.qr_rawm`:  
* `gufuncs.qr_rawn`:  
    Implement `qr` in `raw` mode.
* `gufuncs.rmatmul`
* `gufuncs.rtrue_tivide`:  
    Reversed versions of `matmul` and `np.true_divide`. Used by `pinvarray` and
    `invarray`.

## Wrappers
* `wrappers.wrap_one`:  
    Create version of `numpy` function with single `lnarray` output.
* `wrappers.wrap_several`:  
    Create version of `numpy` function with multiple `lnarray` outputs.
* `wrappers.wrap_some`:  
    Create version of `numpy` function with some `lnarray` outputs, some
    non-array outputs.
* `wrappers.wrap_sub`:  
    Create version of `numpy` function with single `lnarray` output, passing
    through subclasses.
* `wrappers.wrap_subseveral`:  
    Create version of `numpy` function with multiple `lnarray` outputs, passing
    through subclasses.
* `wrappers.wrap_subsome`:  
    Create version of `numpy` function with some `lnarray` outputs, some
    non-array outputs, passing through subclasses.

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


## Building the CPython modules

You will need to have the appropriate C compilers. On Linux, you should already have them.
On Windows, [see here](https://wiki.python.org/moin/WindowsCompilers).

You will need a BBLAS/Lapack distribution. Anaconda usually uses MKL, but they
recently moved the headers to a different package. You can find them on
[Intel's anaconda channel](https://software.intel.com/en-us/articles/using-intel-distribution-for-python-with-anaconda):
```
> conda install mkl -c intel --no-update-deps
```
Or you can downgrade to a version that has the headers, e.g.
```
> conda install mkl=2018.0.3
```
Another option is [OpenBLAS](https://www.openblas.net/)
```
> conda install openblas -c conda-forge
```
([see here](https://docs.continuum.io/mkl-optimizations/#uninstalling-mkl) under
Uninstalling MKL).

If your BLAS/Lapack distribution is installed somewhere numpy isn't expecting,
you can provide directions in a [site.cfg file](https://github.com/numpy/numpy/blob/master/site.cfg.example).

Once you have all of the above, you can build the CPython modules in-place:
```
> python setup.py build_ext
```
or you can install it system-wide:
```
> python setup.py install
```
