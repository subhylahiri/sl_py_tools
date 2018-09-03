# -*- coding: utf-8 -*-
# =============================================================================
# Created on Fri Dec  1 15:09:52 2017
#
# @author: Subhy
#
# Module: import_tricks
# =============================================================================
"""
IPython's deep reload with custom excluded modules and packages.

I recommend that you exclude any modules/packages from any libraries that
you import and did not write yourself, or are not currently editing.

The defaults are defined in the `Reloader` class ``__init__``.
They can be viewed in `reload.excluded_mods` and `reload.excluded_pkgs`.

Replacing the built in `reload`, as described in `IPython.lib.deepreload`, may
not work with this `reload` as it is a callable object rather than a function.

Example
=======
>>> import my_module
>>> from import_tricks import reload
>>> reload.exclude_mod('std_module')
>>> reload.exclude_pkg('std_package')
>>> reload(my_module)
"""
__all__ = ['Reloader', 'reload']

# frippery
from collections.abc import Callable
from typing import Tuple, Set
# essential
import sys
import os
from functools import wraps
from contextlib import contextmanager

from IPython.lib import deepreload

ORIGINAL_IMPORT_SUBMODULE = deepreload.import_submodule

# =============================================================================
# %%* Reloader class
# =============================================================================


class Reloader(Callable):
    """Customised deep reload for IPython

    This callable class stores an extended set of modules to exclude from
    deep reloading:
        typing, collections, collections.abc, types, numbers,
        functools, itertools, pkgutil.
    They are added to the default option of `IPython.lib.deepreload.reload`.
    It also excludes some packages, all modules in those packages are excluded:
        matplotlib.*, numpy.*, scipy.*, importlib.*, IPython.*.

    Parameters
    ==========
    also_exclude_mods : Tuple[str, ...] = ()
        Tuple of module names to add to the excluded list.
    also_exclude_pkgs : Tuple[str, ...] = ()
        Tuple of package names to add to the excluded list.
    exclude_std : bool = True
        Exclude all packages from the Python standard library?

    Methods
    =======
    exclude_mod('module1', 'module2', ...)
        Append 'module1', 'module2', ... to the set of modules to exclude from
        deep reloading.
    exclude_pkg('package1', 'package2', ...)
        Append 'package1', 'package2', ... to the set of packages to exclude
        from deep reloading.
    (module) : as it is a callable class.
        Deep reload `module` with the above excluded.

    Example
    =======
    >>> import my_module
    >>> import import_tricks
    >>> reload = import_tricks.Reloader(('std_module',), ('std_package',))
    >>> reload.exclude_mod('other_std_module','another_std_module')
    >>> reload.exclude_pkg('other_std_package','another_std_package')
    >>> reload(my_module)
    """
    def __init__(self,
                 also_exclude_mods: Tuple[str, ...] = (),
                 also_exclude_pkgs: Tuple[str, ...] = (),
                 exclude_std: bool = True):
        self.exclude_std = exclude_std

        self.excluded_mods = also_exclude_mods
        self.exclude_mod(*deepreload.reload.__defaults__[0])
#        self.exclude_mod('typing', 'collections', 'collections.abc', 'abc')
#        self.exclude_mod('types', 'numbers', 'io', 'contextlib', 'datetime')
#        self.exclude_mod('itertools', 'functools', 'pkgutil')

        self.excluded_pkgs = set(also_exclude_pkgs)
        self.exclude_pkg('numpy', 'matplotlib', 'scipy', 'IPython', 'numba',
                         'llvmlite')
#        self.exclude_pkg('importlib')

    def __call__(self, module, *args, **kwd):
        """Deep reload `module` with specified modules and packages excluded.
        """
        if self.exclude_std:
            context = replace_import_submodule_hook_std(self.excluded_pkgs)
        else:
            context = replace_import_submodule_hook(self.excluded_pkgs)

        with context:
            deepreload.reload(module, *args, exclude=self.excluded_mods, **kwd)

    def exclude_mod(self, *modules):
        """Add more modules to excluded set.

        Parameters
        ==========
        module1, module2, ... : str
            module names
        """
        self.excluded_mods += modules

    def exclude_pkg(self, *packages):
        """Add more packages to excluded set.

        Parameters
        ==========
        package1, package2, ... : str
            package names
        """
        self.excluded_pkgs.update(packages)


# Use instance to do your reloads.
reload = Reloader()

# =============================================================================
# %%* Helper functions for Reloader
# =============================================================================


def parent_names(fullname: str) -> Set[str]:
    """List parent packages
    """
    parents = set()
    dotindex = len(fullname)
    while dotindex > 0:
        parents.add(fullname[:dotindex])
        dotindex = fullname[:dotindex].rfind('.')
    return parents


def package_check(fullname: str, packages: Set[str]) -> bool:
    """Is it a submodule of any of the packages?
    """
    return not packages.isdisjoint(parent_names(fullname))


def import_submodule_wrapper(excluded_pkgs):
    """Wrapper for deepreload.import_submodule that excludes entire packages.
    """
    @wraps(ORIGINAL_IMPORT_SUBMODULE)
    def my_import_submodule(mod, subname, fullname):
        if package_check(fullname, excluded_pkgs) and fullname in sys.modules:
            deepreload.found_now[fullname] = 1
            return sys.modules[fullname]
        return ORIGINAL_IMPORT_SUBMODULE(mod, subname, fullname)
    return my_import_submodule


@contextmanager
def replace_import_submodule_hook(excluded_pkgs):
    """Temporarily replace import__submodule__ in IPython.lib.deepreload.

    Our version excludes some entire packages.
    """
    saved_import_submodule = deepreload.import_submodule
    deepreload.import_submodule = import_submodule_wrapper(excluded_pkgs)
    try:
        yield
    finally:
        deepreload.import_submodule = saved_import_submodule


# =============================================================================
# %%* ALso excluding stdlib packages
# =============================================================================

# from https://stackoverflow.com/questions/22195382/how-to-check-if-a-module-
# library-package-is-part-of-the-python-standard-library
# by: https://stackoverflow.com/users/100297/martijn-pieters

# paths for stdlib
_paths = (os.path.abspath(p) for p in sys.path)
stdlib = tuple(p for p in _paths
               if p.startswith((sys.prefix, sys.exec_prefix,
                                sys.base_exec_prefix, sys.base_prefix))
               and 'site-packages' not in p)


def stdlib_check(module):
    """Check if a module is from stdlib.

    from https://stackoverflow.com/questions/22195382/how-to-check-if-a-module-
    library-package-is-part-of-the-python-standard-library
    by: https://stackoverflow.com/users/100297/martijn-pieters
    """
    if (not hasattr(module, '__name__')
            or module.__name__ in sys.builtin_module_names
            or not hasattr(module, '__file__')):
        # an import sentinel, built-in module or not a real module, really
        return True

    fname = module.__file__
    if 'site-packages' in fname:
        return False

    if fname.endswith(('__init__.py', '__init__.pyc', '__init__.pyo')):
        fname = os.path.dirname(fname)

    if os.path.dirname(fname).startswith(stdlib):
        # stdlib path, skip
        return True
    return False


def import_submodule_wrapper_std(excluded_pkgs):
    """Wrapper for deepreload.import_submodule that excludes entire packages
    and anything in stdlib.
    """
    @wraps(ORIGINAL_IMPORT_SUBMODULE)
    def my_import_submodule(mod, subname, fullname):
        if ((package_check(fullname, excluded_pkgs) or stdlib_check(mod))
                and fullname in sys.modules):
            deepreload.found_now[fullname] = 1
            return sys.modules[fullname]
        return ORIGINAL_IMPORT_SUBMODULE(mod, subname, fullname)
    return my_import_submodule


@contextmanager
def replace_import_submodule_hook_std(excluded_pkgs):
    """Temporarily replace import__submodule__ in IPython.lib.deepreload.

    Our version excludes some entire packages.
    """
    saved_import_submodule = deepreload.import_submodule
    deepreload.import_submodule = import_submodule_wrapper_std(excluded_pkgs)
    try:
        yield
    finally:
        deepreload.import_submodule = saved_import_submodule
