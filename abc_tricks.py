# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:29:11 2017

@author: subhy

Functions to help define ABCs (abstract base classes) from a template.
"""
__all__ = [
    'typename', 'ABCauto',
    'get_abstracts', 'subclass_hook', 'subclass_hook_nosub',
    'check_methods', 'check_attributes', 'check_properties',
]

import abc
import types
from typing import Union, List, Tuple, Callable, Any

CheckResult = Union[bool, type(NotImplemented)]
Checker = Callable[[type, str], CheckResult]
PropTypes = (property, types.MemberDescriptorType)


def typename(inst: Any) -> str:
    """String of name of type"""
    return type(inst).__name__


def supername(cls: type, base: type = object) -> str:
    """String of name of superclass

    Searches for first subclass of `base` in `cls.__mro__` other than `cls`.
    """
    for scls in cls.__mro__:
        if scls is not cls and issubclass(scls, base):
            break
    return scls.__name__

# =============================================================================
# Type check utilities
# =============================================================================


def _check_dict(B: type, method: str) -> CheckResult:
    """Check if method is in class dictionary.
    """
    if method in B.__dict__:
        if B.__dict__[method] is None:
            return NotImplemented
        return True
    return False


def _check_annotations(B: type, prop: str) -> CheckResult:
    """Check if attribute is in class annotations.
    """
    return prop in getattr(B, '__annotations__', {})


def _check_property(B: type, prop: str) -> CheckResult:
    """Check if prop is in class dictionary (as a property) or annotations.
    """
    ok = _check_dict(B, prop)
    if ok is NotImplemented:
        return NotImplemented
    if ok:
        return isinstance(B.__dict__[prop], PropTypes)
    return _check_annotations(B, prop)


def _check_generic(C: type, check: Checker, *methods: str) -> CheckResult:
    """Check class for methods
    """
    mro = C.__mro__
    for method in methods:
        for B in mro:
            ok = check(B, method)
            if ok is NotImplemented:
                return NotImplemented
            if ok:
                break
        else:
            return NotImplemented
    return True


def check_methods(C: type, *methods: str) -> CheckResult:
    return _check_generic(C, _check_dict, *methods)


def check_attributes(C: type, *properties: str) -> CheckResult:
    return _check_generic(C, _check_annotations, *properties)


def check_properties(C: type, *properties: str) -> CheckResult:
    return _check_generic(C, _check_property, *properties)


def get_abstracts(C: type) -> Tuple[List[str], ...]:
    abstracts = getattr(C, '__abstractmethods__', set())
    methods, properties = [], []
    for abt in abstracts:
        if isinstance(getattr(C, abt, None), property):
            properties.append(abt)
        else:
            methods.append(abt)
    return methods, properties


def subclass_hook(cls: type, subcls: type) -> CheckResult:
    """Inheritable implementation of __subclasshook__.

    Use in `__subclasshook__(cls, subcls)` as
    `return subclass_hook(cls, subcls)`
    """
    methods, properties = get_abstracts(cls)
    ok = check_methods(subcls, *methods)
    if ok is not True:
        return ok
    return check_properties(subcls, *properties)


def subclass_hook_nosub(mycls: type, cls: type, subcls: type) -> CheckResult:
    """Non-inheritable implementation of __subclasshook__.

    Use in `__subclasshook__(cls, subcls)` as
    `return subclass_hook_nosub(MyClass, cls, subcls)`
    """
    if cls is mycls:
        return subclass_hook(cls, subcls)
    return NotImplemented

# =============================================================================
# %%* ABC mixin with __subclasshook__
# =============================================================================


class ABCauto(abc.ABC):
    """Base class for ABCs with automatic subclass check for abstract methods.
    """

    @classmethod
    def __subclasshook__(cls, subcls):
        return subclass_hook(cls, subcls)

    def __init_subclass__(cls, typecheckonly: bool = False):
        if not typecheckonly:
            supname = supername(cls, ABCauto)
            raise TypeError(f'{supname} should not be used as a superclass.'
                            ' It is meant for instance/subclass checks only.')
