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
PROP_TYPES = (property, types.MemberDescriptorType)


def typename(inst: Any) -> str:
    """String of name of type"""
    return type(inst).__name__


def supername(cls: type, base: type = object) -> str:
    """String of name of superclass

    Searches for first subclass of `base` in `cls.__mro__` other than `cls`.
    raises `ValueError` if not found.
    """
    for scls in cls.__mro__:
        if scls is not cls and issubclass(scls, base):
            return scls.__name__
    raise ValueError(f"{base.__name__} is not a superclass of {cls.__name__}")

# =============================================================================
# Type check utilities
# =============================================================================


def _check_dict(the_class: type, method: str) -> CheckResult:
    """Check if method is in class dictionary.
    """
    if method in the_class.__dict__:
        if the_class.__dict__[method] is None:
            return NotImplemented
        return True
    return False


def _check_annotations(the_class: type, prop: str) -> CheckResult:
    """Check if attribute is in class annotations.
    """
    return prop in getattr(the_class, '__annotations__', {})


def _check_property(the_class: type, prop: str) -> CheckResult:
    """Check if prop is in class dictionary (as a property) or annotation.
    """
    is_ok = _check_dict(the_class, prop)
    if is_ok is NotImplemented:
        return NotImplemented
    if is_ok:
        return isinstance(the_class.__dict__[prop], PROP_TYPES)
    return _check_annotations(the_class, prop)


def _check_generic(the_cls: type, check: Checker, *methods: str) -> CheckResult:
    """Check class for methods
    """
    mro = the_cls.__mro__
    for method in methods:
        for super_class in mro:
            is_ok = check(super_class, method)
            if is_ok is NotImplemented:
                return NotImplemented
            if is_ok:
                break
        else:
            return NotImplemented
    return True


def check_methods(the_class: type, *methods: str) -> CheckResult:
    """Check if methods are in class dictionary.
    """
    return _check_generic(the_class, _check_dict, *methods)


def check_attributes(the_class: type, *properties: str) -> CheckResult:
    """Check if attributes are in class annotations.
    """
    return _check_generic(the_class, _check_annotations, *properties)


def check_properties(the_class: type, *properties: str) -> CheckResult:
    """Check if properties are in class dictionary (as property) or annotations.
    """
    return _check_generic(the_class, _check_property, *properties)


def get_abstracts(the_class: type) -> Tuple[List[str], ...]:
    """Get names of abstract methods and properties
    """
    abstracts = getattr(the_class, '__abstractmethods__', set())
    methods, properties = [], []
    for abt in abstracts:
        if isinstance(getattr(the_class, abt, None), property):
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
    is_ok = check_methods(subcls, *methods)
    if is_ok is not True:
        return is_ok
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
# ABC mixin with __subclasshook__
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
