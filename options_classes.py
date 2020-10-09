# -*- coding: utf-8 -*-
"""Base class for options classes
"""
from __future__ import annotations

import operator
import collections.abc
import re
import typing as _ty
from typing import Any, Tuple

import matplotlib as mpl

import sl_py_tools.dict_tricks as _dt

_LINE_SEP = re.compile('\n {4,}')
# =============================================================================
# Fitter video options class
# =============================================================================


def _public(key: str) -> bool:
    """Is it a name of a public member?"""
    return not key.startswith('_')


def _norm_str(norm: mpl.colors.Normalize) -> str:
    """string rep of Normalize"""
    return norm.__class__.__name__ + f"({norm.vmin}, {norm.vmax})"


_FIX_STR = {mpl.colors.Colormap: operator.attrgetter('name'),
            mpl.colors.Normalize: _norm_str,
            collections.abc.Callable: operator.attrgetter('__name__')}


def _fmt_sep(format_spec: str) -> Tuple[str, str, str]:
    """helper for Options.__format__: process `format_spec`."""
    if '#' not in format_spec:
        conv, next_spec = '', format_spec
    else:
        conv, next_spec = format_spec.split('#', maxsplit=1)
    sep = ',' + next_spec if next_spec else ', '
    conv = "!" + conv if conv else conv
    return sep, conv, next_spec


def _fmt_help(key: str, val: Any, conv: str, next_spec: str) -> str:
    """helper for Options.__format__: entry for one item"""
    for cls, fun in _FIX_STR.items():
        if isinstance(val, cls):
            val = fun(val)
            break
    if conv != '!r' or _LINE_SEP.fullmatch(next_spec) is None:
        item = "{}={" + conv + "}"
        return item.format(key, val)
    val = repr(val).replace('\n', next_spec)
    return "{}={}".format(key, val)


# =============================================================================
# Base options class
# =============================================================================


# pylint: disable=too-many-ancestors
class Options(collections.abc.MutableMapping):
    """Base class for options classes

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    If the name is not found, it will search the attributes whose names are
    listed in `map_attributes`. This does not apply to use as attributes
    (either `obj.name` or `getattr(obj, 'name')`). Iterating and unpacking do
    not recurse through these attributes, nor include properties and private
    attributes, unless their names are included in `prop_attributes`.
    If an attribute's value is only set by a default value in a type hint, and
    not set in `__init__`, it will be omitted when iterating, unpacking or
    printing. If it is both a member of `self.__dict__` and listed in
    `prop_attributes`, it will appear twice.

    Setting attributes may be achieved via subscripting like a dictionary.
    If the name is not found, it will search the attributes listed in
    `map_attributes`. In case name exists in multiple map attributes, the
    first one in definition order will be chosen. You also specify which map
    attribute to choose with dotted keys: `options['suboptions.name']`. If
    a key is not found when recursing these attributes, a `KeyError` is raised.
    If an attribute's name is found in `map_attributes`, the attribute is
    updated when set rather than replaced like other attributes. These three
    statements do not apply to setting as an attribute. New keys may be added
    by setting as attributes, e.g. `obj.name=val` or `setattr(obj,'name',val)`.

    If a method `set_<name>(val)` exists, then `'<name>'` can be used as a key
    for setting but (unless it exists as a property or attribute) it cannot
    be used for getting. For such keys testing with `in` will return `False`,
    iteration will not include them and setting in a parent class will not
    propagate to this class.

    The attributes listed in `map_attributes` should be `MutableMapping`s.
    They, as well as the attributes in `prop_attributes`, will raise a
    `TypeError` if you try to delete them.

    If the same item appears in more than one of the `map_attributes`, or
    in `self`, they can be partially synchronised by making it a property in
    the parent `Options` with a `set_<key>` method that updates the children.

    Parameters
    ----------
    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.

    Raises
    ------
    KeyError
        If an invalid key is used when subscripting. This does not apply to use
        as attributes (either `obj.name` or `getattr(obj, 'name')`).
    """
    map_attributes: _ty.ClassVar[Tuple[str, ...]] = ()
    prop_attributes: _ty.ClassVar[Tuple[str, ...]] = ()
    key_last: _ty.ClassVar[Tuple[str, ...]] = ()
    key_first: _ty.ClassVar[Tuple[str, ...]] = ()

    def __init__(self, *args, **kwds) -> None:
        """The recommended approach to a subclass constructor is
        ```
        def __init__(self, *args, **kwds) -> None:
            self.my_attr = its_default
            self.other_attr = other_default
            ...
            self.last_attr = last_default
            order = ('my_attr', 'other_attr', ..., 'last_attr')
            args = sort_dicts(args, order, -1)
            kwds = sort_dict(kwds, order, -1)
            super().__init__(*args, **kwds)
        ```
        Mappings provided as positional arguments will be popped for the
        relevant items.
        """
        # put kwds in order
        for mapping in args:
            self.pop_my_args(mapping)
        self.update(kwds)
        # for name in self.map_attributes:
        #     if isinstance(self[name], MasterOptions):
        #         raise TypeError(
        #             f"{name} is a {type(self[name]).__name__}, a subclass of"
        #             + "MasterOptions. It should not be a member of another "
        #             + "Options class, as it will block searches for unknown "
        #             + "keys when setting.")

    def __format__(self, format_spec: str) -> str:
        """formatted string representing object.

        Parameters
        ----------
        format_spec : str
            Formating choice. If it does not contain a `'#'` it is added to
            `","` as a separator and inserted before the first member.
            When it takes the form `'x#blah'`, any non `Options` members are
            converted as `"{}={!x}".format(key, val)`. `Options` members are
            converted as "{}={:x#blah   }".format(key, val)` if blah consists
            only of a newline followed by a minimum of four spaces, or
            "{}={!x:blah}".format(key, val)` otherwise.

        Returns
        -------
        str
            String reoesentation of object.
        """
        sep, conv, next_spec = _fmt_sep(format_spec)
        attrs = sep.join(_fmt_help(key, val, conv, next_spec)
                         for key, val in self.items())
        return type(self).__name__ + f"({sep[1:]}{attrs})"

    def __repr__(self) -> str:
        return self.__format__('r#\n    ')

    # def __getattr__(self, name: str) -> Any:
    #     for attr in self.map_attributes:
    #         try:
    #             return getattr(getattr(self, attr), name)
    #         except AttributeError:
    #             pass
    #     raise AttributeError(s
    #         f"{type(self).__name__} has no item named {name}")

    def __getitem__(self, key: str) -> Any:
        """Get an attribute"""
        if '.' in key:
            *args, attr = key.split('.')
            obj = self
            for arg in args:
                obj = getattr(obj, arg)
            return obj[attr]
        try:
            return getattr(self, key)
        except AttributeError:
            for attr in self.map_attributes:
                try:
                    return getattr(self, attr)[key]
                except KeyError:
                    pass
        raise KeyError(f"Unknown key: {key}.")

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an existing attribute"""
        if '.' in key:
            *args, attr = key.split('.')
            obj = self
            for arg in args:
                obj = getattr(obj, arg)
            obj[attr] = value
            return
        if hasattr(self, 'set_' + key):
            getattr(self, 'set_' + key)(value)
        elif key in self.map_attributes:
            self[key].update(value)
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            for attr in self.map_attributes:
                if key in self[attr]:
                    getattr(self, attr).__setitem__(key, value)
                    return
                # try:
                #     getattr(self, attr).__setitem__(key, value)
                # except KeyError:
                #     pass
                # else:
                #     return
            raise KeyError(f"Unknown key: {key}.")

    def __delitem__(self, key: str) -> None:
        if key in self.map_attributes + self.prop_attributes:
            raise TypeError(f"`del {type(self).__name__}['{key}']` disallowed")
        if '.' in key:
            *args, attr = key.split('.')
            obj = self
            for arg in args:
                obj = getattr(obj, arg)
            del obj[attr]
        try:
            delattr(self, key)
        except AttributeError:
            # raise KeyError(f"Unknown key: {key}.")
            for attr in self.map_attributes:
                try:
                    del getattr(self, attr)[key]
                except KeyError:
                    pass
                else:
                    return
            raise KeyError(f"Unknown key: {key}.") from None

    def __len__(self) -> int:
        # len(self.__dict__) + len(self.prop_attributes) includes privates.
        # tuple(self) appears to call len(self) -> infinite recursion.
        # return len(tuple(x for x in self))
        # barely any speed difference:
        count = 0
        for _ in self:
            count += 1
        return count

    def __iter__(self) -> _ty.Iterator[str]:
        yield from filter(_public, self.__dict__)
        yield from self.prop_attributes

    def update(self, __m: StrDictable = (), /, **kwargs) -> None:
        """Update from mappings/iterables"""
        # put kwds in order
        __m, kwargs = self.order_keys(__m, kwargs)
        super().update(__m, **kwargs)

    def copy(self) -> Options:
        """Get a shallow copy of the object.

        Onlu copies those attributes that appear when iterating.
        """
        return type(self)(**self)

    def pop_my_args(self, kwds: StrDict) -> None:
        """Pop any key from dict that can be set and use the value to set.
        """
        # put kwds in order
        sorted_kwds, = self.order_keys(kwds)
        to_pop = []
        # update what we can
        for key, val in sorted_kwds.items():
            try:
                self[key] = val
            except KeyError:
                pass
            else:
                to_pop.append(key)
        # pop the ones we used
        for key in to_pop:
            del kwds[key]

    @classmethod
    def order_keys(cls, *kwds: StrDict) -> _ty.List[StrDict]:
        """Lists of keys in order for start and end"""
        key_first = cls.key_first + cls.map_attributes
        key_last = cls.prop_attributes + cls.key_last
        return _dt.sort_ends_dicts(kwds, key_first, key_last)
# pylint: enable=too-many-ancestors


# =============================================================================
# Fallback options class
# =============================================================================


# pylint: disable=too-many-ancestors
class AnyOptions(Options):
    """Same to `Options`, except it stores unknown keys as attributes.

    This can be used as a default place to store unknown items.
    """
    def __setitem__(self, key: str, val: Any) -> None:
        try:
            super().__setitem__(key, val)
        except KeyError:
            setattr(self, key, val)
# pylint: enable=too-many-ancestors


# =============================================================================
# Master options class
# =============================================================================


# pylint: disable=too-many-ancestors
class MasterOptions(Options):
    """Same to `Options`, except ot stores unknown keys in a mapping attribute.

    The name of the fallback mapping attribute is specified with the keyword
    `fallback` in the class definition.

    fallback : str|None
        Name of a mutable mapping attribute to store any completely unknown
        keys, i.e. if recursing through the mappings listed in `map_attributes`
        does not turn anything up. Any empty string indacates using `self` as
        the fallback storage, `None` indicates raising a `KeyError`.
    """
    _fallback_mapping: _ty.ClassVar[str]

    def __init_subclass__(cls, fallback: _ty.Optional[str] = None) -> None:
        cls._fallback_mapping = fallback

    def __setitem__(self, key: str, val: Any) -> None:
        try:
            super().__setitem__(key, val)
        except KeyError:
            if self._fallback_mapping is None:
                raise
            if self._fallback_mapping:
                getattr(self, self._fallback_mapping).__setitem__(key, val)
            else:
                setattr(self, key, val)
# pylint: enable=too-many-ancestors


# =============================================================================
# Helpers (don't work)
# =============================================================================


def list_to_be(*args: Immutable) -> Saved[_ty.List[Immutable]]:
    """Delayed list creation.

    The returned tuple will, upon passing unpacked to `get_now`, return a
    `list` containing `args`.
    """
    return (list, (args,))


def dict_to_be(**kwds: _dt.Val) -> Saved[StrDict[_dt.Val]]:
    """Delayed dict creation.

    The returned tuple will, upon passing unpacked to `get_now`, return a
    `dict` containing `kwds`.
    """
    return (dict, _dict_iter(kwds))  # we'll pass dict kwds positionally


# These are really the same
# but hopefully the names are more evocative
def to_be(cls: _ty.Type[Mutable], *args: Immutable, **kwds: Immutable
          ) -> Saved[Mutable]:
    """Delayed class instance creation

    The returned tuple will, upon passing unpacked to `get_now`, return an
    instance of `cls` with `args` and `kwds` as parameters.
    """
    return (cls, args, _dict_iter(kwds))


def later(func: Factory[Mutable], *args: Immutable, **kwds: Immutable
          ) -> Saved[Mutable]:
    """Delayed function calling

    The returned object will, upon being called with new parameters, call the
    function `func` with `args` and `kwds` inserted after the new parameters.
    """
    return (func, args, _dict_iter(kwds))


def _dict_iter(to_store: _ty.Mapping[str, _dt.Val]) -> Kwds[_dt.Val]:
    """Convert dict to tuple of items"""
    return tuple(to_store.items())


def get_now(func: Factory[Mutable], args: Args[Any] = (), kwds: Kwds[Any] = ()
            ) -> Mutable:
    """Reconstruct a mutable value from saved parameters

    Parameters
    ----------
    func : Callable[..., Mutable]
        Function that returns the mutable value.
    args : Tuple[Any]
        Tuple of saved positional arguments.
    kwds : Tuple[Tuple[str, Any], ...]
        Saved keyword parameters.

    Returns
    -------
    value : Mutable
        The reconstructed value.
    """
    return func(*args, **dict(kwds))


# =============================================================================
# Hinting aliases
# =============================================================================
StrDict = _ty.Dict[str, _dt.Val]
StrDictable = _dt.Dictable[str, _dt.Val]
Attrs = _ty.ClassVar[Tuple[str, ...]]
Mutable = _ty.TypeVar('Mutable')
Immutable = _ty.TypeVar('Immutable')
Factory = _ty.Callable[..., Mutable]
Args = Tuple[Immutable, ...]
Kwds = Tuple[Tuple[str, Immutable], ...]
Saved = Tuple[Factory[Mutable], Args[Any], Kwds[Any]]
