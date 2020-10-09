# -*- coding: utf-8 -*-
"""Helper classes for `matplotlib_tricks
"""
from __future__ import annotations

from functools import partial, wraps
from typing import Callable, Iterable, Literal, Optional
import typing as ty
import contextlib as cx

import matplotlib as mpl

import sl_py_tools.options_classes as op

# =============================================================================


class FontSize:
    """Font size property set.

    Parameters
    ----------
    func : Callable[op.Options, Axes -> Iterable[Text]]
        Function being decorated. It should return a list/tuple of the `Text`
        instances that need to be modified. The resulting decorated method
        takes an `Axes` as input and modifies it in-place.
    boolean : bool, optional
        Default value for boolean property, by default `True`.
    scale : float, optional
        Default value for scale property, by default `1`.
    doc : str|None, optional
        Docstring, by default `None`

    The owner class's `__new__` must contain a call to `self.prepare()` with
    the new instance as the parameter.

    See Also
    --------
    `fontsize_manager` - a decorator to implement this class.
    """
    func: TextGetter
    name: str
    _name: str
    _bool_default: bool
    _scale_default: float

    def __init__(self, func: TextGetter, *, boolean: bool = True,
                 scale: float = 1., doc: Optional[str] = None) -> None:
        self.func = func
        self.__doc__ = doc or func.__doc__
        self.name = func.__name__
        self._name = '_' + func.__name__
        self._bool_default = boolean
        self._scale_default = scale

    def prepare(self, obj: op.Options) -> None:
        """Call in __new__ of owner - essential."""
        setattr(obj, self._name, {'': self._bool_default, 'size': None,
                                  'scale': self._scale_default})

    def __get__(self, obj: op.Options, objtype: OwnerType = None) -> Modifier:
        if obj is None:
            return self

        @self._catch_new()
        @wraps(self.func)
        def method(axs: mpl.axes.Axes) -> None:
            """modify fonts"""
            if obj[self._name]['']:
                for txt in self.func(obj, axs):
                    txt.set_fontsize(obj[self.name + 'fontsize'])
                    txt.set_fontfamily(obj.fontfamily)
        return method

    def prop(self, key: Literal['', 'scale', 'size']) -> property:
        """Property for `<name>font<key>`.

        Parameters
        ----------
        key : ''|'scale'|'size'
            Which property to return.

        Returns
        -------
        `<name>font<key>` : property[float]
            Either the font size, the multiplier to get it from `obj.fontsize`
            or the flag which controls whether the elements returned by `func`
            will have their fonts modified.
        Where `<name> == func.__name__`. The class variables that are assigned
        the properties must have these names if the `<name>` method is to work.
        """
        @self._catch_new()
        def fget(obj: op.Options) -> float:
            return obj[self._name][key]

        @self._catch_new()
        def fset(obj: op.Options, value: float) -> None:
            obj[self._name][key] = value

        if key == 'size':
            fget, fset = self._size()

        self._set_propname(key, fget, fset)
        return property(fget, fset, None, self.__doc__)

    def _size(self) -> property:
        """Functions for property for `...fontsize`
        """
        @self._catch_new()
        def fget(obj: op.Options) -> Size:
            if obj[self._name]['size'] is not None:
                return obj[self._name]['size']
            if isinstance(obj.fontsize, str):
                return obj.fontsize
            return obj.fontsize * obj[self._name]['scale']

        @self._catch_new()
        def fset(obj: op.Options, value: Size) -> None:
            if isinstance(value, str):
                obj[self._name]['size'] = value
            else:
                obj[self._name]['size'] = None
                obj[self._name]['scale'] = value / obj.fontsize

        return fget, fset

    def props(self) -> ty.Tuple[property, ...]:
        """The properties for read/write access to font size control.

        Returns
        -------
        <name>font : property[bool]
            Flag which controls whether the elements returned by `func` will
            have their fonts modified.
        <name>fontsize : property[float|str]
            The fontsize to use for the elements returned by `func`. Whenever
            `obj.fontsize` changes, this property changes proportionally
            (provided both are numeric).
        <name>fontscale : property[float]
            The multiplier to get `<name>fontsize` from `obj.fontsize`. If one
            of this property and `<name>fontsize` is modified, the other is
            modified accordingly.
        Where `<name> == func.__name__`. The class variables that are assigned
        the properties must have these names if the `<name>` method is to work.
        """
        return self.prop(''), self.prop('size'), self.prop('scale')

    def _set_propname(self, name: str, *fns) -> None:
        """Set the __name__, __qualname__ & __doc__ of a property"""
        name = 'font' + name
        for fdo in fns:
            fdo.__name__ = self.func.__name__ + name
            fdo.__qualname__ = self.func.__qualname__ + name
            fdo.__doc__ = self.__doc__

    @cx.contextmanager
    def _catch_new(self) -> None:
        """Raise error if not initialised properly"""
        try:
            yield
        except KeyError as exc:
            name = self.name
            raise NotImplementedError(
                f"Property '{name}' has not been set up properly. The "
                f"`__new__` method should call `cls.{name}.prepare(obj)` "
                f"and the properties '{name}font', '{name}fontsize' and "
                f"'{name}fontscale' should be defined as {name}.props()."
            ) from exc
        finally:
            pass


def fontsize_manager(func: Optional[TextGetter] = None, *,
                     boolean: bool = True, scale: float = 1.,
                     doc: Optional[str] = None) -> FontSize:
    """Decorate a method to turn it into a FontSize descriptor.

    Parameters
    ----------
    func : Callable[Options, Axes -> Iterable[Text]]
        Function being decorated. It should return a list/tuple of the `Text`
        instances that need to be modified. The resulting decorated method
        takes an `Axes` as input and modifies it in-place.
    boolean : bool, optional
        Default value for boolean property, by default `True`.
    scale : float, optional
        Default value for scale property, by default `1`.
    doc : str|None, optional
        Docstring, by default `None`.

    If `func` is omitted, it returns a parameterised decorator.

    The decorated object has methods `prop(key)` and `props()` that return
    read-writable `properties`.

    See Also
    --------
    `FontSize` - the descriptor resulting from this function.

    Example
    -------
    class AxesOptions(options_classes.AnyOptions):

        @fontsize_manager(scale=1.)
        def axis(self, axs: mpl.axes.Axes) -> None:
            '''Modify font of axis-labels.'''
            if not self.axisfont:
                return ()
            return (axs.get_xaxis().get_label(), axs.get_yaxis().get_label())

        axis = typing.cast(FontSize, axis)  # to help the linter
        axisfont, axisfontsize, axisfontscale = axis.props()
    """
    if func is None:
        return partial(FontSize, boolean=boolean, scale=scale, doc=doc)
    return FontSize(func, boolean=boolean, scale=scale, doc=doc)


# =============================================================================
# Helper classes for axes
# =============================================================================


class CentredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an specified
    label for ticks at "centre".

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington

    Parameters
    ----------
    centre : float, optional
        Position of the tick to treat specially, by default 0
    label : str, optional
        Text to use for the special tick-label, by default ''
    """
    centre : float
    label: str

    def __init__(self, centre: float = 0, label: str = '', **kwds):
        super().__init__(**kwds)
        self.centre = centre
        self.label = label

    def __call__(self, value: float, pos=None):
        if value == self.centre:
            return self.label
        return super().__call__(value, pos)


class FancyArrowSpine(mpl.spines.Spine, mpl.patches.FancyArrowPatch):
    """Spine with an arrow.

    Parameters
    ----------
    other : mpl.spines.Spine
        The spine being replaced. Several attributes are copied from it,
        possibly overwriting some keyword parameters.
    Keywords are passed to the `__init__`s of `FancyArrowPatch` and `Spine`.
    The `FancyArrowPatch` parameters `path`, `posA`, `patchA`, `shrinkA` etc.
    have no effect.
    """

    def __init__(self, other: mpl.spines.Spine, **kwds) -> None:
        kwds.update(posA=[0, 0], posB=[1, 1])
        kwds.setdefault('arrowstyle', "-|>")
        kwds.setdefault('mutation_scale', 10)
        path = other.get_path()
        super().__init__(other.axes, other.spine_type, path, **kwds)
        self.update_from(other)
        self._path_original = path
        self._posA_posB = None
        self.register_axis(other.axis)
        self.set_position(other.get_position())

    def draw(self, renderer):
        """Make sure we call the right draw"""
        self._adjust_location()
        return mpl.patches.FancyArrowPatch.draw(self, renderer)


# =============================================================================
# Aliases
# =============================================================================
Owner = ty.TypeVar('Owner')
Size = ty.Union[int, float, str]
OwnerType = Optional[ty.Type[Owner]]
TextGetter = Callable[[op.Options, mpl.axes.Axes], Iterable[mpl.text.Text]]
Modifier = Callable[[mpl.axes.Axes], None]
