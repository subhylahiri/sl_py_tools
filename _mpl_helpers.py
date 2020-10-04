# -*- coding: utf-8 -*-
"""Helper classes for `matplotlib_tricks
"""
from __future__ import annotations

import functools as _ft
import typing as _ty

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
    """
    func: TextGetter
    name: str
    _name: str
    _bool_default: bool
    _scale_default: float

    def __init__(self, func: TextGetter, *, boolean: bool = True,
                 scale: float = 1., doc: _ty.Optional[str] = None) -> None:
        self.func = func
        self.__doc__ = doc or func.__doc__
        self.name = func.__name__
        self._name = '_' + func.__name__
        self._bool_default = boolean
        self._scale_default = scale

    def prepare(self, obj: op.Options) -> None:
        """Call in __new__ of owner"""
        setattr(obj, self._name, {'bool': self._bool_default, 'size': None,
                                  'scale': self._scale_default})

    def __get__(self, obj: op.Options, objtype: OwnerType = None) -> Modifier:
        if obj is None:
            return self

        @_ft.wraps(self.func)
        def method(axs: mpl.axes.Axes) -> None:
            """modify fonts"""
            for txt in self.func(obj, axs):
                txt.set_fontsize(obj[self.name + 'fontsize'])
                txt.set_fontfamily(obj.fontfamily)
        return method

    def prop(self, field: str, name: _ty.Optional[str] = None) -> property:
        """Accessor property for `...<name>` or `...font<field>`"""
        def fget(obj: op.Options) -> float:
            return obj[self._name][field]

        def fset(obj: op.Options, value: float) -> None:
            obj[self._name][field] = value

        self._set_propname(name or 'font' + field, fget, fset)
        return property(fget, fset, None, self.__doc__)

    def size(self) -> property:
        """Accessor property for `...fontsize`"""
        def fget(obj: op.Options) -> Size:
            if obj[self._name]['size'] is not None:
                return obj[self._name]['size']
            if isinstance(obj.fontsize, str):
                return obj.fontsize
            return obj.fontsize * obj[self._name]['scale']

        def fset(obj: op.Options, value: Size) -> None:
            if isinstance(value, str):
                obj[self._name]['size'] = value
            else:
                obj[self._name]['size'] = None
                obj[self._name]['scale'] = value / obj.fontsize

        self._set_propname('fontsize', fget, fset)
        return property(fget, fset, None, self.__doc__)

    def props(self) -> _ty.Tuple[property, ...]:
        """The properties for access"""
        return self.prop('bool', 'font'), self.size(), self.prop('scale')

    def _set_propname(self, name: str, *fns) -> None:
        """Set the __name__, __qualname__ & __doc__ of a property"""
        for fdo in fns:
            fdo.__name__ = self.func.__name__ + name
            fdo.__qualname__ = self.func.__qualname__ + name
            fdo.__doc__ = self.__doc__


def fontsize_manager(func: _ty.Optional[TextGetter] = None, *,
                     boolean: bool = True, scale: float = 1.,
                     doc: _ty.Optional[str] = None) -> FontSize:
    """Decorate a method to turn it into a FontSize descriptor.
    """
    if func is None:
        return _ft.partial(FontSize, boolean=boolean, scale=scale, doc=doc)
    return FontSize(func, boolean=boolean, scale=scale, doc=doc)


# =============================================================================
# Helper classes for axes
# =============================================================================


class CentredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "centre".

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
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
    """Spine with an arrow
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
Owner = _ty.TypeVar('Owner')
Size = _ty.Union[int, float, str]
OwnerType = _ty.Optional[_ty.Type[Owner]]
TextGetter = _ty.Callable[[op.Options, mpl.axes.Axes],
                          _ty.Iterable[mpl.text.Text]]
Modifier = _ty.Callable[[mpl.axes.Axes], None]
