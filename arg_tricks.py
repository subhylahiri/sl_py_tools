# -*- coding: utf-8 -*-
"""Tools for processing function arguments.
"""
import typing as _ty

Some = _ty.TypeVar('Some')
Other = _ty.TypeVar('Other')
# =============================================================================


def default(optional: _ty.Optional[Some], default_value: Some) -> Some:
    """Replace optional with default value if it is None

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    default_value : Some
        Default value for the argument, used when `optional` is `None`.

    Returns
    -------
    use_value : Some
        Either `optional`, if it is not `None` or `default_value` if it is.
    """
    return default_value if (optional is None) else optional


def default_eval(optional: _ty.Optional[Some],
                 default_fn: _ty.Callable[[], Some], *args, **kwds) -> Some:
    """Replace optional with evaluation of default function if it is None

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    default_fn : Callable[()->Some]
        Evaluates to default value for the argument, only evaluated and used
        when `optional` is `None`. Does not take any arguments.

    Returns
    -------
    use_value : Some
        Either `optional`, if it is not `None` or `default_fn()` if it is.
    """
    if optional is None:
        return default_fn(*args, **kwds)
    return optional


def eval_or_default(optional: _ty.Optional[Some],
                    non_default_fn: _ty.Callable[[Some], Other],
                    default_value: Other, *args, **kwds) -> Other:
    """Evaluate function on optional if it is not None

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    non_default_fn : Callable[(Some)->Other]
        Evaluated on `optional`if it is not `None`.
    default_value : Other
        Default value for the argument, used when `optional` is `None`.

    Returns
    -------
    use_value : Other
        Either `non_default_fn(optional)`, if `optional` is not `None` or
        `default_value` if it is.
    """
    if optional is None:
        return default_value
    return non_default_fn(optional, *args, **kwds)


def eval_or_default_eval(optional: _ty.Optional[Some],
                         non_default_fn: _ty.Callable[[Some], Other],
                         default_fn: _ty.Callable[[], Other],
                         *args, **kwds) -> Other:
    """Evaluate function on optional if it is not None, else evaluate default.

    Parameters
    ----------
    optional : Some or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    non_default_fn : Callable[(Some)->Other]
        Evaluated on `optional`if it is not `None`.
    default_fn : Callable[()->Other]
        Evaluates to default value for the argument, only evaluated and used
        when `optional` is not `None`.

    Returns
    -------
    use_value : Other
        Either `non_default_fn(optional)`, if `optional` is not `None` or
        `default_fn()` if it is.
    """
    if optional is None:
        return default_fn(*args, **kwds)
    return non_default_fn(optional, *args, **kwds)


def defaults(optionals: _ty.Iterable[_ty.Optional[Some]],
             default_vals: _ty.Iterable[Some]) -> _ty.Tuple[Some, ...]:
    """Replace arguments with defaults if they are None

    If the `optionals` and `default_vals` have different lengths, the shorter
    one is padded with `None`.

    Parameters
    ----------
    optionals : Iterable[Some or None]
        Tuple of arguments, where entries of `None` indicate that the default
        value should be used instead.
    default_vals : Iterable[Some]
        Tuple of default values for arguments, they are used when the
        corresponding element of `optionals` is `None`.

    Returns
    -------
    use_vals : Tuple[Some]
        The corresponding elements of `optionals` or `default_vals`, using
        the latter only when the former is `None` or absent.
    """
    extra = len(default_vals) - len(optionals)
    optionals = tuple(optionals) + (None,) * extra
    default_vals = tuple(default_vals) + (None,) * -extra
    return tuple(default(opt, df) for opt, df in zip(optionals, default_vals))


default_non_eval = eval_or_default
non_default_eval = eval_or_default_eval


def args_to_kwargs(args: _ty.Tuple[Some],
                   kwargs: _ty.Dict[str, Some],
                   names: _ty.Iterable[str],
                   kw_rank: bool = False) -> _ty.Tuple[Some]:
    """Convert positional arguments to keyword arguments

    Parameters
    ----------
    args : Tuple[Some, ...]
        Tuple of positional arguments.
    kwargs : Dict[str, Other]
        Dict of keyword arguments.
    names : List[str]
        Names of the positional arguments.
    kw_rank : bool, optional
        If positional and keyword versions of an argument both exist, does the
        keyword version have priority? By default, False.

    Returns
    -------
    extra : Tuple[Some,...]
        positional arguments with no names
    """
    if kw_rank:
        for key, val in zip(names, args):
            kwargs.setdefault(key, val)
    else:
        kwargs.update(zip(names, args))
    return args[len(names):]

# =============================================================================
# * Dummy type hint
# =============================================================================


class Export:  # pylint: disable=too-few-public-methods
    """Dummy module/package level type hint to fool pyflakes/pylint.

    Should behave like a type hint that 'these were imported to make them
    available to the users of this module rather than for use in the module'.

    Does not actually do anything. It is really only intended for files like
    `__init__.py`, where you might import things from private modules to make
    them part of the public interface, etc. Including expressions such as
    `_EXPORTED = Export[import1, import2, ...]` will stop pyflakes/pylint from
    complaining that `'import1' imported but unused`, etc.
    """

    def __class_getitem__(cls, *arg) -> None:  # pylint: disable=unused-argument
        pass


def dummy(*args) -> None:  # pylint: disable=unused-argument
    """Doesn't do anything."""


# =============================================================================
# Class and instance attributes
# =============================================================================


def _dir_dict(obj: _ty.Any) -> _ty.Dict[str, None]:
    """Get the unsorted dictionary keys of a class/instance (like dir())
    """
    my_dir = {}
    for base in getattr(obj, '__bases__', ()):
        my_dir.update(_dir_dict(base))
    # could use cls.__dir__() or dir(cls)
    my_dir.update((k, None) for k in obj.__dict__)
    return my_dir


def dir_nosort(obj: _ty.Any) -> _ty.List[str]:
    """Get the unsorted directory of a class/instance (like dir())
    """
    return list(_dir_dict(obj))


def inst_attrs(obj: _ty.Any) -> _ty.List[str]:
    """List of names of instance attributes that are not class attributes

    Parameters
    ----------
    obj : Any
        The instance whose attributes we want

    Returns
    -------
    attrs : List[str]
        Instance attribute names
    """
    return list(set(dir(obj)) - set(dir(type(obj))))


def inst_attrs_nosort(obj: _ty.Any) -> _ty.List[str]:
    """List of names of instance attributes that are not class attributes

    Parameters
    ----------
    obj : Any
        The instance whose attributes we want

    Returns
    -------
    attrs : List[str]
        Instance attribute names
    """
    inst, clss = dir_nosort(obj), dir_nosort(type(obj))
    return [attr for attr in inst if attr not in clss]
