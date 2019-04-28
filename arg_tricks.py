"""Tools for processing function arguments.
"""
import typing as _ty
A = _ty.TypeVar('A')


def default(optional: _ty.Optional[A], default_value: A) -> A:
    """Replace argument with default if it is None


    Parameters
    ----------
    optional : A or None
        The optional argument, where `None` indicates that the default value
        should be used instead.
    default_value : A
        Default value for the argument, used when `optional` is `None`.

    Returns
    -------
    use_value : A
        Either `optional`, if it is not `None` or `default_value` if it is.
    """
    return default_value if (optional is None) else optional


def defaults(optionals: _ty.Iterable[_ty.Optional[A]],
             default_vals: _ty.Iterable[A]) -> _ty.Tuple[A, ...]:
    """Replace arguments with defaults if they are None

    Parameters
    ----------
    optionals : Iterable[A or None]
        Tuple of arguments, where entries of `None` indicate that the default
        value should be used instead.
    default_vals : Iterable[A]
        Tuple of default values for arguments, they are used when the
        corresponding element of `optionals` is `None`.

    Returns
    -------
    use_vals : Tuple[A]
        The corresponding elements of `optionals` or `default_vals`, using
        the latter only when the former is `None`.
    """
    return tuple(default(opt, df) for opt, df in zip(optionals, default_vals))
