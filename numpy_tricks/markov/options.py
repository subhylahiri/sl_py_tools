# -*- coding: utf-8 -*-
"""Class for specifying topology of Markovian models
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

# import numpy as np

# import numpy_linalg as la
import sl_py_tools.numpy_tricks.markov._helpers as _mh
import sl_py_tools.options_classes as _opt

# =============================================================================
# Class for specifying topology of parameterised synapses
# =============================================================================


# pylint: disable=too-many-ancestors
class TopologyOptions(_opt.Options):
    """Class that contains topology specifying options.

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    Parameters
    ----------
    serial : bool, optional keyword
        Restrict to models of serial topology? By default `False`.
    ring : bool, optional keyword
        Restrict to models of ring topology? By default `False`.
    uniform : bool, optional keyword
        Restrict to models with equal rates per direction? By default `False`.
    directions: Tuple[int] (P,), optional keyword
        If nonzero, only include transitions in direction `i -> i + sgn(drn)`,
        one value for each plasticity type. By default `(0, 0)`.
    discrete : bool
        Are we kaking transition matrices for a discrete-time Markov process?
        By default `False`.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    key_last: _opt.Attrs = ('directions', 'npl')
    serial: bool = False
    ring: bool = False
    uniform: bool = False
    directions: Tuple[int, ...] = (0, 0)
    discrete: bool = False

    def __init__(self, *args, **kwds) -> None:
        self.serial = self.serial
        self.ring = self.ring
        self.uniform = self.uniform
        self.directions = self.directions
        self.discrete = self.discrete
        super().__init__(*args, **kwds)
        if self.constrained and 'directions' not in kwds:
            # different default if any(serial, ring, uniform)
            self.directions = (1, -1)
            if 'npl' in kwds:
                self.npl = kwds['npl']

    def directed(self, which: Union[int, slice, None] = slice(None), **kwds
                 ) -> Dict[str, Any]:
        """Dictionary of Markov parameter options

        Parameters
        ----------
        which : int, slice, None, optional
            Which element of `self.directions` to use as the `'drn'` value,
            where `None` -> omit `'drn'` item. By default `slice(None)`
        Extra arguments are default values or unknown keys in `opts`

        Returns
        -------
        opts : Dict[str, Any]
            All options for `sl_py_tools.numpy_tricks.markov.params`.
        """
        if which is not None:
            kwds['drn'] = self.directions[which]
        kwds.update(serial=self.serial, ring=self.ring, uniform=self.uniform)
        if self.discrete:
            kwds['stochastifier'] = _mh.stochastify_pd
        return kwds

    @property
    def constrained(self) -> bool:
        """Are there any constraints on the topology?
        """
        return any((self.serial, self.ring, self.uniform) + self.directions)

    @constrained.setter
    def constrained(self, value: Optional[bool]) -> None:
        """Remove all constraints on topology by setting it `False`.

        Does nothing if `value` is `None`. Raises `ValueError if it is `True`.
        """
        if value is None:
            return
        if value:
            raise ValueError("Cannot directly set `constrained=True`. "
                             + "Set a specific constraint instead.")
        self.serial = False
        self.ring = False
        self.uniform = False
        self.directions = (0,) * self.npl

    @property
    def npl(self) -> int:
        """Number of transition matrices
        """
        return len(self.directions)

    @npl.setter
    def npl(self, value: Optional[int]) -> None:
        """Set the number of transition matrices.

        Does nothing if `value` is `None`. Removes end elements of `directions`
        if shortening. Appends zeros if lengthening.
        """
        if value is None:
            return
        self.directions = self.directions[:value] + (0,) * (value - self.npl)
