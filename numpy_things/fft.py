# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:46:13 2018

@author: Subhy
"""

from typing import Optional, Sequence, Tuple
import numpy as np
from numpy import fft


def fftfreqs(shape: Sequence[int],
             spacings: Optional[Sequence[float]] = None) -> Tuple[np.ndarray]:
    """
    Return the Discrete Fourier Transform sample frequencies for each axis.

    Parameters
    ----------
    shape : Sequence[int]
        Axes length.
    spacings : Sequence[scalar], optional
        Sample spacing s(inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    freqs : Tuple[np.ndarray]
        Arrays of frequencies.

    See Also
    --------
    numpy.fft.fftfreq : for each axes
    """
    freqs = ()
    if spacings is None:
        spacings = (1.,) * len(shape)
    for n, d in zip(shape, spacings):
        freqs += fft.fftfreq(n, d)
    return freqs


def rfftfreqs(shape: Sequence[int],
              spacings: Optional[Sequence[float]] = None) -> Tuple[np.ndarray]:
    """
    The Discrete Real Fourier Transform sample frequencies for each axis.

    Parameters
    ----------
    shape : Sequence[int]
        Axes length.
    spacings : Sequence[scalar], optional
        Sample spacing s(inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    freqs : Tuple[np.ndarray]
        Arrays of frequencies.

    See Also
    --------
    numpy.fft.fftfreq : for all axes but last.
    numpy.fft.rfftfreq : for last axis.
    """
    freqs = ()
    if spacings is None:
        spacings = (1.,) * len(shape)
    for n, d in zip(shape[:-1], spacings[:-1]):
        freqs += (fft.fftfreq(n, d),)
    freqs += (fft.rfftfreq(shape[-1], spacings[-1]),)
    return freqs


def rfft_to_fft(rfftarray: np.ndarray,
                axes: Sequence[int] = (-1,),
                even: bool = True) -> np.ndarray:
    """
    Extend Real Fourier transform to all frequencies.

    The fourier transform of a real quantity has a Hermitean symmetry.
    Therefore half of the frequency components are redundant and not kept by
    ``rfft*``. This function fills in those frequencies.

    Parameters
    ----------
    rfftarray : np.ndarray
        Fourier transform excluding redundant frequencies.
    axes : Sequence[int], optional
        Which axes were Fourier transformed? Default = (-1,).
    even : bool, optional
        Is the full length of `axis` even? Default = True.

    Returns
    -------
    fftarray : np.ndarray
        Fourier transform including redundant frequencies.

    See Also
    --------
    fft_to_rfft : inverse of this function
    """
    axis = axes[-1]
    if axis < 0:
        axis += rfftarray.ndim
    inds = (slice(None),) * axis
    if even:
        inds += (slice(1, -1),)
    else:
        inds += (slice(1, None),)
    redundant = np.flip(rfftarray.conj()[inds], axis)
    for ax in axes[:-1]:
        redundant = np.roll(np.flip(redundant, ax), 1, ax)
    return np.concatenate((rfftarray, redundant), axis)


def fft_to_rfft(fftarray: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Parameters
    ----------
    Extend Real Fourier transform to all frequencies.

    The fourier transform of a real quantity has a Hermitean symmetry.
    Therefore half of the frequency components are redundant and not kept by
    ``rfft*``. This function removes those frequencies.

    Parameters
    ----------
    fftarray : np.ndarray
        Fourier transform including redundant frequencies.
    axis : int, optional
        Which axis was the last one of the Fourier transform? Default = -1.

    Returns
    -------
    rfftarray : np.ndarray
        Fourier transform excluding redundant frequencies.

    See Also
    --------
    rfft_to_fft : inverse of this function
    """
    siz = fftarray.shape[axis]
    new_siz = siz // 2 + 1
    if axis < 0:
        axis += fftarray.ndim
    inds = (slice(None),) * axis + (slice(new_siz),)
    return fftarray[inds]
