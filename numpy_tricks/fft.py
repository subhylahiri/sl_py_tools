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
        Sample spacings (inverse of the sampling rates). Defaults to 1.

    Returns
    -------
    freqs : Tuple[np.ndarray]
        Arrays of frequencies.

    See Also
    --------
    numpy.fft.fftfreq : for each axis
    """
    freqs = ()
    if spacings is None:
        spacings = (1.,) * len(shape)
    for siz, step in zip(shape, spacings):
        freqs += fft.fftfreq(siz, step)
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
        Sample spacings (inverse of the sampling rates). Defaults to 1.

    Returns
    -------
    freqs : Tuple[np.ndarray]
        Arrays of frequencies.

    See Also
    --------
    numpy.fft.fftfreq : for all axes except last.
    numpy.fft.rfftfreq : for last axis.
    """
    if spacings is None:
        spacings = (1.,) * len(shape)
    freqs = fftfreqs(shape[:-1], spacings[:-1])
    freqs += (fft.rfftfreq(shape[-1], spacings[-1]),)
    return freqs


def fft_flip(fftarray: np.ndarray,
             axes: Sequence[int] = (-1,),) -> np.ndarray:
    """
    Swap positive/negative frequencies and complex conjugate Fourier transform.

    Result is the Fourier transform of the complex conjugate of the inverse
    transform of the input array.
    If the input array is from a real Fourier transform the final axis should
    be excluded, as it doesn't contain negative frequencies.

    Parameters
    ----------
    fftarray : np.ndarray
        Fourier transform.
    axes : Sequence[int], optional
        Which axes were Fourier transformed? Default = (-1,).
        Exclude final axis for a real Fourier transform.

    Returns
    -------
    rfftarray : np.ndarray
        Fourier transform with frequencies flipped, complex conjugated.
    """
    fftconj = fftarray.conj()
    return np.roll(np.flip(fftconj, axes), 1, axes)


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
        Is the full length of `axes[-1]` even? Default = True.

    Returns
    -------
    fftarray : np.ndarray
        Fourier transform including redundant frequencies.

    See Also
    --------
    fft_to_rfft : inverse of this function
    """
    axis = axes[-1]
    indices = np.arange(1, rfftarray.shape[axis] - even)
    redundant = np.take(rfftarray, indices, axis=axis)
    redundant = np.flip(fft_flip(redundant, axes[:-1]), axis)
    return np.concatenate((rfftarray, redundant), axis)


def fft_to_rfft(fftarray: np.ndarray,
                axes: Sequence[int] = (-1,),) -> np.ndarray:
    """Restrict Fourier transform to independent frequencies.

    The fourier transform of a real quantity has a Hermitean symmetry.
    Therefore half of the frequency components are redundant and not kept by
    ``rfft*``. This function symmetrises and then removes those frequencies.

    Parameters
    ----------
    fftarray : np.ndarray
        Fourier transform including redundant frequencies.
    axes : Sequence[int], optional
        Which axes were Fourier transformed? Default = (-1,).

    Returns
    -------
    rfftarray : np.ndarray
        Fourier transform excluding redundant frequencies.

    See Also
    --------
    rfft_to_fft : inverse of this function
    """
    fftconj = fft_flip(fftarray, axes)
    fftconj += fftarray
    fftconj /= 2.
    axis = axes[-1]
    new_siz = fftarray.shape[axis] // 2 + 1
    return np.take(fftconj, np.arange(new_siz), axis=axis)
