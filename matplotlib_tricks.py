# -*- coding: utf-8 -*-
"""Tools for making matplotlib nicer
"""
import typing as _ty
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from . import arg_tricks as _ag
from . import containers as _cn


def rc_fonts():
    """Global font options
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = 'serif'


def clean_axes(ax: plt.Axes, fontsize=20, **kwds):
    """Make axes look prettier
    """
    if ax is None:
        ax = plt.gca()
    if kwds.pop('box', True):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if kwds.pop('axisfont', True):
        ax.get_yaxis().get_label().set_fontsize(fontsize)
    if kwds.pop('titlefont', True):
        ax.title.set_fontsize(fontsize)
    if ax.legend_ is not None:
        if kwds.pop('legendbox', True):
            ax.legend_.set_frame_on(False)
        if kwds.pop('legendfont', True):
            for x in ax.legend_.get_texts():
                x.set_fontsize(fontsize)
    ax.set(**kwds)


def plot_equality(axs: plt.Axes, line: mpl.lines.Line2D = None, npt=2, **kwds):
    """Plot the equality line on Axes
    """
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    eq_vals = np.linspace(max(xlim[0], ylim[0]), min(xlim[1], ylim[1]), npt)
    if line is None:
        kwds.setdefault('label', "equality")
        kwds.setdefault('color', "k")
        kwds.setdefault('linestyle', "-")
        line = axs.plot(eq_vals, eq_vals, **kwds)
    else:
        line.set_xdata(eq_vals)
        line.set_ydata(eq_vals)
        line.update(kwds)
    return line


def common_clim(imh: _ty.Sequence[mpl.collections.QuadMesh],
                cmin: _ty.Optional[float] = None,
                cmax: _ty.Optional[float] = None):  # set all clims equal
    """
    Make the color limits for each image in sequence the same

    Parameters
    ----------
    imh : Sequence[pcolormesh]
        sequence of pcolormesh objects with heatmaps
    cmin : optional
        Fixed lower end of clim. If `None`, use min of `imh.clim[0]`.
    cmax : optional
        fixed upper end of clim. If `None`, use max of `imh.clim[1]`.
    """
    imh = _cn.listify(imh)
    old_clim = np.array([im.get_clim() for im in imh])

    cmin = _ag.default(cmin, np.amin(old_clim[:, 0]))
    cmax = _ag.default(cmax, np.amax(old_clim[:, 1]))

    for im in imh:
        im.set_clim((cmin, cmax))


def centre_clim(imh: _ty.Sequence[mpl.collections.QuadMesh],
                centre: float = 0.):  # set all clims equal
    """
    Make the color limits for each image in sequence the same

    Parameters
    ----------
    imh : Sequence[pcolormesh]
        sequence of pcolormesh objects with heatmaps
    cmin : optional
        Fixed lower end of clim. If `None`, use min of `imh.clim[0]`.
    cmax : optional
        fixed upper end of clim. If `None`, use max of `imh.clim[1]`.
    """
    imh = _cn.listify(imh)
    for img in imh:
        old_clim = img.get_clim()
        cdiff = max(old_clim[1] - centre, centre - old_clim[0])
        img.set_clim((centre - cdiff, centre + cdiff))
