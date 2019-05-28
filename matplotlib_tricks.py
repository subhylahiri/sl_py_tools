# -*- coding: utf-8 -*-
"""Tools for making matplotlib nicer
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


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
