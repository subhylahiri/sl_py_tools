# -*- coding: utf-8 -*-
"""Tools for making matplotlib nicer
"""
import typing as _ty
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects
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


def clean_axes(axs: plt.Axes, fontsize=20, **kwds):
    """Make axes look prettier
    """
    if axs is None:
        axs = plt.gca()
    if kwds.pop('box', True):
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
    if kwds.pop('axisfont', True):
        axs.get_yaxis().get_label().set_fontsize(fontsize)
    if kwds.pop('titlefont', True):
        axs.title.set_fontsize(fontsize)
    if axs.legend_ is not None:
        if kwds.pop('legendbox', True):
            axs.legend_.set_frame_on(False)
        if kwds.pop('legendfont', True):
            for x in axs.legend_.get_texts():
                x.set_fontsize(fontsize)
    axs.set(**kwds)


def centre_spines(axs: _ty.Optional[plt.Axes] = None,
                  centrex=0, centrey=0, **kwds):
    """Centres the axis spines at <centrex, centrey> on the axis "axs", and
    places arrows at the end of the axis spines.

    Parameters
    ----------
    axs : plt.Axes, optional
        Axes to be centred, default: plt.gca.
    in_bounds : {bool, Sequence[bool]}
        Ensure that spines are within axes limits? If it is a scalar, it
        applies to both axes. If it is a sequence, `in_bounds[0/1]` applies to
        x/y-axis respectively.
    arrow : {bool, Sequence[bool]}
        Draw an arrow on splines? If it is a scalar, it applies to both axes.
        If it is a sequence, `arrow[0/1]` applies to x/y-axis respectively.
    centre_tick : str, optional
        How we label central tick.
            x:
                Use x-coordinate of centre.
            y:
                Use y-coordinate of centre.
            both:
                Use both coordinates of centre as (x, y).
            none:
                Do not label.
        Any other value interpreted as 'none', default: 'both'.

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
    """
    if axs is None:
        axs = plt.gca()

    in_bounds = _cn.tuplify(kwds.pop('in_bounds', False), 2)
    if in_bounds[0]:
        centrex = _cn.Interval(*axs.get_xlim()).clip(centrex)
    if in_bounds[1]:
        centrey = _cn.Interval(*axs.get_ylim()).clip(centrey)

    # Set the axis's spines to be centred at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)
    axs.spines['left'].set_position(('data', centrex))
    axs.spines['bottom'].set_position(('data', centrey))
    axs.spines['right'].set_position(('data', centrex - 1))
    axs.spines['top'].set_position(('data', centrey - 1))

    # Hide the line (but not ticks) for "extra" spines
    for side in ['right', 'top']:
        axs.spines[side].set_color('none')

    arrow = _cn.tuplify(kwds.pop('arrow', False), 2)
    # Draw an arrow at the end of the spines
    add_axes_arrows(axs, *arrow)

    # On both the x and y axes...
    for axis, centre in zip([axs.xaxis, axs.yaxis], [centrex, centrey]):
        # Turn ticks
        axis.set_ticks_position('both')

        # Hide the ticklabels at <centrex, centrey>
        formatter = CentredFormatter(centre)
        axis.set_major_formatter(formatter)

    # Add offset ticklabels at <centrex, centrey> using annotation
    # (Should probably make these update when the plot is redrawn...)
    centre_tick = kwds.pop('centre_tick', 'both').lower()  # {both,x,y,none}
    xlab, ylab = map(formatter.format_data, [centrex, centrey])
    ctr_lab = {'x': f"{xlab}", 'y': f"{ylab}", 'both': f"({xlab}, {ylab})"}
    if centre_tick in ctr_lab.keys():
        axs.annotate(ctr_lab[centre_tick], (centrex, centrey), xytext=(-4, -4),
                     textcoords='offset points', ha='right', va='top')


class CentredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "centre".

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
    """
    centre = 0

    def __init__(self, centre=0, **kwds):
        super().__init__(**kwds)
        self.centre = centre

    def __call__(self, value, pos=None):
        if value == self.centre:
            return ''
        else:
            return super().__call__(value, pos)


# Note: I'm implementing the arrows as a path effect rather than a custom
#       Spines class. In the long run, a custom Spines class would be a better
#       way to go. One of the side effects of this is that the arrows aren't
#       reversed when the axes are reversed! (Joe Kington)


class EndArrow(mpl.patheffects._Base):
    """A matplotlib patheffect to add arrows at the end of a path.

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
    """

    def __init__(self, headwidth=5, headheight=5, facecolor=(0, 0, 0), **kwds):
        super(mpl.patheffects._Base, self).__init__()
        self.width, self.height = headwidth, headheight
        self._gc_args = kwds
        self.facecolor = facecolor

        self.trans = mpl.transforms.Affine2D()

        self.arrowpath = mpl.path.Path(
                np.array([[-0.5, -0.2], [0.0, 0.0], [0.5, -0.2],
                          [0.0, 1.0], [-0.5, -0.2]]),
                np.array([1, 2, 2, 2, 79]))

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        scalex = renderer.points_to_pixels(self.width)
        scaley = renderer.points_to_pixels(self.height)

        x0, y0 = tpath.vertices[-1]
        dx, dy = tpath.vertices[-1] - tpath.vertices[-2]
        azi = np.arctan2(dy, dx) - np.pi / 2.0
        trans = affine + self.trans.clear().scale(scalex, scaley
                                                  ).rotate(azi
                                                           ).translate(x0, y0)

        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        self._update_gc(gc0, self._gc_args)

        if self.facecolor is None:
            color = rgbFace
        else:
            color = self.facecolor

        renderer.draw_path(gc0, self.arrowpath, trans, color)
        renderer.draw_path(gc, tpath, affine, rgbFace)
        gc0.restore()


def add_axes_arrows(axs: _ty.Optional[plt.Axes] = None,
                    to_xaxis: bool = True, to_yaxis: bool = True):
    """Add arrows to axes.

    Parameters
    ----------
    axs : plt.Axes, optional
        Axes on which to add arrows, default: plt.gca.
    to_xaxis : bool, optional
        Whether we add arrow to x-axis, default: True.
    to_yaxis : bool, optional
        Whether we add arrow to y-axis, default: True.

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
    """
    if axs is None:
        axs = plt.gca()
    # Draw an arrow at the end of the spines
    if to_xaxis:
        axs.spines['left'].set_path_effects([EndArrow()])
    if to_yaxis:
        axs.spines['bottom'].set_path_effects([EndArrow()])


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
