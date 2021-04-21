# -*- coding: utf-8 -*-
"""Tools for making matplotlib nicer
"""
from __future__ import annotations
import typing as _ty
import logging
import os
from typing import Optional

import matplotlib as mpl
import matplotlib.animation as mpa
import matplotlib.backends.backend_pdf as pdf
import matplotlib.pyplot as plt
import numpy as np

import sl_py_tools.arg_tricks as _ag
import sl_py_tools.containers as _cn
import sl_py_tools.tol_colors as _tc
import sl_py_tools.options_classes as _op
import sl_py_tools._mpl_helpers as _mph

_log = logging.getLogger(__name__)
# =============================================================================
# Global Options
# =============================================================================


def rc_fonts(family: str = 'serif') -> None:
    """Global font options, use LaTeX.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = family
    mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath,amssymb}"
    if family == 'sans-serif':
        mpl.rcParams['text.latex.preamble'] += r"\usepackage[T1]{fontenc}"
        mpl.rcParams['text.latex.preamble'
                     ] += r"\renewcommand\familydefault\sfdefault"
        mpl.rcParams['text.latex.preamble'] += r"\usepackage{euler}"
        mpl.rcParams['text.latex.preamble'] += r"\usepackage{mathastext}"


def rc_colours(cset: str = 'bright', cmap: str = 'YlOrBr',
               reg: _ty.Tuple[str, ...] = ()) -> None:
    """Global line colour options.
    """
    prop_cycle = mpl.cycler(color=list(_tc.tol_cset(cset)))
    mpl.rcParams['axes.prop_cycle'] = prop_cycle
    for cmp in reg:
        if cmp in _tc.tol_cmap():
            mpl.cm.register_cmap(cmp, _tc.tol_cmap(cmp))
        elif cmp in _tc.tol_cset():
            mpl.cm.register_cmap(cmp, _tc.tol_cset(cmp))
        else:
            raise ValueError(f"Unknown colourmap {cmp}")
    mpl.cm.register_cmap(cmap, _tc.tol_cmap(cmap))
    mpl.rcParams['image.cmap'] = cmap


# =============================================================================
# Figures
# =============================================================================


def fig_square(fig: mpl.figure.Figure) -> None:
    """Adjust figure width so that it is square, and tight layout.

    Parameters
    ----------
    fig : mpl.figure.Figure
        `Figure` instance.
    """
    fig.set_size_inches(mpl.figure.figaspect(1))
    fig.tight_layout()


# =============================================================================
# Axes lines, etc
# =============================================================================


def equal_axlim(axs: mpl.axes.Axes, mode: str = 'union') -> None:
    """Make x/y axes limits the same.

    Parameters
    ----------
    axs : mpl.axes.Axes
        `Axes` instance whose limits are to be adjusted.
    mode : str
        How do we adjust the limits? Options:
            'union'
                Limits include old ranges of both x and y axes, *default*.
            'intersect'
                Limits only include values in both ranges.
            'x'
                Set y limits to x limits.
            'y'
                Set x limits to y limits.
    Raises
    ------
    ValueError
        If `mode` is not one of the options above.
    """
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    modes = {'union': (min(xlim[0], ylim[0]), max(xlim[1], ylim[1])),
             'intersect': (max(xlim[0], ylim[0]), min(xlim[1], ylim[1])),
             'x': xlim, 'y': ylim}
    if mode not in modes:
        raise ValueError(f"Unknown mode '{mode}'. Shoulde be one of: "
                         "'union', 'intersect', 'x', 'y'.")
    new_lim = modes[mode]
    axs.set_xlim(new_lim)
    axs.set_ylim(new_lim)


def calc_axlim(data: np.ndarray, err: _ty.Optional[np.ndarray] = None,
               log: bool = False, buffer: float = 0.05,
               ) -> _ty.Tuple[float, float]:
    """Calculate axes limits that will show all data

    Parameters
    ----------
    data : np.ndarray (n)
        array of numbers plotted along the axis
    err : None or np.ndarray (n) or (2,n), optional
        error bars for data, by default: `None`
    log : bool, optional
        is it a log scale? By default: `False`
    buffer : float, optional
        fractional padding around data, by default: `0.05`.
    """
    errs = np.broadcast_to(_ag.default(err, 0.), (2,) + data.shape)
    lim = np.array([np.nanmin(data - errs[0]), np.nanmax(data + errs[1])])
    if log:
        np.log(lim, out=lim)
    lim += np.array([-1, 1]) * buffer * (lim[1] - lim[0])
    if log:
        np.exp(lim, out=lim)
    return tuple(lim)


def set_new_axlim(axs: plt.Axes,
                  data: np.ndarray, err: _ty.Optional[np.ndarray] = None, *,
                  yaxis: bool = True, reset: bool = False, buffer: float = 0.05
                  ) -> None:
    """Set axes limits that will show all data, including existing.

    Parameters
    ----------
    axs : mpl.axes.Axes
        `Axes` instance whose limits are to be adjusted.
    data : np.ndarray (n)
        Array of numbers plotted along the axis.
    err : None|np.ndarray (n,)|(2,n), optional
        Error bars for data, by default: `None`.
    yaxis : bool, optional
        Are we modifying the y axis? By default: `True`.
    reset : bool, optional
        Do we ignore the existing axis limits? By default: `False`.
    buffer : float, optional
        Fractional padding around data, by default `0.05`.
    """
    log = (axs.get_yscale() if yaxis else axs.get_xscale()) == 'log'
    lim = calc_axlim(data, err, log, buffer)
    if not reset:
        if yaxis:
            axlim = axs.get_ylim()
        else:
            axlim = axs.get_xlim()
        lim = min(lim[0], axlim[0]), max(lim[1], axlim[1])
    if yaxis:
        axs.set_ylim(lim)
    else:
        axs.set_xlim(lim)


def clean_axes(axs: plt.Axes, fontsize: _ty.Union[int, str] = 20,
               fontfamily: str = "sans-serif", **kwds):
    """Make axes look prettier.

    All non-font size kewwords default to `True`.
    This can be changed with the keyword `all`.

    Parameters
    ----------
    axs : plt.Axes
        `Axes` object to modify.
    fontsize : float|str
        Font size for axes labels, ticks & title. By default `20`.
    fontfamily : str
        Font family for axes labels, ticks & title. By default `'sans-serif'`.

    Keyword only
    ------------
    box : bool
        Remove axes box?
    axisfont : bool
        Change axes font size?
    titlefont : bool
        Change title font size?
    legendbox : bool
        Remove legend box?
    legendfont : bool
        Change legend font size?
    tickfont : bool
        Change tick-label font size?
    tight : bool
        Apply tight_layout to figure?
    all : bool
        Choice for any of the above that is unspecified, default: True
    axisfontsize : float|str
        Font size for axis-labels, default: `fontsize * titlefontscale`.
    titlefontsize : float|str
        Font size for title, default: `fontsize * axisfontscale`.
    legendfontsize : float|str
        Font size for legend entries, default: `fontsize * legendfontscale`.
    tickfontsize : float|str
        Font size for tick-labels, default: `fontsize * tickfontscale`.
    axisfontscale : number
        Multiplies `fontsize` (if numeric) for axis-labels, by default `1`.
    titlefontscale : number
        Multiplies `fontsize` (if numeric) for the title, by default `1.2`.
    legendfontscale : number
        Multiplies `fontsize` (if numeric) for legend entries, by default `1`.
    tickfontscale : number
        Multiplies `fontsize` (if numeric) for tick-labels, by default `0.694`.
    """
    clean_kws = AxesOptions(kwds, fontsize=fontsize, fontfamily=fontfamily)
    clean_kws.pop_my_args(kwds)
    if axs is None:
        axs = plt.gca()
    if clean_kws['box']:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
    clean_kws.title(axs)
    clean_kws.axis(axs)
    clean_kws.tick(axs)
    if axs.legend_ is not None:
        if clean_kws['legendbox']:
            axs.legend_.set_frame_on(False)
        clean_kws.legend(axs)
    axs.set(**kwds)
    if clean_kws['tight']:
        axs.figure.tight_layout()


def adjust_legend_font(leg: mpl.legend.Legend, **kwds):
    """Adjust font properties of legend text

    Parameters
    ----------
    leg
        legend instance
    **kwds
        keyword arguments passed to font properties manager.
    see `mpl.font_manager.FontProperties` for a list of keywords.
    """
    for txt in leg.get_texts():
        txt.set_fontproperties(mpl.font_manager.FontProperties(**kwds))


def centre_spines(axs: _ty.Optional[plt.Axes] = None,
                  centrex: float = 0, centrey: float = 0, **kwds) -> None:
    """Centres the axis spines at <centrex, centrey> on the axis "axs", and
    places arrows at the end of the axis spines.

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington

    Parameters
    ----------
    axs : plt.Axes, optional
        Axes to be centred, default: plt.gca().
    centrex : float, optional
        x-coordinate of centre, where y axis is drawn, default: 0
    centrey : float, optional
        y-coordinate of centre, where x axis is drawn, default: 0

    Keyword only
    ------------
    in_bounds : {bool, Sequence[bool]}
        Ensure spines are within axes limits? If it is a scalar, it applies to
        both axes. If it is a sequence, `in_bounds[0/1]` applies to x/y-axis
        respectively.
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
                Use both coordinates of centre as 'x,y'.
            paren:
                Use both coordinates of centre as '(x,y)'.
            none:
                Do not label.
        Any other value interpreted as 'none', default: 'both'.
    """
    axs = _ag.default_eval(axs, plt.gca)

    in_bounds = _cn.tuplify(kwds.pop('in_bounds', False), 2)
    if in_bounds[0]:
        centrex = _cn.Interval(*axs.get_xlim()).clip(centrex)
    if in_bounds[1]:
        centrey = _cn.Interval(*axs.get_ylim()).clip(centrey)

    # Set the axis's spines to be centred at the given point
    axs.tick_params(direction='inout')
    axs.spines['left'].set_position(('data', centrex))
    axs.spines['bottom'].set_position(('data', centrey))
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)

    # Remove tick labels at centre
    xformatter = _mph.CentredFormatter(centrex, '')
    yformatter = _mph.CentredFormatter(centrey, '')
    axs.xaxis.set_major_formatter(xformatter)
    axs.yaxis.set_major_formatter(yformatter)
    lab = [xformatter.format_data(centrex), yformatter.format_data(centrey)]

    # Add offset ticklabels at <centrex, centrey>
    centre_tick = kwds.pop('centre_tick', 'both').lower()  # {both,x,y,none}
    centre_labs = {'none': "", 'x': "{} ", 'y': "{1} ", 'both': "{},{} ",
                   'paren': "({},{}) "}
    xformatter.label = centre_labs.get(centre_tick, "").format(*lab)
    which, = (axs.get_xticks() == centrex).nonzero()
    if which.size:
        axs.get_xticklabels()[which[0]].set_ha('right')
        # also shift position by (-yaxis.get_tick_padding(), 0) in pt

    # Draw an arrow at the end of the spines
    arrow = _cn.tuplify(kwds.pop('arrow', False), 2)
    add_axes_arrows(axs, *arrow)


def add_axes_arrows(axs: _ty.Optional[mpl.axes.Axes] = None,
                    to_xaxis: bool = True, to_yaxis: bool = True, **kwds):
    """Add arrows to axes.

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington

    Parameters
    ----------
    axs : plt.Axes, optional
        Axes on which to add arrows, default: plt.gca.
    to_xaxis : bool, optional
        Whether we add arrow to x-axis, default: True.
    to_yaxis : bool, optional
        Whether we add arrow to y-axis, default: True.
    """
    axs = axs or plt.gca()
    # Draw an arrow at the end of the spines
    if to_xaxis:
        spine_arrow(axs, 'bottom', **kwds)
    if to_yaxis:
        spine_arrow(axs, 'left', **kwds)


def spine_arrow(axs: mpl.axes.Axes, spine: str, **kwds) -> None:
    """Add an arrow to a spine

    Parameters
    ----------
    axs : mpl.axes.Axes
        Axes holding the spine.
    spine : str
        Name of the spine to change.
    """
    axs.spines[spine] = _mph.FancyArrowSpine(axs.spines[spine], **kwds)



# =============================================================================
# Lines
# =============================================================================


def plot_equality(axs: mpl.axes.Axes, **kwds) -> mpl.lines.Line2D:
    """Plot the equality line on Axes

    Parameters
    ----------
    axs : mpl.axes.Axes
        Axes on which we draw the equality line
    **kwds
        Passed to `Axes.axline`.

    Returns
    -------
    line : mpl.lines.Line2D
        The reulting line object

    See Also
    --------
    matplotlib.axes.Axes.axline([0, 0], [1, 1], **kwds)
    """
    return axs.axline([0, 0], [1, 1], **kwds)


def stepify_data(boundaries: np.ndarray, values: np.ndarray,
                 axis: _ty.Union[int, _ty.Sequence[int]] = -1
                 ) -> _ty.Tuple[np.ndarray, np.ndarray]:
    """Create data for a step plot

    Parameters
    ----------
    boundaries : np.ndarray (N+1,)
        Edges of flat regions in step plot.
    values : np.ndarray (N,)
        Heights of flat regions in step plot.
    axis : int or Sequence[int], optional
        Which axis the data sets lie along, applies to both `boundaries` and
        `values` if a single number is given, by default -1

    Returns
    -------
    xdata, ydata : [np.ndarray, np.ndarray], (2N,), (2N,)
        Data for `plt.plot` to produce a step plot.

    Raises
    ------
    ValueError
         if `boundaries.shape[axis[0]] != values.shape[axis[1]] + 1`.
    """
    axis = _cn.tuplify(axis, 2)
    if boundaries.shape[axis[0]] != values.shape[axis[1]] + 1:
        raise ValueError(f"Length of `boundaries` along axis {axis[0]} should"
                         + f"1 more than `values` along axis {axis[1]},\n"
                         + f"but {boundaries.shape[axis[0]]} != "
                         + f"{values.shape[axis[1]]} + 1.")
    edges = np.moveaxis(boundaries, axis[0], -1)
    heights = np.moveaxis(values, axis[1], -1)
    xdata = np.stack((edges[..., :-1], edges[..., 1:]), axis=-1)
    ydata = np.stack((heights, heights), axis=-1)
    xdata = np.moveaxis(xdata.reshape(xdata.shape[:-2] + (-1,)), -1, axis[0])
    ydata = np.moveaxis(ydata.reshape(ydata.shape[:-2] + (-1,)), -1, axis[1])
    return xdata, ydata


# =============================================================================
# Colour limits, etc
# =============================================================================


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
    old_clim = np.array([img.get_clim() for img in imh])

    cmin = _ag.default(cmin, np.amin(old_clim[:, 0]))
    cmax = _ag.default(cmax, np.amax(old_clim[:, 1]))

    for img in imh:
        img.set_clim((cmin, cmax))


def centre_clim(imh: _ty.Sequence[mpl.collections.QuadMesh],
                centre: float = 0.):  # set all clims equal
    """
    Make the color limits for each image in sequence the samesymmetrical.

    Parameters
    ----------
    imh : QuadMesh|Sequence[QuadMesh]
        Sequence of pcolormesh objects with heatmaps.
    centre : float, optional
        Value to centre color climits around, by default 0.
    """
    imh = _cn.listify(imh)
    for img in imh:
        old_clim = img.get_clim()
        cdiff = max(old_clim[1] - centre, centre - old_clim[0])
        img.set_clim((centre - cdiff, centre + cdiff))


# =============================================================================
# Options classes
# =============================================================================


# pylint: disable=too-many-ancestors,too-many-instance-attributes
class AxesOptions(_op.Options):
    """Options for `clean_axes`.

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    Parameters
    ----------
    fontsize : float|str
        Base font size for axes labels, ticks, legends & title. By default 20.
    fontfamily : str
        Font family for axis/tick-labels/legend/title. By default 'sans-serif'.
    all : bool
        Choice for any `bool` below that is unspecified, by default `True`.
    box : bool
        Remove axes box?
    legendbox : bool
        Remove legend box?
    tight : bool
        Apply tight_layout to figure?
    <element>font : bool
        Change <element> font? By default `True`.
    <element>fontsize : float|str
        Font size for <element>, by default `fontsize * <element>fontscale`.
    <element>fontscale : float
        Multiplier of `fontsize` (if numeric) for <element>, defaults below.
    Where <element> is:-
        `axis`:
            Axis labels, default `axisfontscale`: `1`.
        `title`:
            Axes title, default `titlefontscale`: `1.2`.
        `legend`:
            Legend entries, default `legendfontscale`: `1`.
        `tick`:
            Tick labels, default `tickfontscale`: `0.694`.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    prop_attributes: _op.Attrs = ('axisfont', 'titlefont', 'legendfont',
                                  'tickfont', 'axisfontsize', 'titlefontsize',
                                  'legendfontsize', 'tickfontsize')
    key_first: _op.Attrs = ('all', 'fontsize')

    fontsize : _ty.Union[int, float, str] = 20
    fontfamily : str = 'sans-serif'
    tight : bool = True
    box : bool = True
    legendbox : bool = True

    def __new__(cls, *args, **kwds) -> AxesOptions:
        obj = super().__new__(cls)
        # pylint: disable=no-member
        cls.axis.prepare(obj)
        cls.title.prepare(obj)
        cls.legend.prepare(obj)
        cls.tick.prepare(obj)
        assert any((True, args, kwds))
        return obj

    def __init__(self, *args, **kwds) -> None:
        self.fontsize = self.fontsize
        self.fontfamily = self.fontfamily
        self.tight = self.tight
        self.box = self.box
        self.legendbox = self.legendbox
        super().__init__(*args, **kwds)

    @_mph.fontsize_manager(scale=1.)
    def axis(self, axs: mpl.axes.Axes) -> None:
        """Modify font of axis-labels."""
        if not self.axisfont:
            return ()
        return (axs.get_xaxis().get_label(), axs.get_yaxis().get_label())

    axis = _ty.cast(_mph.FontSize, axis)
    axisfont, axisfontsize, axisfontscale = axis.props()

    @_mph.fontsize_manager(scale=1.2)
    def title(self, axs: mpl.axes.Axes) -> None:
        """Modify font of the title."""
        if not self.titlefont:
            return ()
        return (axs.title,)

    title = _ty.cast(_mph.FontSize, title)
    titlefont, titlefontsize, titlefontscale = title.props()

    @_mph.fontsize_manager(scale=1.)
    def legend(self, axs: mpl.axes.Axes) -> None:
        """Modify font of legend entries"""
        if axs.legend_ is None or not self.legendfont:
            return ()
        return axs.legend_.get_texts()

    legend = _ty.cast(_mph.FontSize, legend)
    legendfont, legendfontsize, legendfontscale = legend.props()

    @_mph.fontsize_manager(scale=0.694)
    def tick(self, axs: mpl.axes.Axes) -> None:
        """Modify font of tick-labels."""
        if not self.tickfont:
            return ()
        return axs.get_xticklabels() + axs.get_yticklabels()

    tick = _ty.cast(_mph.FontSize, tick)
    tickfont, tickfontsize, tickfontscale = tick.props()

    def set_all(self, value: bool) -> None:
        """Set all boolean options.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.box =value
            self.legendbox = value
            self.tight = value
            self.axisfont = value
            self.titlefont = value
            self.legendfont = value
            self.tickfont = value
# pylint: enable=too-many-ancestors


# pylint: disable=too-many-ancestors
class ImageOptions(_op.AnyOptions):
    """Options for heatmaps

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    Parameters
    ----------
    cmap : str|Colormap
        Colour map used to map numbers to colours. By default, `'YlOrBr'`.
    norm : Normalize
        Maps heatmap values to interval `[0, 1]` for `cmap`.
        By default: `Normalise(0, 1)`.
    vmin : float
        Lower bound of `norm`. By default: `0`.
    vmax : float
        Lower bound of `norm`. By default: `1`.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    prop_attributes: _op.Attrs = ('cmap',)
    _cmap: mpl.colors.Colormap = _op.later(mpl.cm.get_cmap, 'YlOrBr')
    norm: mpl.colors.Normalize = _op.to_be(mpl.colors.Normalize, 0., 1.)

    def __init__(self, *args, **kwds) -> None:
        self._cmap = _op.get_now(*self._cmap)
        self.norm = _op.get_now(*self.norm)
        super().__init__(*args, **kwds)

    @property
    def cmap(self) -> mpl.colors.Colormap:
        """Get the colour map.
        """
        return self._cmap

    @cmap.setter
    def cmap(self, value: _ty.Union[str, mpl.colors.Colormap]) -> None:
        """Set the colour map.

        Does nothing if `value` is `None`. Converts to `Colormap` if `str`.
        """
        if value is None:
            pass
        elif isinstance(value, str):
            self._cmap = mpl.cm.get_cmap(value)
        elif isinstance(value, mpl.colors.Colormap):
            self._cmap = value
        else:
            raise TypeError("cmap must be `str` or `mpl.colors.Colormap`, not "
                            + type(value).__name__)

    @property
    def vmin(self) -> float:
        """The lower bound for the colour map.
        """
        return self.norm.vmin

    @vmin.setter
    def vmin(self, value: float) -> None:
        """Set the lower bound for the colour map.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.norm.vmin = value

    @property
    def vmax(self) -> float:
        """The upper bound for the colour map.
        """
        return self.norm.vmax

    @vmax.setter
    def vmax(self, value: float) -> None:
        """Set the upper bound for the colour map.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.norm.vmax = value

    def val_to_colour(self, values) -> np.ndarray:
        """Normalise and convert values to colours

        Parameters
        ----------
        values : array_like (N,)
            Values to convert to colours

        Returns
        -------
        cols : np.ndarray (N, 4)
            RGBA array representing colours.
        """
        return self._cmap(self.norm(values))
# pylint: enable=too-many-ancestors


# pylint: disable=too-many-ancestors
class AnimationOptions(_op.AnyOptions):
    """Options for animations

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    Parameters
    ----------
    interval : float
        The gap between frames in milliseconds, by default `500`.
    repeat_delay : float
        The gap between repetitions in milliseconds, by default `1000`.
    repeat : bool
        Whether the animation repeats when the sequence of frames is completed,
        by default `True`.
    blit : bool
        Do we only redraw the parts that have changed? By default `False`.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    interval: float = 500
    repeat_delay: float = 1000
    repeat: bool = True
    blit: bool = False

    def __init__(self, *args, **kwds) -> None:
        self.interval = self.interval
        self.repeat_delay = self.repeat_delay
        self.repeat = self.repeat
        self.blit = self.blit
        super().__init__(*args, **kwds)
# pylint: enable=too-many-ancestors


# =============================================================================
# Animation writers
# =============================================================================


@mpa.writers.register('file_seq')
class FileSeqWriter(mpa.FileMovieWriter):
    """Write an animation as a sequence of image files.

    Parameters
    ----------
    fps : int
        Movie frame rate (per second), by default `5`.
    codec : str|None
        The codec to use, by default `None` -> :rc:`animation.codec`.
    bitrate : int|None
        The bitrate of the movie, in kilobits per second.  Higher values
        means higher quality movies, but increase the file size.  A value
        of -1 lets the underlying movie encoder select the bitrate.
        By default `None` -> :rc:`animation.bitrate`
    metadata : Dict[str, str]|None
        A dictionary of keys and values for metadata to include in the
        output file. Some keys that may be of use include:
        title, artist, genre, subject, copyright, srcform, comment.
        By default `None -> {}`.
    """
    supported_formats: _ty.ClassVar[_ty.List[str]] = [
        'pdf', 'svg', 'png', 'jpeg', 'ppm', 'tiff', 'sgi', 'bmp', 'pbm', 'raw',
    ]
    fname_format_str : str
    _ndigit: int = 7

    def __init__(self, *args, ndigit: int = 7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ndigit = ndigit

    def setup(self,  # pylint: disable=arguments-differ
              fig: mpl.figure.Figure, outfile: os.PathLike,
              dpi: _ty.Optional[float] = None,
              ndigit: _ty.Optional[int] = None,
              ) -> None:
        """Set the output file properties.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure to grab the rendered frames from.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, optional
            The dpi of the output file. This, with the figure size,
            controls the size in pixels of the resulting movie file.
            Default is ``fig.dpi``.
        ndigit : int, optional
            Number of digits to leave space for in numbered file names.
        """
        outfile = os.fsdecode(outfile)
        if '.' in outfile:
            frame_prefix, self.frame_format = outfile.rsplit('.', 1)
        else:
            frame_prefix = outfile
        super().setup(fig, outfile, dpi, frame_prefix=frame_prefix)
        if ndigit is not None:
            self._ndigit = ndigit
        self.fname_format_str = f'%s%%0{self._ndigit}d.%s'

    def finish(self) -> None:
        """Perform cleanup - nothing!"""

    def cleanup(self) -> None:
        """Perform cleanup - nothing!"""

    @classmethod
    def isAvailable(cls) -> bool:
        return True


@mpa.writers.register('pdf_pages')
class PdfPagesWriter(mpa.AbstractMovieWriter):
    """Write animation as a multi-page pdf file.

    Parameters
    ----------
    fps : int
        Movie frame rate (per second), by default `5`.
    codec : str|None
        The codec to use, by default `None` -> :rc:`animation.codec`.
    bitrate : int|None
        The bitrate of the movie, in kilobits per second.  Higher values
        means higher quality movies, but increase the file size.  A value
        of -1 lets the underlying movie encoder select the bitrate.
        By default `None` -> :rc:`animation.bitrate`
    metadata : Dict[str, str]|None
        A dictionary of keys and values for metadata to include in the
        output file. Some keys that may be of use include:
        title, artist, genre, subject, copyright, srcform, comment.
        By default `None -> {}`.
    """
    supported_formats: _ty.ClassVar[_ty.List[str]] = ['pdf']
    _file: _ty.Optional[pdf.PdfPages]

    def __init__(self, fps: int = 5, codec: Optional[str] = None,
                 bitrate: Optional[int] = None,
                 metadata: Optional[_ty.Dict[str, str]]=None) -> None:
        super().__init__(fps=fps, metadata=metadata, codec=codec,
                         bitrate=bitrate)
        self.frame_format = 'pdf'
        self._file = None

    def setup(self, fig, outfile, dpi=None):
        """Setup for writing the movie file.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file. By default ``fig.dpi``.
        """
        _log.info('Call AbstractMovieWriter.setup')
        super().setup(fig, outfile, dpi=dpi)
        _log.info('Open PDF file')
        self._file = pdf.PdfPages(outfile, metadata=self.metadata)

    def grab_frame(self, **savefig_kwargs):
        """
        Grab the image information from the figure and save as a movie frame.

        All keyword arguments in *savefig_kwargs* are passed on to the
        `~.Figure.savefig` call that saves the figure.
        """
        _log.debug('Save to PDF file')
        self._file.savefig(self.fig, **savefig_kwargs)

    def finish(self):
        """Finish any processing for writing the movie."""
        _log.info('Close PDF file')
        self._file.close()

    @classmethod
    def isAvailable(cls) -> bool:  # pylint: disable=invalid-name
        """Make sure the registry knows we're available"""
        return True
