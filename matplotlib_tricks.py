# -*- coding: utf-8 -*-
"""Tools for making matplotlib nicer
"""
import typing as _ty

import matplotlib as mpl
import matplotlib.patheffects as mpf
import matplotlib.pyplot as plt
import numpy as np

import sl_py_tools.arg_tricks as _ag
import sl_py_tools.containers as _cn
import sl_py_tools.tol_colors as tol
import sl_py_tools.options_classes as op

# =============================================================================
# Global Options
# =============================================================================


def rc_fonts(family: str = 'serif'):
    """Global font options, use LaTeX.
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = family
    mpl.rcParams['text.latex.preamble'] = "\\usepackage{amsmath,amssymb}"


def rc_colours(cset: str = 'bright', cmap: str = 'YlOrBr',
               reg: _ty.Tuple[str, ...] = ()):
    """Global line colour options.
    """
    prop_cycle = mpl.cycler(color=list(tol.tol_cset(cset)))
    mpl.rcParams['axes.prop_cycle'] = prop_cycle
    for cmp in reg:
        if cmp in tol.tol_cmap():
            mpl.cm.register_cmap(cmp, tol.tol_cmap(cmp))
        elif cmp in tol.tol_cset():
            mpl.cm.register_cmap(cmp, tol.tol_cset(cmp))
        else:
            raise ValueError(f"Unknown colourmap {cmp}")
    mpl.cm.register_cmap(cmap, tol.tol_cmap(cmap))
    mpl.rcParams['image.cmap'] = cmap


# =============================================================================
# Figures
# =============================================================================


def fig_square(fig: mpl.figure.Figure):
    """Adjust figure width so that it is square, and tight layout

    Parameters
    ----------
    fig
        instance of matplotlib.figure.Figure
    """
    fig.set_size_inches(mpl.figure.figaspect(1))
    fig.tight_layout()


# =============================================================================
# Axes lines, etc
# =============================================================================


def equal_axlim(axs: mpl.axes.Axes, mode: str = 'union'):
    """Make x/y axes limits the same

    Parameters
    ----------
    axs : mpl.axes.Axes
        axes instance whose limits are to be adjusted
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


def calc_axlim(data, err=None, log=False, buffer=0.05):
    """Calculate axes limits that will show all data

    Parameters
    ----------
    data : np.ndarray (n)
        array of numbers plotted along the axis
    err : None or np.ndarray (n) or (2,n), optional
        error bars for data, default: None
    log : bool, optional
        is it a log scale? default: False
    buffer : float, optional
        fractional padding around data
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
                  yaxis: bool = True, reset: bool = False, log: bool = False,
                  buffer: float = 0.05):
    """Calculate axes limits that will show all data, including existing

    Parameters
    ----------
    axs : mpl.axes.Axes
        axes instance whose limits are to be adjusted
    data : np.ndarray (n)
        array of numbers plotted along the axis
    err : None or np.ndarray (n) or (2,n), optional
        error bars for data, default: None
    yaxis : bool, optional keyword
        are we modifying the y axis? default: True
    reset : bool, optional keyword
        do we ignore the existing axis limits? default: False
    log : bool, optional keyword
        is it a log scale? default: False
    buffer : float, optional keyword
        fractional padding around data
    """
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
    """Make axes look prettier

    All non-font size kewwords default to `True`.
    This can be changed with the keyword `all`.

    Parameters
    ----------
    axs : plt.Axes
        Axes object to modify
    fontsize : number, str, default: 20
        Font size for axes labels, ticks & title.
    fontfamily : str, default: sans-serif
        Font family for axes labels, ticks & title.

    Keyword only
    ------------
    box : bool, keyword
        Remove axes box?
    axisfont : bool, keyword only
        Change axes font size?
    titlefont : bool, keyword only
        Change title font size?
    legendbox : bool, keyword only
        Remove legend box?
    legendfont : bool, keyword only
        Change legend font size?
    tickfont : bool, keyword only
        Change tick-label font size?
    tight : bool, keyword only
        Apply tight_layout to figure?
    all : bool, keyword only
        Choice for any of the above that is unspecified, default: True
    titlefontsize : number, str
        Font size for title, default: `fontsize * titlefontscale`.
    legendfontsize : number, str
        Font size for legend entries, default: `fontsize * legendfontscale`.
    tickfontsize : number, str
        Font size for tick-labels, default: `fontsize * tickfontscale`.
    titlefontscale : number
        Multiplier of `fontsize` (if numeric) for title, default: 1.2.
    legendfontsize : number
        Multiplier of `fontsize` (if numeric) for legend entries, default: 1.
    tickfontsize : number
        Multiplier of `fontsize` (if numeric) for tick-labels, default: 0.694.
    """
    clean_kws = clean_axes_keys(kwds, fontsize)
    if axs is None:
        axs = plt.gca()
    if clean_kws['box']:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
    if clean_kws['titlefont']:
        axs.title.set_fontsize(clean_kws['titlefontsize'])
        axs.title.set_fontfamily(fontfamily)
    if clean_kws['axisfont']:
        axs.get_xaxis().get_label().set_fontsize(fontsize)
        axs.get_yaxis().get_label().set_fontsize(fontsize)
        axs.get_xaxis().get_label().set_fontfamily(fontfamily)
        axs.get_yaxis().get_label().set_fontfamily(fontfamily)
    if clean_kws['tickfont']:
        axs.tick_params(labelsize=clean_kws['tickfontsize'])
        for tkl in axs.get_xticklabels() + axs.get_yticklabels():
            tkl.set_fontfamily(fontfamily)
    if axs.legend_ is not None:
        if clean_kws['legendbox']:
            axs.legend_.set_frame_on(False)
        if clean_kws['legendfont']:
            adjust_legend_font(axs.legend_, size=clean_kws['legendfontsize'],
                               family=fontfamily)
    axs.set(**kwds)
    if clean_kws['tight']:
        axs.figure.tight_layout()


def clean_axes_keys(kwargs: _ty.Dict[str, _ty.Any],
                    fontsize: _ty.Union[int, str] = 20
                    ) -> _ty.Dict[str, _ty.Union[bool, int]]:
    """Extract keywords applicable to clean_axes

    Parameters
    ----------
    kwargs : Dict[str, _ty.Any]
        Dictionary of keyword options. Those applicable to `clean_axes` will
        be popped.

    Returns
    -------
    clean_kws: Dict[str, Union[bool, int]]
        Dictionary of keyword options used by `clean_axes`
    """
    clean_kws = {}
    if isinstance(fontsize, (int, float)):
        clean_kws['titlefontsize'] = kwargs.pop(
            'titlefontsize', round(fontsize * kwargs.pop('titlefontscale', 1.2)))
        clean_kws['legendfontsize'] = kwargs.pop(
            'legendfontsize', round(fontsize * kwargs.pop('legendfontscale', 1)))
        clean_kws['tickfontsize'] = kwargs.pop(
            'tickfontsize', round(fontsize * kwargs.pop('tickfontscale', .694)))
    else:
        clean_kws['titlefontsize'] = kwargs.pop('titlefontsize', fontsize)
        clean_kws['legendfontsize'] = kwargs.pop('legendfontsize', fontsize)
        clean_kws['tickfontsize'] = kwargs.pop('tickfontsize', fontsize)

    allopts = kwargs.pop('all', True)
    for key in ('box', 'axisfont', 'titlefont', 'tickfont', 'legendbox',
                'legendfont', 'tight'):
        clean_kws[key] = kwargs.pop(key, allopts)
    return clean_kws


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
                  centrex=0, centrey=0, **kwds):
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
                Use both coordinates of centre as 'x,y'.
            paren:
                Use both coordinates of centre as '(x,y)'.
            none:
                Do not label.
        Any other value interpreted as 'none', default: 'both'.
    """
    axs = _ag.default_eval(axs, plt.gca)
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
    lab = []
    for axis, centre in zip([axs.xaxis, axs.yaxis], [centrex, centrey]):
        # Turn ticks
        axis.set_ticks_position('both')

        # Hide the ticklabels at <centrex, centrey>
        formatter = CentredFormatter(centre)
        axis.set_major_formatter(formatter)
        lab.append(formatter.format_data(centre))

    # Add offset ticklabels at <centrex, centrey> using annotation
    # (Should probably make these update when the plot is redrawn...)
    centre_tick = kwds.pop('centre_tick', 'both').lower()  # {both,x,y,none}
    centre_labs = {'none': "", 'x': f"{lab[0]}", 'y': f"{lab[1]}",
                   'both': ",".join(lab), 'paren': f"({lab[0]},{lab[1]})"}
    centre_label = centre_labs.get(centre_tick, "")
    if centre_label:
        axs.annotate(centre_label, (centrex, centrey), xytext=(-4, -4),
                     textcoords='offset points', ha='right', va='top')


def add_axes_arrows(axs: _ty.Optional[mpl.axes.Axes] = None,
                    to_xaxis: bool = True, to_yaxis: bool = True):
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
    if axs is None:
        axs = plt.gca()
    # Draw an arrow at the end of the spines
    if to_xaxis:
        axs.spines['left'].set_path_effects([EndArrow()])
    if to_yaxis:
        axs.spines['bottom'].set_path_effects([EndArrow()])


# =============================================================================
# Lines
# =============================================================================


def plot_equality(axs: mpl.axes.Axes,
                  line: _ty.Optional[mpl.lines.Line2D] = None,
                  npt=2, **kwds):
    """Plot the equality line on Axes

    Parameters
    ----------
    axs : mpl.axes.Axes
        Axes on which we draw the equality line
    line : mpl.lines.Line2D or None, optional
        Previous line object, default: None
    npt : int, optional
        number of points to use for equality line
    **kwds
        Passed to `plt.plot`.

    See Also
    --------
    matplotlib.axes.Axes.axline([0, 0], [1, 1])
    """
    xlim = axs.get_xlim()
    ylim = axs.get_ylim()
    eq_vals = np.linspace(max(xlim[0], ylim[0]), min(xlim[1], ylim[1]), npt)
    if line is None:
        kwds.setdefault('label', "equality")
        kwds.setdefault('color', "k")
        kwds.setdefault('linestyle', "-")
        line = axs.plot(eq_vals, eq_vals, **kwds)[0]
    else:
        line.set_xdata(eq_vals)
        line.set_ydata(eq_vals)
        line.update(kwds)
    return line


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


# =============================================================================
# Options classes
# =============================================================================


# pylint: disable=too-many-ancestors
class ImageOptions(op.AnyOptions):
    """Options for heatmaps

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    Parameters
    ----------
    cmap : str|Colormap
        Colour map used to map numbers to colours. By default, `'YlOrBr'`.
    norm : Normalize
        Mapsheatmap values to interval `[0, 1]` for `cmap`.
        By default: `Normalise(0, 1)`.
    vmin : float
        Lower bound of `norm`. By default: `0`.
    vmax : float
        Lower bound of `norm`. By default: `1`.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    prop_attributes: op.Attrs = ('cmap',)
    _cmap: mpl.colors.Colormap
    norm: mpl.colors.Normalize

    def __init__(self, *args, **kwds) -> None:
        self._cmap = mpl.cm.get_cmap('YlOrBr')
        self.norm = mpl.colors.Normalize(0., 1.)
        super().__init__(*args, **kwds)

    @property
    def cmap(self) -> mpl.colors.Colormap:
        """Get the colour map.
        """
        return self._cmap

    def set_cmap(self, value: _ty.Union[str, mpl.colors.Colormap]) -> None:
        """Set the colour map.

        Does noting if `value` is `None`. Converts to `Colormap` if `str`.
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

    def set_vmin(self, value: float) -> None:
        """Set the lower bound for the colour map.

        Does noting if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.norm.vmin = value

    def set_vmax(self, value: float) -> None:
        """Set the upper bound for the colour map.

        Does noting if `value` is `None`.
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
class AnimationOptions(op.AnyOptions):
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
    interval: float
    repeat_delay: float
    repeat: bool
    blit: bool

    def __init__(self, *args, **kwds) -> None:
        self.interval = 500
        self.repeat_delay = 1000
        self.repeat = True
        self.blit = False
        super().__init__(*args, **kwds)
# pylint: enable=too-many-ancestors


# =============================================================================
# Helper classes
# =============================================================================


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
        return super().__call__(value, pos)


# Note: I'm implementing the arrows as a path effect rather than a custom
#       Spines class. In the long run, a custom Spines class would be a better
#       way to go. One of the side effects of this is that the arrows aren't
#       reversed when the axes are reversed! (Joe Kington)


class EndArrow(mpf.AbstractPathEffect):
    """A matplotlib patheffect to add arrows at the end of a path.

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
    """

    def __init__(self, headwidth=5, headheight=5, facecolor=(0, 0, 0), **kwds):
        super().__init__()
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

        x_0, y_0 = tpath.vertices[-1]
        d_x, d_y = tpath.vertices[-1] - tpath.vertices[-2]
        azi = np.arctan2(d_y, d_x) - np.pi / 2.0
        trans = affine + self.trans.clear(
                        ).scale(scalex, scaley).rotate(azi).translate(x_0, y_0)

        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)
        self._update_gc(gc0, self._gc_args)

        if self.facecolor is not None:
            rgbFace = self.facecolor

        renderer.draw_path(gc0, self.arrowpath, trans, rgbFace)
        renderer.draw_path(gc, tpath, affine, rgbFace)
        gc0.restore()
