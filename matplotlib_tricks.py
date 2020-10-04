# -*- coding: utf-8 -*-
"""Tools for making matplotlib nicer
"""
from __future__ import annotations
import typing as _ty
import functools as _ft

import matplotlib as mpl
import matplotlib.animation as mpa
import matplotlib.backends.backend_pdf as pdf
import matplotlib.pyplot as plt
import numpy as np

import sl_py_tools.arg_tricks as _ag
import sl_py_tools.containers as _cn
import sl_py_tools.tol_colors as tol
import sl_py_tools.options_classes as op

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
    mpl.rcParams['text.latex.preamble'] = "\\usepackage{amsmath,amssymb}"
    if family == 'sans-serif':
        mpl.rcParams['text.latex.preamble'] += r"\usepackage{euler}"
        mpl.rcParams['text.latex.preamble'] += r"\usepackage{sansmath}"
        mpl.rcParams['text.latex.preamble'] += r"\sansmath"


def rc_colours(cset: str = 'bright', cmap: str = 'YlOrBr',
               reg: _ty.Tuple[str, ...] = ()) -> None:
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


def fig_square(fig: mpl.figure.Figure) -> None:
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


def equal_axlim(axs: mpl.axes.Axes, mode: str = 'union') -> None:
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
                  yaxis: bool = True, reset: bool = False, log: bool = False,
                  buffer: float = 0.05):
    """Set axes limits that will show all data, including existing

    Parameters
    ----------
    axs : mpl.axes.Axes
        axes instance whose limits are to be adjusted
    data : np.ndarray (n)
        array of numbers plotted along the axis
    err : None or np.ndarray (n) or (2,n), optional
        error bars for data, by default: `None`.
    yaxis : bool, optional keyword
        are we modifying the y axis? By default: `True`.
    reset : bool, optional keyword
        do we ignore the existing axis limits? By default: `False`.
    log : bool, optional keyword
        is it a log scale? By default: `False`.
    buffer : float, optional keyword
        fractional padding around data, by default `0.05`.
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
    legendfontscale : number
        Multiplier of `fontsize` (if numeric) for legend entries, default: 1.
    tickfontscale : number
        Multiplier of `fontsize` (if numeric) for tick-labels, default: 0.694.
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
    xformatter = CentredFormatter(centrex, '')
    yformatter = CentredFormatter(centrey, '')
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
        axs.spines['bottom'] = FancyArrowSpine(axs.spines['bottom'], **kwds)
    if to_yaxis:
        axs.spines['left'] = FancyArrowSpine(axs.spines['left'], **kwds)


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


class FontSize:
    """Font size property.

    Parameters
    ----------
    boolean : bool, optional
        Default value for boolean property, by default `True`.
    scale : float, optional
        Default value for scale property, by default `1`.
    doc : str|None, optional
        Docstring, by default `None`
    """
    name: str
    _name: str
    _bool_default: bool
    _scale_default: float

    def __init__(self, func: _ty.Optional[TextGetter] = None, *,
                 boolean: bool = True, scale: float = 1.,
                 doc: _ty.Optional[str] = None) -> None:
        self.func = func
        self.__doc__ = doc or func.__doc__
        self.name = func.__name__
        self._name = '_' + func.__name__
        self._bool_default = boolean
        self._scale_default = scale

    def prepare(self, obj: AxesOptions) -> None:
        """Call in __new__ of owner"""
        obj[self._name] = {'bool': self._bool_default,
                           'scale': self._scale_default, 'size': None}

    def __get__(self, obj: AxesOptions, objtype: OwnerType = None) -> bool:
        if obj is None:
            return self

        @_ft.wraps(self.func)
        def method(axs: mpl.axes.Axes) -> None:
            """modify fonts"""
            for txt in self.func(obj, axs):
                txt.set_fontsize(obj[self.name + 'fontsize'])
                txt.set_fontfamily(obj.fontfamily)
        return method

    def prop(self, field: str, name: _ty.Optional[str] = None) -> property:
        """Accessor property for `field``"""

        def fget(obj: AxesOptions) -> float:
            return obj[self._name][field]

        def fset(obj: AxesOptions, value: float) -> None:
            obj[self._name][field] = value

        self._set_propname(name or 'font' + field, fget, fset)
        return property(fget, fset, None, self.__doc__)

    def size(self) -> property:
        """Accessor property for _size"""

        def fget(obj: AxesOptions) -> Size:
            if obj[self._name]['size'] is not None:
                return obj[self._name]['size']
            if isinstance(obj.fontsize, str):
                return obj.fontsize
            return obj.fontsize * obj[self._name]['scale']

        def fset(obj: AxesOptions, value: Size) -> None:
            if isinstance(value, str):
                obj[self._name]['size'] = value
            else:
                obj[self._name]['size'] = None
                obj[self._name]['scale'] = value / obj.fontsize

        self._set_propname('fontsize', fget, fset)
        return property(fget, fset, None, self.__doc__)

    def props(self) -> _ty.Tuple[property, ...]:
        """The properties for access"""
        return self.prop('bool', 'font'), self.size(), self.prop('scale')

    def _set_propname(self, name: str, *fns) -> None:
        """Set the __name__ and __qualname__ of a property"""
        for fdo in fns:
            fdo.__name__ = self.func.__name__ + name
            fdo.__qualname__ = self.func.__qualname__ + name


def fontsize_manager(func: _ty.Optional[TextGetter] = None, *,
                     boolean: bool = True, scale: float = 1.,
                     doc: _ty.Optional[str] = None) -> FontSize:
    """decorate a method to turn it into a FontSize descriptor"""
    if func is None:
        return _ft.partial(FontSize, boolean=boolean, scale=scale, doc=doc)
    return FontSize(func, boolean=boolean, scale=scale, doc=doc)


# pylint: disable=too-many-ancestors,too-many-instance-attributes
class AxesOptions(op.AnyOptions):
    """Options for `clean_axes`.

    The individual options can be accessed as object instance attributes
    (e.g. `obj.name`) or as dictionary items (e.g. `obj['name']`) for both
    getting and setting.

    Parameters
    ----------
    box : bool, keyword
        Remove axes box?
    legendbox : bool, keyword only
        Remove legend box?
    tight : bool, keyword only
        Apply tight_layout to figure?
    <element>font : bool, keyword only
        Change <element> font size? Where <element> is:-
            `axis`: Axis labels, by default `True`, default `scale`: `1`.
            `title`: Axes title, by default `True`, default `scale`: `1.2`.
            `legend`: Legend entries, by default `True`, default `scale`: `1`.
            `tick`: Tick labels, by default `True`, default `scale`: `0.694`.
    all : bool, keyword only
        Choice for any of the above that is unspecified, default: True
    <element>fontsize : number, str
        Font size for <element>, default: `fontsize * <element>fontscale`.
    <element>fontscale : number
        Multiplier of `fontsize` (if numeric) for <element>.
    fontsize : number, str, default: 20
        Base font size for axes labels, ticks, legends & title.
    fontfamily : str, default: sans-serif
        Font family for axes labels, ticks, legends & title.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    prop_attributes: op.Attrs = ('axisfont', 'titlefont', 'legendfont',
                                 'tickfont', 'axisfontsize', 'titlefontsize',
                                 'legendfontsize', 'tickfontsize')
    key_first: op.Attrs = ('all', 'fontsize')

    fontsize : Size
    fontfamily : str
    box : bool
    legendbox : bool
    tight : bool

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
        self.fontsize = 20
        self.fontfamily = "sans-serif"
        self.box = True
        self.legendbox = True
        self.tight = True
        super().__init__(*args, **kwds)

    @fontsize_manager(scale=1.)
    def axis(self, axs: mpl.axes.Axes) -> None:
        """Modify font of axis-labels"""
        if not self.axisfont:
            return ()
        return (axs.get_xaxis().get_label(), axs.get_yaxis().get_label())

    axis = _ty.cast(FontSize, axis)
    axisfont, axisfontsize, axisfontscale = axis.props()

    @fontsize_manager(scale=1.2)
    def title(self, axs: mpl.axes.Axes) -> None:
        """Modify font of axis-labels"""
        if not self.titlefont:
            return ()
        return (axs.title,)

    title = _ty.cast(FontSize, title)
    titlefont, titlefontsize, titlefontscale = title.props()

    @fontsize_manager(scale=1.)
    def legend(self, axs: mpl.axes.Axes) -> None:
        """Modify font of legend-labels"""
        if axs.legend_ is None or not self.legendfont:
            return ()
        return axs.legend_.get_texts()

    legend = _ty.cast(FontSize, legend)
    legendfont, legendfontsize, legendfontscale = legend.props()

    @fontsize_manager(scale=1.)
    def tick(self, axs: mpl.axes.Axes) -> None:
        """Modify font of tick-labels"""
        if not self.tickfont:
            return ()
        return axs.get_xticklabels() + axs.get_yticklabels()

    tick = _ty.cast(FontSize, tick)
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

    def set_vmin(self, value: float) -> None:
        """Set the lower bound for the colour map.

        Does nothing if `value` is `None`.
        """
        if value is None:
            pass
        else:
            self.norm.vmin = value

    def set_vmax(self, value: float) -> None:
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
# Animation writers
# =============================================================================


@mpa.writers.register('file_seq')
class FileSeqWriter(mpa.FileMovieWriter):
    """Write an animation as a sequence of image files.
    """
    supported_formats: _ty.ClassVar[_ty.List[str]] = [
        'pdf', 'svg', 'png', 'jpeg', 'ppm', 'tiff', 'sgi', 'bmp', 'pbm', 'raw',
    ]
    fname_format_str : str
    _ndigit: int = 7

    def setup(self,  # pylint: disable=arguments-differ
              fig: mpl.figure.Figure, outfile: str,
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
        if '.' in outfile:
            frame_prefix, self.frame_format = outfile.rsplit('.', 1)
        else:
            frame_prefix = outfile
        super().setup(fig, outfile, dpi, frame_prefix=frame_prefix)
        if ndigit is not None:
            self._ndigit = ndigit
        self.fname_format_str = f'%s%%0{ndigit}d.%s'

    def cleanup(self) -> None:
        """Perform cleanup - nothing!"""

    def _run(self) -> None:
        """Perform post-processing - nothing!"""

    def _args(self) -> _ty.List[str]:
        """External command - nothing!"""
        return []

    @classmethod
    def isAvailable(cls) -> bool:
        return True


@mpa.writers.register('pdf_pages')
class PdfPagesWriter(mpa.AbstractMovieWriter):
    """Write animation as a multi-page pdf file"""
    supported_formats: _ty.ClassVar[_ty.List[str]] = ['pdf']
    _file: _ty.Optional[pdf.PdfPages]

    def __init__(self, fps=5, codec=None, bitrate=None, metadata=None) -> None:
        """
        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        metadata : Dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        """
        super().__init__(fps=fps, metadata=metadata, codec=codec,
                         bitrate=bitrate)
        self.frame_format = 'pdf'
        self._file = None

    def setup(self, fig, outfile, dpi=None):
        """Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        """
        super().setup(fig, outfile, dpi=dpi)
        self._file = pdf.PdfPages(outfile, metadata=self.metadata)

    def grab_frame(self, **savefig_kwargs):
        """
        Grab the image information from the figure and save as a movie frame.

        All keyword arguments in *savefig_kwargs* are passed on to the
        `~.Figure.savefig` call that saves the figure.
        """
        self._file.savefig(self.fig, **savefig_kwargs)

    def finish(self):
        """Finish any processing for writing the movie."""
        self._file.close()

    @classmethod
    def isAvailable(cls) -> bool:  # pylint: disable=invalid-name
        """Make sure the registry knows we're available"""
        return True


# =============================================================================
# Helper classes for axes
# =============================================================================


class CentredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "centre".

    From: https://stackoverflow.com/a/4718438/9151228 by Joe Kington
    """
    centre : float
    label: str

    def __init__(self, centre: float = 0, label: str = '', **kwds):
        super().__init__(**kwds)
        self.centre = centre
        self.label = label

    def __call__(self, value: float, pos=None):
        if value == self.centre:
            return self.label
        return super().__call__(value, pos)


class FancyArrowSpine(mpl.spines.Spine, mpl.patches.FancyArrowPatch):
    """Spine with an arrow
    """

    def __init__(self, other: mpl.spines.Spine, **kwds) -> None:
        kwds.update(posA=[0, 0], posB=[1, 1])
        kwds.setdefault('arrowstyle', "-|>")
        kwds.setdefault('mutation_scale', 10)
        path = other.get_path()
        super().__init__(other.axes, other.spine_type, path, **kwds)
        self.update_from(other)
        self._path_original = path
        self._posA_posB = None
        self.register_axis(other.axis)
        self.set_position(other.get_position())

    def draw(self, renderer):
        """Make sure we call the right draw"""
        self._adjust_location()
        return mpl.patches.FancyArrowPatch.draw(self, renderer)


# =============================================================================
# Aliases
# =============================================================================
Var = _ty.TypeVar('Var')
Val = _ty.TypeVar('Val')
Owner = _ty.TypeVar('Owner')
Size = _ty.Union[int, float, str]
OwnerType = _ty.Optional[_ty.Type[Owner]]
TextGetter = _ty.Callable[[AxesOptions, mpl.axes.Axes],
                          _ty.Iterable[mpl.text.Text]]
