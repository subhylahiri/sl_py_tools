# -*- coding: utf-8 -*-
"""Tools for working with graphs and plotting them
"""
from __future__ import annotations

import typing as ty
from numbers import Number
from typing import Callable, Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import sl_py_tools.arg_tricks as ag
import sl_py_tools.containers as cn
import sl_py_tools.graph_tricks as gt
import sl_py_tools.matplotlib_tricks as mpt
import sl_py_tools.options_classes as op
from sl_py_tools.graph_tricks import ArrayLike, Edge, GraphAttrs
from sl_py_tools.numpy_tricks.markov import TopologyOptions

# =============================================================================
# Options
# =============================================================================


# pylint: disable=too-many-ancestors
class ColourOptions(mpt.ImageOptions):
    """Options for node/edge colours when drawing graphs.

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
    attr : str
        Name of node/edge attribute used to determine node area/edge width.
    entity : str
        Type of graph element, 'node' or 'edge'

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    map_attributes: op.Attrs = ()
    prop_attributes: op.Attrs = ()
    # topology specifying options
    attr: str
    entity: str

    def __init__(self, *args, **kwds) -> None:
        self.attr = 'key'
        self.entity = 'node'
        super().__init__(*args, **kwds)

    def to_colour(self, graph: GraphAttrs) -> np.ndarray:
        """Get sizes from graph attributes

        Parameters
        ----------
        graph : GraphAttrs
            The graph whose edge/node attributes set node area/edge width.

        Returns
        -------
        sizes : np.ndarray
            Array of node areas/edge widths.
        """
        vals = getattr(graph, f"get_{self.entity}_attr")(self.attr)
        self.set_vmin(vals.min())
        self.set_vmax(vals.max())
        return self.val_to_colour(vals)
# pylint: enable=too-many-ancestors

# pylint: disable=too-many-ancestors
class SizeOptions(op.Options):
    """Options for node/edge sizes when drawing graphs.

    Parameters
    ----------
    attr : str
        Name of node/edge attribute used to determine node area/edge width.
    size : float
        Scale factor between `node/edge[attr]` and node area/edge width.
    entity : str
        Type of graph element, 'node' or 'edge'

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    map_attributes: op.Attrs = ()
    prop_attributes: op.Attrs = ()
    attr: str
    mult: float
    entity: str

    def __init__(self, *args, **kwds) -> None:
        self.attr = 'value'
        self.mult = 1.
        self.entity = 'node'
        super().__init__(*args, **kwds)

    def to_size(self, graph: GraphAttrs) -> np.ndarray:
        """Get sizes from graph attributes

        Parameters
        ----------
        graph : GraphAttrs
            The graph whose edge/node attributes set node area/edge width.

        Returns
        -------
        sizes : np.ndarray
            Array of node areas/edge widths.
        """
        return getattr(graph, f"get_{self.entity}_attr")(self.attr) * self.mult
# pylint: enable=too-many-ancestors


# pylint: disable=too-many-ancestors
class GraphOptions(op.Options):
    """Options for drawing graphs.

    Parameters
    ----------
    topology : TopologyOptions
        Topology specifying options, for creating graphs/reference for `judge`.
    layout : Callable[DiGraph -> Dict[Node, ArrayLike]]
        Function to compute node positions.
    node_colours : ImageOptions
        Options for mapping `node[attr]` to node colour..
    edge_colours : ImageOptions
        Options for mapping `edge[attr]` to edge colour.
    node_size : float
        Options for mapping `node[attr]` to node area.
    edge_size : float
        Options for mapping `edge[attr]` to edge thickness.
    rad : List[float]
        Curvature of edges: ratio betweeen max perpendicular distance from
        straight line to curve and length of straight line. Positive ->
        counter-clockwise. The list members are for [good, bad] directions.
    judge : Callable[[graph, toplogy] -> ndarray[bool]]
        Function that decides which edges are good and which are bad.

    All parameters are optional keywords. Any dictionary passed as positional
    parameters will be popped for the relevant items. Keyword parameters must
    be valid keys, otherwise a `KeyError` is raised.
    """
    map_attributes: op.Attrs = ('topology', 'node_colours', 'edge_colours')
    prop_attributes: op.Attrs = ()
    # topology specifying options
    topology: TopologyOptions
    layout: Layout
    node_colours: ColourOptions
    edge_colours: ColourOptions
    node_size: SizeOptions
    edge_size: SizeOptions
    rad: ty.List[float]
    judge: Optional[Judger]

    def __init__(self, *args, **kwds) -> None:
        self.topology = TopologyOptions(serial=True)
        self.layout = linear_layout
        self.node_size = SizeOptions(entity="node", mult=600)
        self.edge_size = SizeOptions(entity="edge", mult=5)
        self.rad = [-0.7, 0.35]
        self.judge = good_direction
        self.node_colours = ColourOptions(entity="node", cmap='coolwarm')
        self.edge_colours = ColourOptions(entity="edge", cmap='seismic')
        super().__init__(*args, **kwds)

    def choose_rads(self, graph: gt.MultiDiGraph) -> np.ndarray:
        """Choose curvature of each edge"""
        if self.judge is None:
            good_drn = np.ones(len(graph.edges), bool)
        else:
            good_drn = self.judge(graph, self.topology)
        return np.where(good_drn, *self.rad)
# pylint: enable=too-many-ancestors


# =============================================================================
# Plot graph
# =============================================================================


def get_node_colours(graph: GraphAttrs, data: str) -> Dict[str, np.ndarray]:
    """Collect values of node attributes for the colour

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with nodes whose attributes we want.
    data : str
        Name of attribute to map to colour.

    Returns
    -------
    kwargs : Dict[str, np.ndarray]
        Dictionary of keyword arguments for `nx.draw_networkx_nodes` related to
        colour values: `{'node_color', 'vmin', 'vmax'}`.
    """
    vals = graph.get_node_attr(data)
    vmin, vmax = vals.min(), vals.max()
    return {'node_color': vals, 'vmin': vmin, 'vmax': vmax}


def get_edge_colours(graph: GraphAttrs, data: str) -> Dict[str, np.ndarray]:
    """Collect values of edge attributes for the colour

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with edges whose attributes we want.
    data : str
        Aattribute mapped to colour. Ignored if `graph` is a `MultiDiGraph`.

    Returns
    -------
    kwargs : Dict[str, np.ndarray]
        Dictionary of keyword arguments for `nx.draw_networkx_edges` related to
        colour values: `{'edge_color', 'edge_vmin', 'edge_vmax'}`.
    """
    if isinstance(graph, gt.MultiDiGraph) and data == 'key':
        vals = graph.edge_key()
    else:
        vals = graph.get_edge_attr(data)
    vmin, vmax = vals.min(), vals.max()
    return {'edge_color': vals, 'edge_vmin': vmin, 'edge_vmax': vmax}


def linear_layout(graph: nx.Graph, sep: ArrayLike = (1., 0.)) -> NodePos:
    """Layout graph nodes in a line.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph whose nodes need laying out.
    sep : ArrayLike, optional
        Separation of nodes along line, by default `(1.0, 0.0)`.

    Returns
    -------
    pos : Dict[Node, np.ndarray]
        Dictionary of node ids -> position vectors.
    """
    sep = np.asarray(sep)
    return {node: pos * sep for pos, node in enumerate(graph.nodes)}


def good_direction(graph: gt.MultiDiGraph, ideal: TopologyOptions) -> np.ndarray:
    """Which edges are in a good direction?

    Parameters
    ----------
    graph : MultiDiGraph, (N,E)
        The graph whose edges we're testing.
    ideal : TopologyOptions
        Description of the reference topology, which defines good directions.

    Returns
    -------
    good : np.ndarray[bool] (E,)
        True if the direction of the edge is similar to `ideal`.
    """
    edges = np.array(graph.edge_order)
    _, key_inds = gt.list_edge_keys(graph, True)
    best_drn = np.array(ideal.directions)[key_inds]
    real_drn = edges[:, 1] - edges[:, 0]
    if ideal.ring:
        num = len(graph.nodes)
        real_drn = (real_drn + num/2) % num - num/2
    return real_drn * best_drn >= 0


# =============================================================================
# Edge collection
# =============================================================================


class NodeCollection:
    """A collection of node plots"""
    _nodes: NodePlots
    _node_ids: ty.List[gt.Node]
    colours: ColourOptions
    size: SizeOptions
    # actual sizes, after scaling by size.mult
    node_pos: NodePos
    node_size: np.ndarray

    def __init__(self, graph: GraphAttrs, pos: Optional[NodePos] = None,
                 axs: Optional[mpl.axes.Axes] = None,
                 opts: Optional[GraphOptions] = None, **kwds) -> None:
        self._node_ids = list(graph.nodes)
        opts = ag.default_eval(opts, GraphOptions)
        axs = ag.default_eval(axs, plt.gca)
        opts.pop_my_args(kwds)
        self.colours = opts.node_colours
        self.size = opts.node_size
        self.node_pos = ag.default(pos, opts.layout)
        if callable(self.node_pos):
            self.node_pos = self.node_pos(graph)

        self.node_size = self.size.to_size(graph)
        node_col = self.colours.to_colour(graph)

        kwds.update(ax=axs, node_color=node_col, node_size=self.node_size,
                    edgecolors='k')
        self._nodes = nx.draw_networkx_nodes(graph, self.node_pos, **kwds)

    def set_color(self, col_vals: gt.ArrayLike) -> None:
        """Set node colour valuess"""
        cols = self.colours.val_to_colour(col_vals)
        self._nodes.set_color(cols)

    def set_sizes(self, node_siz: ArrayLike) -> None:
        """Set node sizes"""
        self.node_size = np.asarray(node_siz) * self.size.mult
        self._nodes.set_sizes(self.node_size)

    def set_pos(self, pos: NodePos) -> None:
        """Set positions of of nodes"""
        self.node_pos = pos
        pos_array = np.array([pos[node] for node in self._node_ids])
        self._nodes.set_offsets(pos_array)

    def get_sizes(self) -> np.ndarray:
        """Get node sizes"""
        return self.node_size
        # return self._nodes.get_sizes()

    def get_pos(self) -> NodePos:
        """Get node sizes"""
        return self.node_pos
        # return dict(zip(self._node_ids, self._nodes.get_offsets()))


class DiEdgeCollection:
    """A collection of directed edge plots"""
    _edges: Dict[Edge, EdgePlot]
    _node_ids: ty.List[gt.Node]
    colours: ColourOptions
    size: SizeOptions

    def __init__(self, graph: GraphAttrs, nodes: NodeCollection,
                 axs: Optional[mpl.axes.Axes] = None,
                 opts: Optional[GraphOptions] = None, **kwds) -> None:
        self._node_ids = list(graph.nodes)
        opts = ag.default_eval(opts, GraphOptions)
        axs = ag.default_eval(axs, plt.gca)
        opts.pop_my_args(kwds)
        self.colours = opts.edge_colours
        self.size = opts.edge_size

        edge_wid = self.size.to_size(graph)
        edge_col = self.colours.to_colour(graph)

        kwds.update(ax=axs, edge_color=edge_col, width=edge_wid,
                    node_size=nodes.get_sizes(),
                    connectionstyle=f'arc3,rad={opts.rad[0]}')
        edges = nx.draw_networkx_edges(graph, nodes.get_pos(), **kwds)
        self._edges = dict(zip(graph.edges, edges))
        self.set_rads(opts.choose_rads(graph))

    def __len__(self) -> int:
        return len(self._edges)

    def __getitem__(self, key: Edge) -> EdgePlot:
        return self._edges[key]

    def __iter__(self) -> ty.Iterable[Edge]:
        return iter(self._edges)

    def keys(self) -> ty.Iterable[Edge]:
        """A view of edge dictionary keys"""
        return self._edges.keys()

    def values(self) -> ty.Iterable[EdgePlot]:
        """A view of edge dictionary values"""
        return self._edges.values()

    def items(self) -> ty.Iterable[Tuple[Edge, EdgePlot]]:
        """A view of edge dictionary items"""
        return self._edges.items()

    def set_color(self, col_vals: gt.ArrayLike) -> None:
        """Set line colour values"""
        cols = self.colours.val_to_colour(col_vals)
        cols = mpl.colors.to_rgba_array(cols)
        cols = np.broadcast_to(cols, (len(self), 4), True)
        for edge, col in zip(self.values(), cols):
            edge.set_color(col)

    def set_widths(self, edge_vals: ArrayLike) -> None:
        """Set line widths of edges
        edge_vals : ArrayLike
            Values used to set widths, without scaling by `mult`.
        """
        edge_vals = np.broadcast_to(edge_vals, (len(self),), True)
        for edge, wid in zip(self.values(), edge_vals * self.size.mult):
            edge.set_linewidth(wid)

    def set_node_sizes(self, node_siz: ArrayLike) -> None:
        """Set sizes of nodes
        node_siz : ArrayLike
            Node size, already scaled by mult
        """
        siz_dict = dict(zip(self._node_ids, cn.repeatify(node_siz)))
        for edge, edge_plot in self.items():
            edge_plot.shrinkA = _to_marker_edge(siz_dict[edge[0]], 'o')
            edge_plot.shrinkB = _to_marker_edge(siz_dict[edge[1]], 'o')

    def set_node_pos(self, pos: NodePos) -> None:
        """Set positions of of nodes"""
        for edge_id, edge_plot in self.items():
            edge_plot.set_position(pos[edge_id[0]], pos[edge_id[1]])

    def set_rads(self, rads: ArrayLike) -> None:
        """Set the curvature of the edges"""
        rads = np.broadcast_to(np.asanyarray(rads).ravel(), (len(self),), True)
        for edge, rad in zip(self.values(), rads):
            edge.set_connectionstyle('arc3', rad=rad)


def _to_marker_edge(marker_size: Number, marker: str) -> Number:
    """Space to leave for node at end of fancy arrrow patch"""
    if marker in "s^>v<d":  # `large` markers need extra space
        return np.sqrt(2 * marker_size) / 2
    return np.sqrt(marker_size) / 2


class GraphPlots:
    """Class for plotting model as a graph.

    Parameters
    ----------
    graph : DiGraph
        Graph object describing model. Nodes have attributes `key` and
        `value`. Edges have attributes `key`, `value` and `pind` (if the
        model was a `SynapseParamModel`).
    opts : GraphOptions|None, optional
        Options for plotting the graph, by default `None -> GraphOptions()`.
    Other keywords passed to `opt` or `nx.draw_networkx_nodes` and
    `nx.draw_networkx_edges`.
    """
    nodes: NodeCollection
    edges: DiEdgeCollection
    opts: GraphOptions

    def __init__(self, graph: GraphAttrs, pos: Optional[NodePos] = None,
                 axs: Optional[mpl.axes.Axes] = None,
                 opts: Optional[GraphOptions] = None, **kwds) -> None:
        """Class for plotting model as a graph.

        Parameters
        ----------
        graph : GraphAttrs
            Graph object describing model. Nodes and edges have attributes
            `key` and `value`.
        opt : GraphOptions|None, optional
            Options for drawing the graph, by default `None -> GraphOptions()`.
        Other keywords passed to `opt` or `nx.draw_networkx_nodes` and
        `nx.draw_networkx_edges`.
        """
        self.opts = ag.default_eval(opts, GraphOptions)
        self.opts.pop_my_args(kwds)
        axs = ag.default_eval(axs, plt.gca)

        self.nodes = NodeCollection(graph, pos, axs, self.opts, **kwds)
        self.edges = DiEdgeCollection(graph, self.nodes, axs, self.opts, **kwds)

    def update(self, edge_vals: Optional[np.ndarray],
               node_vals: Optional[np.ndarray]) -> None:
        """Update plots.

        Parameters
        ----------
        edge_vals : None|np.ndarray (E,)
            Edge line widdths.
        node_vals : None|np.ndarray (N,)
            Equilibrium distribution,for nodes sizes (area)
        """
        if edge_vals is not None:
            self.set_widths(edge_vals)
        if node_vals is not None:
            self.set_node_sizes(node_vals)

    def update_from(self, graph: GraphAttrs) -> None:
        """Update plots using a graph object.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph object describing model. Nodes have attributes `key` and
            `value`.  Edges have attributes `key`, `value`.
        """
        params = graph.get_edge_attr(self.opts.edge_size.attr)
        peq = graph.get_node_attr(self.opts.node_size.attr)
        self.update(params, peq)

    def set_node_colors(self, cols: gt.ArrayLike) -> None:
        """Set node colours"""
        self.nodes.set_color(cols)

    def set_edge_colors(self, cols: gt.ArrayLike) -> None:
        """Set edge colours"""
        self.edges.set_color(cols)

    def set_node_sizes(self, node_vals: ArrayLike) -> None:
        """Set node sizes"""
        self.nodes.set_sizes(node_vals)
        self.edges.set_node_sizes(self.nodes.get_sizes())

    def set_widths(self, edge_vals: ArrayLike) -> None:
        """Set edge sizes"""
        self.edges.set_widths(np.asarray(edge_vals).ravel())

    def set_node_pos(self, pos: NodePos) -> None:
        """Set positions of of nodes"""
        self.nodes.set_pos(pos)
        self.edges.set_node_pos(pos)

    def set_rads(self, rads: ArrayLike) -> None:
        """Set the curvature of the edges"""
        self.edges.set_rads(rads)


# =============================================================================
# Aliases
# =============================================================================
NodePlots = mpl.collections.PathCollection
EdgePlot = mpl.patches.FancyArrowPatch
NodePos = Dict[gt.Node, ArrayLike]
Layout = Callable[[nx.DiGraph], NodePos]
Colour = Union[str, ty.Sequence[float]]
Judger = Callable[[nx.Graph, TopologyOptions], np.ndarray]
