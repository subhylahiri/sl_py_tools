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
import sl_py_tools.numpy_tricks.markov as ma
import sl_py_tools.options_classes as op
from sl_py_tools.graph_tricks import (ArrayLike, ArrayLikeOf, Edge,
                                      GraphAttrMixin, MultiDiGraph, Node)
from sl_py_tools.numpy_tricks.logic import unique_unsorted

# import sl_py_tools.matplotlib_tricks as mpt
# =============================================================================
# Options
# =============================================================================


# pylint: disable=too-many-ancestors
class GraphOptions(op.Options):
    """Options for drawing graphs.

    Parameters
    ----------
    topology : TopologyOptions
        Topology specifying options, for creating graphs.
    layout : Callable[DiGraph -> Dict[Node, ArrayLike]]
        Function to compute node positions.
    node_cmap : str|mpl.colors.Colormap
        Map `node['key']` to node colour. `str` passed to `mpl.cm.get_cmap`.
    edge_cmap : str|mpl.colors.Colormap
        Map `edge['key']` to edge colour. `str` passed to `mpl.cm.get_cmap`.
    size : float
        Scale factor between `node['value']` and node area.
    width : float
        Scale factor between `edge['value']` and edge thickness.
    rad : float
        Curvature of edges: ratio betweeen max perpendicular distance from
        straight line to curve and length of straight line. Positive ->
        counter-clockwise.
    """
    map_attributes: op.Attrs = ('topology',)
    prop_attributes: op.Attrs = ('node_cmap', 'edge_cmap')
    # topology specifying options
    topology: ma.TopologyOptions
    layout: Layout
    _node_cmap: mpl.colors.Colormap
    _edge_cmap: mpl.colors.Colormap
    size: float
    width: float
    rad: float

    def __init__(self, *args, **kwds) -> None:
        self.topology = ma.TopologyOptions()
        self.layout = linear_layout
        self.size = 600
        self.width = 5
        self.rad = -0.7
        self._node_cmap = mpl.cm.get_cmap('coolwarm')
        self._edge_cmap = mpl.cm.get_cmap('seismic')
        super().__init__(*args, **kwds)

    def set_node_cmap(self, value: Union[str, mpl.colors.Colormap]) -> None:
        """Set the colour map for nodes.

        Does noting if `value` is `None`. Converts to `Colormap` if `str`.
        """
        if value is None:
            pass
        elif isinstance(value, str):
            self._node_cmap = mpl.cm.get_cmap(value)
        else:
            self._node_cmap = value

    def set_edge_cmap(self, value: Union[str, mpl.colors.Colormap]) -> None:
        """Set the colour map for edges.

        Does noting if `value` is `None`. Converts to `Colormap` if `str`.
        """
        if value is None:
            pass
        elif isinstance(value, str):
            self._edge_cmap = mpl.cm.get_cmap(value)
        else:
            self._edge_cmap = value

    @property
    def node_cmap(self) -> mpl.colors.Colormap:
        """Get the colour map for nodes.
        """
        return self._node_cmap

    @property
    def edge_cmap(self) -> mpl.colors.Colormap:
        """Get the colour map for nodes.
        """
        return self._edge_cmap
# pylint: enable=too-many-ancestors


# =============================================================================
# Plot graph
# =============================================================================


def get_node_colours(graph: GraphAttrMixin, key: str) -> Dict[str, np.ndarray]:
    """Collect values of node attributes for the colour

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with nodes whose attributes we want.
    key : str
        Name of attribute to map to colour.

    Returns
    -------
    kwargs : Dict[str, np.ndarray]
        Dictionary of keyword arguments for `nx.draw_networkx_nodes` related to
        colour values: `{'node_color', 'vmin', 'vmax'}`.
    """
    vals = graph.get_node_attr(key)
    vmin, vmax = vals.min(), vals.max()
    return {'node_color': vals, 'vmin': vmin, 'vmax': vmax}


def get_edge_colours(graph: GraphAttrMixin, key: str) -> Dict[str, np.ndarray]:
    """Collect values of edge attributes for the colour

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with edges whose attributes we want.
    key : str
        Name of attribute to map to colour.

    Returns
    -------
    kwargs : Dict[str, np.ndarray]
        Dictionary of keyword arguments for `nx.draw_networkx_edges` related to
        colour values: `{'edge_color', 'edge_vmin', 'edge_vmax'}`.
    """
    if isinstance(graph, MultiDiGraph):
        vals = graph.edge_key()
    else:
        vals = graph.get_edge_attr(key)
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
    sep = np.array(sep)
    return {node: pos * sep for pos, node in enumerate(graph.nodes)}


def good_direction(graph: MultiDiGraph, topology: ma.TopologyOptions):
    """Which edges are in a good direction?"""
    edges = np.array(graph.edge_order)
    _, key_inds = unique_unsorted(edges[:, 2], True)
    best_drn = np.array(topology.directions)[key_inds]
    real_drn = np.diff(edges)[:, 0]
    if topology.ring:
        num = len(graph.nodes)
        real_drn = (real_drn + num/2) % num - num/2
    return real_drn * best_drn > 0


def draw_graph(graph: GraphAttrMixin,
               pos: Union[NodePos, Layout, None] = None,
               axs: Optional[mpl.axes.Axes] = None,
               opts: Optional[GraphOptions] = None,
               ideal: Optional[ma.TopologyOptions] = None,
               **kwds) -> Tuple[NodePlots, DirectedEdgeCollection]:
    """Draw a synapse model's graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph we are drawing.
    pos : NodePos|Layout|None, optional
        Dictionary of node positions, or a function to compute them,
        by default `None -> linear_layout`.
    axs : Axes|None, optional
        Axes on which we draw, by default `None -> plt.gca()`.
    opts : GraphOptions|None, optional
        Options for graph plot, by default `None -> GraphOptions()`.
    Other keywords passed to `opts`, if accepted, or to both
    `nx.draw_networkx_nodes` and `nx.draw_networkx_edges`.

    Returns
    -------
    nodes : PathCollection
        Collection of for the plots of the nodes.
    edges : DirectedEdgeCollection
        Collection of objects for the plots of the edges.
    """
    opts = ag.default_eval(opts, GraphOptions)
    opts.pop_my_args(kwds)
    ideal = ag.default(ideal, ma.TopologyOptions(serial=True))
    axs = ag.default_eval(axs, plt.gca)
    pos = ag.default(pos, opts.layout)
    if callable(pos):
        pos = pos(graph)

    node_col = get_node_colours(graph, 'key')
    # note: PathCollection.size is area
    node_siz = graph.get_node_attr('value') * opts.size
    edge_col = get_edge_colours(graph, 'key')
    edge_wid = graph.get_edge_attr('value') * opts.width

    node_col.update(kwds, ax=axs, cmap=opts.node_cmap, edgecolors='k')
    edge_col.update(kwds, ax=axs, edge_cmap=opts.edge_cmap, node_size=node_siz,
                    connectionstyle=f'arc3,rad={opts.rad}')

    nodes = nx.draw_networkx_nodes(graph, pos, node_size=node_siz, **node_col)
    edges = nx.draw_networkx_edges(graph, pos, width=edge_wid, **edge_col)
    edge_plots = DirectedEdgeCollection(edges, graph)

    good_drn = good_direction(graph, ideal)
    rads = np.where(good_drn, opts.rad, -opts.rad/2)
    edge_plots.set_rads(rads)

    return nodes, edge_plots


def update_graph(nodes: NodePlots, edges: DirectedEdgeCollection,
                 node_siz: np.ndarray, edge_wid: np.ndarray,
                 opts: Optional[GraphOptions] = None, **kwds) -> None:
    """Update a synapse model's graph plot.

    Parameters
    ----------
    nodes : PathCollection
        The objects for the plots of the nodes.
    edges : List[FancyArrowPatch]
        The objects for the plots of the edges.
    node_siz : np.ndarray
        Sizes of the nodes (proportional to area)
    edge_wid : np.ndarray
        Widths of edges.
    opts : GraphOptions|None, optional
        Options for graph plot, by default `None -> GraphOptions()`.
    Other keywords passed to `opts`.
    """
    opts = ag.default_eval(opts, GraphOptions)
    opts.pop_my_args(kwds)
    nodes.set_sizes(node_siz * opts.size)
    edges.set_widths(edge_wid * opts.width)
    edges.set_node_sizes(node_siz * opts.size)


# =============================================================================
# Edge collection
# =============================================================================


class DirectedEdgeCollection:
    """A collection of edge plots"""
    _edges: Dict[Edge, EdgePlot]
    _node_ids: ty.List[Node]

    def __init__(self, edges: ty.Iterable[EdgePlot], graph: nx.Graph) -> None:
        self._edges = dict(zip(graph.edges, edges))
        self._node_ids = list(graph.nodes)

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

    def set_widths(self, edge_wid: ArrayLike) -> None:
        """Set line widths of edges"""
        edge_wid = np.broadcast_to(edge_wid, (len(self),), True)
        for edge, wid in zip(self._edges.values(), edge_wid):
            edge.set_linewidth(wid)

    def set_node_sizes(self, node_siz: ArrayLike) -> None:
        """Set sizes of nodes edges"""
        node_siz = cn.tuplify(node_siz, len(self._node_ids))
        for edge_id, edge_plot in self._edges.items():
            src, dst = [self._node_ids.index(node) for node in edge_id]
            edge_plot.shrinkA = _to_marker_edge(node_siz[src], 'o')
            edge_plot.shrinkB = _to_marker_edge(node_siz[dst], 'o')

    def set_edge_colors(self, cols: ArrayLikeOf[Colour]) -> None:
        """Set line colours"""
        cols = _bcast_cols(cols, len(self))
        for edge, col in zip(self._edges.values(), cols):
            edge.set_color(col)

    def set_node_pos(self, pos: NodePos) -> None:
        """Set sizes of nodes edges"""
        for edge_id, edge_plot in self._edges.items():
            edge_plot.set_position(pos[edge_id[0]], pos[edge_id[1]])

    def set_rads(self, rads: ArrayLike):
        """Set the curvature of the edges"""
        rads = np.broadcast_to(np.asanyarray(rads).ravel(), (len(self),), True)
        for edge, rad in zip(self._edges.values(), rads):
            edge.set_connectionstyle('arc3', rad=rad)


def _to_marker_edge(marker_size: Number, marker: str) -> Number:
    if marker in "s^>v<d":  # `large` markers need extra space
        return np.sqrt(2 * marker_size) / 2
    return np.sqrt(marker_size) / 2


def _bcast_cols(cols: ArrayLikeOf[Colour], num: int) -> np.ndarray:
    """broadcast cols to array of size num"""
    return np.broadcast_to(mpl.colors.to_rgba_array(cols), (num, 4), True)


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
    Other keywords passed to `opt` or `complex_synapse.graph.draw_graph`.
    """
    nodes: NodePlots
    edges: DirectedEdgeCollection
    # pinds: np.ndarray
    opts: GraphOptions

    def __init__(self, graph: GraphAttrMixin,
                 opts: Optional[GraphOptions] = None, **kwds) -> None:
        """Class for plotting model as a graph.

        Parameters
        ----------
        graph : DiGraph
            Graph object describing model. Nodes have attributes `key` and
            `value`.  Edges have attributes `key`, `value` and `pind` (if the
            model was a `SynapseParamModel`).
        opt : GraphOptions|None, optional
            Options for drawing the graph, by default `None -> GraphOptions()`.
        Other keywords passed to `opt` or `complex_synapse.graph.draw_graph`.
        """
        self.opts = ag.default_eval(opts, GraphOptions)
        self.opts.pop_my_args(kwds)
        self.nodes, self.edges = draw_graph(graph, opts=self.opts, **kwds)
        # if has_edge_attr(graph, 'pind'):
        #     self.pinds = edge_attr_vec(graph, 'pind')
        # else:
        #     self.pinds = np.arange(len(self.edges))

    def update(self, edge_vals: np.ndarray, node_vals: Optional[np.ndarray]
               ) -> None:
        """Update plots.

        Parameters
        ----------
        edge_vals : np.ndarray (E,)
            Transition probabilities, for edge line widdths.
        node_vals : None|np.ndarray (N,), optional
            Equilibrium distribution,for nodes sizes (area), by default `None`
            -> calculate from `params`.
        """
        edge_vals = np.asarray(edge_vals).ravel()
        # peq = ag.default_eval(peq, lambda: serial_eqp(params))
        self.nodes.set_sizes(node_vals * self.opts.size)
        self.edges.set_widths(edge_vals * self.opts.width)
        self.edges.set_node_sizes(node_vals * self.opts.size)

    def update_from(self, graph: GraphAttrMixin) -> None:
        """Update plots using a graph object.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph object describing model. Nodes have attributes `key` and
            `value`.  Edges have attributes `key`, `value`.
        """
        params = graph.get_edge_attr('value')
        peq = graph.get_node_attr('value')
        self.update(params, peq)

    def set_node_colors(self, cols: ArrayLikeOf[Colour]) -> None:
        """Set node colours"""
        cols = _bcast_cols(cols, len(self.nodes))
        self.nodes.set_color(cols)

    def set_edge_colors(self, cols: ArrayLikeOf[Colour]) -> None:
        """Set edge colours"""
        cols = _bcast_cols(cols, len(self.edges))
        self.edges.set_edge_colors(cols)

    def set_node_sizes(self, node_vals: ArrayLike) -> None:
        """Set node sizes"""
        self.nodes.set_sizes(node_vals * self.opts.size)
        self.edges.set_node_sizes(node_vals * self.opts.size)

    def set_widths(self, edge_vals: ArrayLike) -> None:
        """Set edge sizes"""
        self.edges.set_widths(edge_vals * self.opts.width)


# =============================================================================
# Aliases
# =============================================================================
NodePlots = mpl.collections.PathCollection
EdgePlot = mpl.patches.FancyArrowPatch
NodePos = Dict[Node, ArrayLike]
Layout = Callable[[nx.DiGraph], NodePos]
Colour = Union[str, ty.Sequence[float]]
