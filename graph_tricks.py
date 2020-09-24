# -*- coding: utf-8 -*-
"""Tools for working with graphs and plotting them
"""
from __future__ import annotations

import operator as opr
import typing as ty
from numbers import Number
from typing import Callable, Optional, Tuple, Union

import networkx as nx
import numpy as np

import sl_py_tools.arg_tricks as ag
import sl_py_tools.containers as cn
import sl_py_tools.iter_tricks as it
import sl_py_tools.numpy_tricks.markov as ma
from sl_py_tools.numpy_tricks.logic import unique_unsorted

# =============================================================================
# Graph view class
# =============================================================================


class OutEdgeDataView(nx.classes.reportviews.OutEdgeDataView):
    """Custom edge data view for DiGraph

    This view is primarily used to iterate over the edges reporting edges as
    node-tuples with edge data optionally reported. It is returned when the
    `edges` property of `DiGraph` is called. The argument `nbunch` allows
    restriction to edges incident to nodes in that container/singleton.
    The default (nbunch=None) reports all edges. The arguments `data` and
    `default` control what edge data is reported. The default `data is False`
    reports only node-tuples for each edge. If `data is True` the entire edge
    data dict is returned. Otherwise `data` is assumed to hold the name of the
    edge attribute to report with default `default` if  that edge attribute is
    not present.

    The iteration order is the same as the order the edges were first added.
    The iterator does not behave like a `dict`. The values yielded by the
    iterator are `tuple`s: `(from_node, to_node, data)`. Membership tests
    are for these tuples. It also has methods that iterate over subsets of
    these: `keys() -> (from_node, to_node)`, `values() -> data`, and
    `items() -> ((from_node, to_node), data)`. Unlike `dict` views, these
    methods *only* provide iterables, they do *not* provide `set` operations.
    """
    __slots__ = ()
    _viewer: OutEdgeView
    _report: Callable[[Node, Node, Attrs], Tuple[Node, Node, Data]]

    def __iter__(self) -> ty.Iterator[Tuple[Node, Node, Data]]:
        """Set-like of edge-data tuples, not dict-like

        Yields
        -------
        edge_data : Tuple[Node, Node, Data]]
            The tuple `(from_node, to_node, data)` describing each edge.
        """
        for edge in self._viewer:
            edge_data = self._report(*edge, self._viewer[edge])
            if edge_data in self:
                yield edge_data

    def __getitem__(self, key: Edge) -> Data:
        """Use edge as key to get data

        Parameters
        ----------
        key : Tuple[Node, Node]
            The tuple `(from_node, to_node)` describing the edge.

        Returns
        -------
        data : Data
            The data associated with the edge, depending on the `data`
            parameter used when creating the instance.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        edge_data = self._report(*key, self._viewer[key])
        if edge_data in self:
            return edge_data[-1]
        raise KeyError(f"Edge {key} not in this (sub)graph.")

    def keys(self) -> ty.Iterable[Edge]:
        """Iterable view of edges

        Yields
        -------
        key : Tuple[Node, Node]
            The tuple `(from_node, to_node)` describing the edge.
        """
        if not self._data:
            return self
        return map(opr.itemgetter(slice(-1)), self)

    def values(self) -> ty.Iterable[Data]:
        """View of edge attribute

        Yields
        -------
        data : Data
            The data associated with the edge, depending on the `data`
            parameter used when creating the instance.
        """
        if not self._data:
            return (None,) * len(self)
        return map(opr.itemgetter(-1), self)

    def items(self) -> ty.Iterable[Tuple[Edge, Data]]:
        """View of edge and attribute

        Yields
        -------
        edge_data : Tuple[Tuple[Node, Node], Data]
            The tuple `((from_node, to_node), data)` describing each edge.
        """
        return zip(self.keys(), self.values())


# pylint: disable=too-many-ancestors
class OutEdgeView(nx.classes.reportviews.OutEdgeView):
    """Custom edge view for DiGraph

    This densely packed View allows iteration over edges, data lookup
    like a dict and set operations on edges represented by node-tuples.
    In addition, edge data can be controlled by calling this object
    possibly creating an EdgeDataView. Typically edges are iterated over
    and reported as `(u, v)` node tuples. Those edge representations can also
    be used to lookup the data dict for any edge. Set operations also are
    available where those tuples are the elements of the set.

    Calling this object with optional arguments `data`, `default` and `keys`
    controls the form of the tuple (see EdgeDataView). Optional argument
    `nbunch` allows restriction to edges only involving certain nodes.
    If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
    If `data is True` iterate over 3-tuples `(u, v, datadict)`.
    Otherwise iterate over `(u, v, datadict.get(data, default))`.
    For Multigraphs, if `keys is True`, replace `u, v` with `u, v, key` above.
    """
    __slots__ = ()
    _graph: DiGraph
    dataview: ty.ClassVar[type] = OutEdgeDataView

    def __iter__(self) -> ty.Iterator[Edge]:
        """Iterable view of edges

        Yields
        -------
        key : Tuple[Node, Node]
            The tuple `(from_node, to_node)` describing the edge.
        """
        if hasattr(self._graph, 'edge_order'):
            yield from self._graph.edge_order
        else:
            yield from super().__iter__()
# pylint: enable=too-many-ancestors


class OutMultiEdgeDataView(nx.classes.reportviews.OutMultiEdgeDataView):
    """Custom edge data view for MultiDiGraph

    This view is primarily used to iterate over the edges reporting edges as
    node-tuples with edge data optionally reported. It is returned when the
    `edges` property of `MultiDiGraph` is called. The argument `nbunch` allows
    restriction to edges incident to nodes in that container/singleton.
    The default (nbunch=None) reports all edges. The arguments `data` and
    `default` control what edge data is reported. The default `data is False`
    reports only node-tuples for each edge. If `data is True` the entire edge
    data dict is returned. Otherwise `data` is assumed to hold the name of the
    edge attribute to report with default `default` if  that edge attribute is
    not present. The argument `keys` controls whether or not `key` is included
    in the `tuple`s yielded by the iterators below.

    The iteration order is the same as the order the edges were first added.
    The iterator does not behave like a `dict`. The values yielded by the
    iterator are `tuple`s: `(from_node, to_node, key, data)`. Membership tests
    are for these tuples. It also has methods that iterate over subsets of
    these: `mkeys() -> (from_node, to_node, key)`, `values() -> data`, and
    `items() -> ((from_node, to_node, key), data)`. Unlike `dict` views, these
    methods *only* provide iterables, they do *not* provide `set` operations.
    """
    __slots__ = ()
    keys: bool
    _data: Union[bool, str]
    _viewer: OutMultiEdgeView
    _report: Callable[[Node, Node, Key, Attrs], Tuple[Node, Node, Key, Data]]

    def __iter__(self) -> ty.Iterator[Tuple[Node, Node, Key, Data]]:
        """Set-like of edge-data tuples, not dict-like

        Yields
        -------
        edge_data : Tuple[Node, Node, Key, Data]]
            The tuple `(from_node, to_node, key, data)` describing each edge.
        """
        for edge in self._viewer:
            edge_data = self._report(*edge, self._viewer[edge])
            if edge_data in self:
                yield edge_data

    def __getitem__(self, key: MEdge) -> Data:
        """Use edge as key to get data

        Parameters
        ----------
        key : Tuple[Node, Node, Key]
            The tuple `(from_node, to_node, key)` describing the edge.

        Returns
        -------
        data : Data
            The data associated with the edge, depending on the `data`
            parameter used when creating the instance.

        Raises
        ------
        KeyError
            If the key is not present.
        """
        edge_data = self._report(*key, self._viewer[key])
        if edge_data in self:
            return cn.unseqify(edge_data[2 + self.keys:])
        raise KeyError(f"Edge {key} not in this (sub)graph.")

    def mkeys(self) -> ty.Iterable[MEdge]:
        """Iterable view of edges

        Yields
        -------
        key : Tuple[Node, Node, Key]
            The tuple `(from_node, to_node, key)` describing the edge.
        """
        if not self._data:
            return self
        return map(opr.itemgetter(slice(-1)), self)

    def values(self) -> ty.Iterable[Data]:
        """View of edge attribute

        Yields
        -------
        data : Data
            The data associated with the edge, depending on the `data`
            parameter used when creating the instance.
        """
        if not self._data:
            return (None,) * len(self)
        return map(opr.itemgetter(-1), self)

    def items(self) -> ty.Iterable[Tuple[MEdge, Data]]:
        """View of edge and attribute

        Yields
        -------
        edge_data : Tuple[Tuple[Node, Node, Key], Data]
            The tuple `((from_node, to_node, key), data)` describing each edge.
        """
        return zip(self.mkeys(), self.values())


# pylint: disable=too-many-ancestors
class OutMultiEdgeView(nx.classes.reportviews.OutMultiEdgeView):
    """Custom edge view for DiGraph

    This densely packed View allows iteration over edges, data lookup
    like a dict and set operations on edges represented by node-tuples.
    In addition, edge data can be controlled by calling this object
    possibly creating an EdgeDataView. Typically edges are iterated over
    and reported as `(u, v)` node tuples or `(u, v, key)` node/key tuples.
    Those edge representations can also be used lookup the data dict for any
    edge. Set operations also are available where those tuples are the
    elements of the set.

    Calling this object with optional arguments `data`, `default` and `keys`
    controls the form of the tuple (see EdgeDataView). Optional argument
    `nbunch` allows restriction to edges only involving certain nodes.
    If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
    If `data is True` iterate over 3-tuples `(u, v, datadict)`.
    Otherwise iterate over `(u, v, datadict.get(data, default))`.
    If `keys is True`, replace `u, v` with `u, v, key` above.
    """
    __slots__ = ()
    _graph: MultiDiGraph
    dataview: ty.ClassVar[type] = OutMultiEdgeDataView

    def __iter__(self) -> ty.Iterator[MEdge]:
        """Iterable view of edges

        Yields
        -------
        key : Tuple[Node, Node, Key]
            The tuple `(from_node, to_node, key)` describing the edge.
        """
        if hasattr(self._graph, 'edge_order'):
            yield from self._graph.edge_order
        else:
            yield from super().__iter__()
# pylint: enable=too-many-ancestors


# =============================================================================
# Graph classes
# =============================================================================


class GraphAttrMixin(nx.Graph):
    """Mixin providing attribute collecting methods"""

    def has_node_attr(self, data: str) -> bool:
        """Test for existence of node attributes.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph with nodes whose attributes we want.
        key : str
            Name of attribute.

        Returns
        -------
        has : bool
            `True` if every node has the attribute, `False` otherwise.
        """
        return all(data in node for node in self.nodes.values())

    def get_node_attr(self, data: str, default: Number = np.nan) -> np.ndarray:
        """Collect values of node attributes.

        Parameters
        ----------
        data : str
            Name of attribute.
        default : Number, optional
            Value to use for nodes without that attribute, by default `nan`.

        Returns
        -------
        vec : np.ndarray (N,)
            Vector of node attribute values.
        """
        return np.array(list(self.nodes(data=data, default=default)))[:, 1]

    def set_node_attr(self, data: str, values: np.ndarray) -> None:
        """Collect values of node attributes.

        Parameters
        ----------
        data : str
            Name of attribute.
        values : ndarray
            Value to assign to the attribute for each node.
        """
        for node_dict, value in zip(self.nodes.values(), values):
            node_dict[data] = value

    def has_edge_attr(self, data: str) -> bool:
        """Test for existence of edge attributes.

        Parameters
        ----------
        data : str
            Name of attribute.

        Returns
        -------
        has : bool
            `True` if every edge has the attribute, `False` otherwise.
        """
        return all(data in edge for edge in self.edges.values())

    def get_edge_attr(self, data: str, default: Number = np.nan) -> np.ndarray:
        """Collect values of edge attributes.

        Parameters
        ----------
        data : str
            Name of attribute.
        default : Number, optional
            Value to use for nodes without that attribute, by default `nan`.

        Returns
        -------
        vec : np.ndarray (E,)
            Vector of edge attribute values.
        """
        return np.array(list(self.edges(data=data, default=default).values()))

    def set_edge_attr(self, data: str, values: np.ndarray) -> None:
        """Collect values of edge attributes.

        Parameters
        ----------
        data : str
            Name of attribute.
        values : ndarray
            Value to assign to the attribute for each edge.
        """
        for edge_dict, value in zip(self.edges.values(), values):
            edge_dict[data] = value


class DiGraph(nx.DiGraph, GraphAttrMixin):
    """Custom directed graph class that remembers edge order (N,E)
    """
    edge_order: ty.List[Edge]

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.edge_order = []

    def add_edge(self, u_of_edge: Node, v_of_edge: Node, **attr) -> None:
        """Add an edge"""
        # attr.setdefault('pind', len(self.edges))
        super().add_edge(u_of_edge, v_of_edge, **attr)
        if (u_of_edge, v_of_edge) not in self.edge_order:
            self.edge_order.append((u_of_edge, v_of_edge))

    @property
    def edges(self) -> OutEdgeView:
        """OutEdgeView of the DiGraph as G.edges or G.edges()."""
        return OutEdgeView(self)

    def edge_attr_matrix(self, key: str, fill: Number = 0.) -> np.ndarray:
        """Collect values of edge attributes in a matrix.

        Parameters
        ----------
        key : str
            Name of attribute to use for matrix elements.
        fill : Number
            Value given to missing edges.

        Returns
        -------
        mat : np.ndarray (N,N)
            Matrix of edge attribute values.
        """
        nodes = list(self.nodes)
        mat = np.full((len(nodes),) * 2, fill)
        for edge, val in self.edges(data=key, default=fill).items():
            ind = tuple(map(nodes.index, edge))
            mat[ind] = val
        return mat


class MultiDiGraph(nx.MultiDiGraph, GraphAttrMixin):
    """Custom directed multi-graph class that remembers edge order (N,E)
    """
    edge_order: ty.List[Edge]

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.edge_order = []

    def add_edge(self, u_for_edge: Node, v_for_edge: Node,
                 key: Optional[Key] = None, **attr) -> Key:
        """Add an edge"""
        # attr.setdefault('pind', len(self.edges))
        key = super().add_edge(u_for_edge, v_for_edge, key, **attr)
        if (u_for_edge, v_for_edge, key) not in self.edge_order:
            self.edge_order.append((u_for_edge, v_for_edge, key))
        return key

    @property
    def edges(self) -> OutMultiEdgeView:
        """OutMultiEdgeView of the MultiDiGraph as G.edges or G.edges(...)."""
        return OutMultiEdgeView(self)

    def edge_key(self) -> np.ndarray:
        """Vector of edge keys

        Returns
        -------
        keys : np.ndarray (E,)
            Vector of keys for each edge, in the order edges were first added.
        """
        return np.array(self.edge_order)[:, 2]

    def edge_attr_matrix(self, key: str, fill: Number = 0.) -> np.ndarray:
        """Collect values of edge attributes in an array of matrices.

        Parameters
        ----------
        key : str
            Name of attribute to use for matrix elements.
        fill : Number
            Value given to missing edges.

        Returns
        -------
        mat : np.ndarray (K,N,N)
            Matrices of edge attribute values. Each matrix in the array
            corresponds to an entry of `self.edge_keys()`, in that order.
        """
        nodes = list(self.nodes)
        keys = unique_unsorted(self.edge_key()).tolist()
        mat = np.full((len(keys), len(nodes), len(nodes)), fill)
        for edge, val in self.edges(data=key, default=fill, keys=True).items():
            ind = (keys.index(edge[2]),) + tuple(map(nodes.index, edge[:2]))
            mat[ind] = val
        return mat


# =============================================================================
# Graph builders
# =============================================================================


def mat_to_graph(mat: np.ndarray, node_values: Optional[np.ndarray] = None,
                 node_keys: Optional[np.ndarray] = None,
                 edge_keys: Optional[np.ndarray] = None) -> MultiDiGraph:
    """Create a directed graph from a parameters of a Markov model.

    Parameters
    ----------
    mat : np.ndarray (P,M,M)
        Array of transition matrices.
    node_values : np.ndarray (M,)
        Value associated with each node.
    node_keys : np.ndarray (M,)
        Node type.
    edge_keys : np.ndarray (P,)
        Edge type.
    topology : ma.TopologyOptions
        Encapsulation of model class.

    Returns
    -------
    graph : DiGraph
        Graph describing model.
    """
    mat = mat[None] if mat.ndim == 2 else mat
    drn = (0,) * mat.shape[0]
    topology = ma.TopologyOptions(directions=drn)
    params = ma.params.gen_mat_to_params(mat, drn)
    if node_values is None:
        axis = tuple(range(mat.ndim-2))
        node_values = ma.calc_peq(mat.sum(axis))
    return param_to_graph(params, node_values, node_keys, edge_keys, topology)


def param_to_graph(param: np.ndarray, node_values: Optional[np.ndarray] = None,
                   node_keys: Optional[np.ndarray] = None,
                   edge_keys: Optional[np.ndarray] = None,
                   topology: Optional[ma.TopologyOptions] = None) -> MultiDiGraph:
    """Create a directed graph from a parameters of a Markov model.

    Parameters
    ----------
    param : np.ndarray (PQ,), Q in [M(M-1), M, M-1, 1]
        Independent parameters of model - a `(P,M,M)` array.
    node_values : np.ndarray (M,)
        Value associated with each node.
    node_keys : np.ndarray (M,)
        Node type.
    edge_keys : np.ndarray (P,)
        Edge type.
    topology : ma.TopologyOptions
        Encapsulation of model class.

    Returns
    -------
    graph : DiGraph
        Graph describing model.
    """
    topology = ag.default_eval(topology, ma.TopologyOptions)
    nstate = ma.params.num_state(param, **topology.directed())
    node_keys = ag.default(node_keys, np.zeros(nstate))
    edge_keys = ag.default(edge_keys, topology.directions)
    if node_values is None:
        mat = ma.params.params_to_mat(param, **topology.directed())
        axis = tuple(range(mat.ndim-2))
        node_values = ma.calc_peq(mat.sum(axis))
    graph = MultiDiGraph()
    for node in it.zenumerate(node_keys, node_values):
        graph.add_node(node[0], key=node[1], value=node[2])
    # (3,)(P,Q)
    inds = ma.indices.param_subs(nstate, ravel=True, **topology.directed())
    for i, j, k, val in zip(*inds, param.ravel()):
        graph.add_edge(j, k, key=edge_keys[i], value=val)
    return graph


# =============================================================================
# Graph attributes
# =============================================================================


def list_node_attrs(graph: nx.Graph) -> ty.List[str]:
    """List of attributes of nodes in the graph"""
    attrs = {}
    for node_val in graph.nodes.values():
        attrs.update(node_val)
    return list(attrs)


def list_edge_attrs(graph: nx.Graph) -> ty.List[str]:
    """List of attributes of edges in the graph"""
    attrs = {}
    for edge_val in graph.edges.values():
        attrs.update(edge_val)
    return list(attrs)


# =============================================================================
# Aliases
# =============================================================================
Some = ty.TypeVar("Some")
ArrayLikeOf = Union[Some, ty.Sequence[Some], np.ndarray]
ArrayLike = ArrayLikeOf[Number]
Node = ty.TypeVar('Node', int, str, ty.Hashable)
Key = ty.TypeVar('Key', int, str, ty.Hashable)
Edge, MEdge = Tuple[Node, Node], Tuple[Node, Node, Key]
Attrs = ty.Dict[str, Number]
Data = Union[Number, Attrs, None]
