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
import sl_py_tools.iter_tricks as it
import sl_py_tools.numpy_tricks.markov as ma
from sl_py_tools.numpy_tricks.logic import unique_unsorted
from sl_py_tools.numpy_tricks.markov import TopologyOptions

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
        if not self._data:
            return None
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
        if not self._data:
            return None
        edge_data = self._report(*key, self._viewer[key])
        if edge_data in self:
            return edge_data[-1]
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


class GraphAttrs(nx.Graph):
    """Mixin providing attribute array methods.

    This class provides methods for working with `np.ndarray`s of node/edge
    attribute: `has_node_attr`, `get_node_attr`, `set_node_attr`,
    `has_edge_attr`, `get_edge_attr`, `set_edge_attr`.

    See Also
    --------
    `networkx.Graph, DiGraph, MultiGraph, MultiDiGraph`.
    """

    def has_node_attr(self, data: str, strict: bool = True) -> bool:
        """Test for existence of node attributes.

        Parameters
        ----------
        key : str
            Name of attribute.
        strict : bool, optopnal
            Only `True` if every node has the attribute. By default `True`.

        Returns
        -------
        has : bool
            `True` when some/every node has the attribute, `False` otherwise.
        """
        fun = all if strict else any
        return fun(data in node for node in self.nodes.values())

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
        """Change values of node attributes.

        Parameters
        ----------
        data : str
            Name of attribute.
        values : ndarray (N,)
            Value to assign to the attribute for each node.
        """
        for node_dict, value in zip(self.nodes.values(), values):
            node_dict[data] = value

    def has_edge_attr(self, data: str, strict: bool = True) -> bool:
        """Test for existence of edge attributes.

        Parameters
        ----------
        data : str
            Name of attribute.
        strict : bool, optopnal
            Only `True` if every edge has the attribute. By default `True`.

        Returns
        -------
        has : bool
            `True` when some/every edge has the attribute, `False` otherwise.
        """
        fun = all if strict else any
        return fun(data in edge for edge in self.edges.values())

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
        values : ndarray (E,)
            Value to assign to the attribute for each edge.
        """
        for edge_dict, value in zip(self.edges.values(), values):
            edge_dict[data] = value


class DiGraph(nx.DiGraph, GraphAttrs):
    """Custom directed graph class that remembers edge order (N,E)

    Iterating over edges is done in the order that the edges were first added.
    It also provides methods for working with `np.ndarray`s of node/edge
    attribute: `has_node_attr`, `get_node_attr`, `set_node_attr`,
    `has_edge_attr`, `get_edge_attr`, `set_edge_attr`.

    See Also
    --------
    `networkx.DiGraph`
    `GraphAttrs`
    """
    edge_order: ty.List[Edge]

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.edge_order = []

    def add_edge(self, u_of_edge: Node, v_of_edge: Node, **attr) -> None:
        """Add/modify an edge.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u, v : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.
        """
        super().add_edge(u_of_edge, v_of_edge, **attr)
        if (u_of_edge, v_of_edge) not in self.edge_order:
            self.edge_order.append((u_of_edge, v_of_edge))

    @property
    def edges(self) -> OutEdgeView:
        """OutEdgeView of the DiGraph as G.edges or G.edges().

        This property provides a `dict/set`-like view of the graph's edge,
        with tuples `(u,v)` as keys and `(u,v,data)` as set-members. Iterating
        over edges is done in the order that the edges were first added.

        Calling this property with optional arguments `data` and `default`
        controls the form of the tuple. Optional argument `nbunch`
        allows restriction to edges only involving certain nodes.
        If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
        If `data is True` iterate over 3-tuples `(u, v, datadict)`.
        Otherwise iterate over `(u, v, datadict.get(data, default))`.
        """
        return OutEdgeView(self)

    def edge_attr_matrix(self, data: str, fill: Number = 0.) -> np.ndarray:
        """Collect values of edge attributes in a matrix.

        Parameters
        ----------
        data : str
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
        for edge, val in self.edges(data=data, default=fill).items():
            ind = tuple(map(nodes.index, edge))
            mat[ind] = val
        return mat


class MultiDiGraph(nx.MultiDiGraph, GraphAttrs):
    """Custom directed multi-graph class that remembers edge order (N,E)

    Iterating over edges is done in the order that the edges were first added.
    It also provides methods for working with `np.ndarray`s of node/edge
    attribute: `has_node_attr`, `get_node_attr`, `set_node_attr`,
    `has_edge_attr`, `get_edge_attr`, `set_edge_attr`.

    See Also
    --------
    `networkx.MultiDiGraph`
    `GraphAttrs`
    """
    edge_order: ty.List[Edge]

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)
        self.edge_order = []

    def add_edge(self, u_for_edge: Node, v_for_edge: Node,
                 key: Optional[Key] = None, **attr) -> Key:
        """Add/modify an edge

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Edge attributes can be specified with keywords or by directly
        accessing the edge's attribute dictionary. See examples below.

        Parameters
        ----------
        u_for_edge, v_for_edge : nodes
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.
        key : hashable identifier, optional
            Used to distinguish multiedges between a pair of nodes.
            By default lowest unused integer
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        key : hashable identifier
            The edge key assigned to the edge.
        """
        # attr.setdefault('pind', len(self.edges))
        key = super().add_edge(u_for_edge, v_for_edge, key, **attr)
        if (u_for_edge, v_for_edge, key) not in self.edge_order:
            self.edge_order.append((u_for_edge, v_for_edge, key))
        return key

    @property
    def edges(self) -> OutMultiEdgeView:
        """OutMultiEdgeView of the MultiDiGraph as G.edges or G.edges(...).

        This property provides a `dict/set`-like view of the graph's edge,
        with tuples `(u,v)` or `(u,v,key)` as keys and `(u,v,key,data)` as
        set-members. Iterating over edges is done in the order that the edges
        were first added.

        Calling this property with optional arguments `data`, `default` and
        `keys` controls the form of the tuple. Optional argument `nbunch`
        allows restriction to edges only involving certain nodes.
        If `data is False` (the default) then iterate over 2-tuples `(u, v)`.
        If `data is True` iterate over 3-tuples `(u, v, datadict)`.
        Otherwise iterate over `(u, v, datadict.get(data, default))`.
        If `keys is True`, replace `u, v` with `u, v, key` above.
        """
        return OutMultiEdgeView(self)

    def edge_key(self) -> np.ndarray:
        """Vector of edge keys

        Returns
        -------
        keys : np.ndarray (E,)
            Vector of keys for each edge, in the order edges were first added.
        """
        return np.array(self.edge_order)[:, 2]

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
        if data == 'key':
            return self.edge_key()
        return super().get_edge_attr(data, default)

    def edge_attr_matrix(self, data: str, fill: Number = 0.) -> np.ndarray:
        """Collect values of edge attributes in an array of matrices.

        Parameters
        ----------
        data : str
            Name of attribute to use for matrix elements.
        fill : Number
            Value given to missing edges.

        Returns
        -------
        mat : np.ndarray (K,N,N)
            Matrices of edge attribute values. Each matrix in the array
            corresponds to a value of `list_edge_keys(self`, in that order.
        """
        nodes = list(self.nodes)
        _, keys = list_edge_keys(self, get_inv=True)
        mat = np.full((keys.max() + 1, len(nodes), len(nodes)), fill)
        for key, edge_val in zip(keys, self.edges(data=data, default=fill)):
            ind = (key,) + tuple(map(nodes.index, edge_val[:2]))
            mat[ind] = edge_val[2]
        return mat


# =============================================================================
# Graph builders
# =============================================================================


def mat_to_graph(mat: np.ndarray, node_values: Optional[np.ndarray] = None,
                 node_keys: Optional[np.ndarray] = None,
                 edge_keys: Optional[np.ndarray] = None) -> MultiDiGraph:
    """Create a directed multi-graph from a parameters of a Markov model.

    Parameters
    ----------
    mat : np.ndarray (P,M,M)
        Array of transition matrices.
    node_values : np.ndarray (M,)
        Value associated with each node. By default `ma.calc_peq(...)`.
    node_keys : np.ndarray (M,)
        Node type. By default `np.zeros(nstate)`.
    edge_keys : np.ndarray (P,)
        Edge type. By default `np.arange(...)`.

    Returns
    -------
    graph : DiGraph
        Graph describing model.
    """
    mat = mat[None] if mat.ndim == 2 else mat
    drn = (0,) * mat.shape[0]
    topology = TopologyOptions(directions=drn)
    params = ma.params.gen_mat_to_params(mat, drn)
    edge_keys = ag.default(edge_keys, np.arange(len(drn)))
    if node_values is None:
        axis = tuple(range(mat.ndim-2))
        node_values = ma.calc_peq(mat.sum(axis))
    return param_to_graph(params, node_values, node_keys, edge_keys, topology)


def param_to_graph(param: np.ndarray, node_values: Optional[np.ndarray] = None,
                   node_keys: Optional[np.ndarray] = None,
                   edge_keys: Optional[np.ndarray] = None,
                   topology: Optional[TopologyOptions] = None) -> MultiDiGraph:
    """Create a directed multi-graph from a parameters of a Markov model.

    Parameters
    ----------
    param : np.ndarray (PQ,), Q in [M(M-1), M, M-1, 1]
        Independent parameters of model - a `(P,M,M)` array.
    node_values : np.ndarray (M,), optional
        Value associated with each node. By default `ma.calc_peq(...)`.
    node_keys : np.ndarray (M,), optional
        Node type. By default `np.zeros(nstate)`.
    edge_keys : np.ndarray (P,), optional
        Edge type. By default `topology.directions`.
    topology : TopologyOptions, optional
        Encapsulation of model class. By default `TopologyOptions()`.

    Returns
    -------
    graph : DiGraph
        Graph describing model.
    """
    topology = ag.default_eval(topology, TopologyOptions)
    param = np.asanyarray(param)
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


def list_node_attrs(graph: GraphAttrs, strict: bool = False) -> ty.List[str]:
    """List of attributes of nodes in the graph

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph
        Graph with nodes whose attributes we want.
    strict : bool, optopnal
        Only list attribute if every node has the it. By default `False`.
    """
    attrs = {}
    for node_val in graph.nodes.values():
        attrs.update(node_val)
    if not strict:
        return list(attrs)
    return list(filter(graph.has_node_attr, attrs))


def list_edge_attrs(graph: GraphAttrs, strict: bool = False) -> ty.List[str]:
    """List of attributes of edges in the graph

    Parameters
    ----------
    graph : DiGraph|MultiDiGraph (N,E)
        Graph with edges whose attributes we want.
    strict : bool, optopnal
        Only list attribute if every node has the it. By default `False`.
    """
    attrs = {}
    for edge_val in graph.edges.values():
        attrs.update(edge_val)
    if not strict:
        return list(attrs)
    return list(filter(graph.has_edge_attr, attrs))


def list_edge_keys(graph: MultiDiGraph, get_inv: bool = False) -> np.ndarray:
    """Vector of unique keys of edges in the graph

    Parameters
    ----------
    graph : MultiDiGraph (N,E)
        Graph with edges whose keys we want.
    get_inv : bool, optopnal
        Also return assignments of edges to keys. By default `False`.

    Returns
    -------
    keys : ndarray (K,)
        Values of keys, in order of first appearance in `graph.edge_key()`.
    inv : ndarray (E,)
        Index array: assignments of each edge's keys to each entry of `keys`.
    `graph.edge_key() == keys[inv]`.
    """
    key_vec = graph.edge_key()
    return unique_unsorted(key_vec, get_inv)


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
