import graph_tool.all as gt
import numpy          as np
import numbers
import copy

from .graph_preparation import set_types_random, set_types_pagerank
from .union_find        import UnionFind

def _matrix2dict(matrix) :
    """Takes an adjacency matrix and returns an adjacency list."""
    n   = len(matrix)
    adj = {k : [] for k in range(n)}
    for k in range(n) :
        for j in range(n) :
            if matrix[k, j] :
                adj[k].append(j)
    
    return adj


def _dict2dict(adj_dict) :
    """Takes a dictionary representation of an adjacency list and returns
    a list based representation.
    """
    vertices = set()
    for key, value in adj_dict.items() :
        vertices.add(key)
        if isinstance(value, numbers.Integral) :
            vertices.add(value)
        else :
            vertices.update(value)

    if min(vertices) == 1 :
        adjacency = {}
        for key, value in adj_dict.items() :
            adjacency[key-1] = [k-1 for k in value]

        for v in vertices :
            if v - 1 not in adjacency :
                adjacency[v-1] = []
    else :
        adjacency = adj_dict.copy()

        for v in vertices :
            if v not in adjacency :
                adjacency[v] = []

    return adjacency


def _list2dict(adj_list) :
    """Takes a dictionary representation of an adjacency list and returns
    a list based representation.
    """
    vertices = set()
    adj_dict = {}
    for key, value in enumerate(adj_list) :
        vertices.add(key)
        vertices.update(value)
  
    for key, value in enumerate(adj_list) :
        adj_dict[key] = value
    
    for v in vertices :
        if v not in adj_dict :
            adj_dict[v] = []
    
    return adj_dict


def _other2dict(adj_dict, other) :
    other_dict = {}
    if isinstance(other, np.ndarray) :
        other = _matrix2dict(other)

    if isinstance(other, dict) :
        for k, value in adj_dict.items() :
            if k in other :
                other_dict[k] = other[k]
            else :
                other_dict[k] = []
    elif isinstance(other, list) :
        for k, value in adj_dict.items() :
            if k < len(other) :
                other_dict[k] = other[k]
            else :
                other_dict[k] = []
    else :
        raise TypeError('edge_type must by either a dict, list, or numpy.ndarray')

    return other_dict


def _adjacency_adjust(adjacency, edge_type, adjust_type, is_directed) :
    """Takes an adjacency list and returns a (possibly) modified adjacency list."""
    ok_adj = True

    for adj in adjacency.values() :
        if len(adj) == 0 :
            ok_adj = False
            break

    if edge_type is None :
        for adj in adjacency.values() :
            edge_type.append([1 for k in adj])
    else :
        if len(adjacency) != len(edge_type) :
            raise RuntimeError("Graph for edge types must match graph from adjacency list/matrix.")
        for k in adjacency.keys() :
            if len(adjacency[k]) != len(edge_type[k]) :
                raise RuntimeError("Graph for edge types must match graph from adjacency list/matrix.")

    if not ok_adj and is_directed :
        if adjust_type == 1 :
            null_nodes = set()

            for k, adj in adjacency.items() :
                if len(adj) == 0 :
                    null_nodes.add(k)

            for k, adj in edge_type.items() :
                edge_type[k] = [0 if j in null_nodes else j for j in adj]

        elif adjust_type == 2 :
            for k, adj in adjacency.items() :
                if len(adj) == 0 :
                    adjacency[k].append(n + 1)
                    edge_type[k].append(0)

            adjacency[len(adjacency)] = []
            edge_type[len(adjacency)] = []

        else :
            for k, adj in adjacency.items() :
                if len(adj) == 0 :
                    adjacency[k].append(k)
                    edge_type[k].append(0)

    return adjacency, edge_type


def adjacency2graph(adjacency, edge_type=None, adjust_type=0, is_directed=True) :
    """Takes an adjacency list, dict, or matrix and returns a graph.

    The purpose of this function is take an adjacency list (or matrix) and
    return a :class:`~graph_tool.Graph` that can be used with
    :class:`~queueing_tool.network.QueueNetwork`. The Graph returned has a 
    ``vType`` vertex property, and ``eType`` edge property. If the adjacency is
    directed and not connected, then the adjacency list is altered.

    Parameters
    ----------
    adjacency : list, dict, or numpy.ndarray
        An adjacency list, dict, or matrix.
    edge_type : list, dict, or numpy.ndarray (optional)
        A mapping that corresponds to that edges ``eType``. For example, if
        ``edge_type`` is a matrix, then ``edge_type[u, v]`` is the type of
        queue along the edge between vertices ``u`` and ``v``. If ``edge_type``
        is not supplied then all but terminal edges have type 1.
    adjust_type : int (optional, the default is 0)
        Specifies what to do when the graph has terminal vertices (nodes with
        no out-edges). There are three choices:

            ``adjust_type = 0``: A loop is added to each terminal node in the
            graph, and their ``eType`` of that edge is set to 0.
            ``adjust_type = 1``: All edges leading to terminal nodes have their
            ``eType`` set to 0.
            ``adjust_type = 2``: A new vertex is created and each node has an
            edge connect to the new vertex. The ``eType`` for each of these new
            edges is set to 0.

        Note that if ``adjust_type`` is not 1 or 2 then it assumed to be 0.
    is_directed : bool (optional, the default is True)
        Sets whether the returned graph is directed or not.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A :class:`~graph_tool.Graph` with the ``vType`` vertex property, and 
        ``eType`` edge property.

    Raises
    ------
    TypeError
        Is raised if ``adjacency`` or ``edge_type`` is not a :class:`.list`\,
        :class:`.dict`\, :class:`~numpy.ndarray` the (``edge_type`` can be 
        ``None``\).
    RuntimeError
        A :exc:`~RuntimeError` is raised if ``edge_type`` does not have the 
        same dimensions as ``adjacency``\.
    """
    if adjacency is None :
        raise TypeError("The `adjacency` parameter must be a list, dict, or a numpy.ndarray.")

    params = [adjacency, edge_type, edge_length]
    string = ['adjacency', 'edge_type', 'edge_length']

    if isinstance(adjacency, np.ndarray) :
        adjacency = _matrix2dict(adjacency)
    elif isinstance(adjacency, dict) :
        adjacency = _dict2dict(adjacency)
    elif isinstance(adjacency, list) :
        adjacency = _list2dict(adjacency)
    else :
        raise TypeError("If the adjacency parameter is supplied it must be a list, dict, or a numpy.ndarray.")

    if edge_type is not None :
        edge_type = _other2dict(adjacency, edge_type)

    adjacency, edge_type = _adjacency_adjust(adjacency, edge_type, adjust_type, is_directed)

    nV  = len(adjacency)
    g   = gt.Graph()
    vs  = g.add_vertex(nV)

    g.set_directed(is_directed)



    vType   = g.new_vertex_property("int")
    eType   = g.new_edge_property("int")
    elength = g.new_edge_property("double")
    vType.a = 1

    for u, adj in adjacency.items() :
        for j, v in enumerate(adj) :
            e = g.add_edge(u, v)
            eType[e]    = edge_type[u][j]
            elength[e]  = edge_length[u][j] if not edge_length_none else 1.0
            if u == v :
                vType[e.source()] = edge_type[u][j]

    g.vp['vType'] = vType
    g.ep['eType'] = eType
    if not edge_length_none :
        g.ep['edge_length'] = elength
    return g
    

def generate_random_graph(nVertices=250, **kwargs) :
    """Creates a random graph where the edge and vertex types are selected 
    using the :func:`~set_types_random` method.

    Calls :func:`~minimal_random_graph` and then calls :func:`~set_types_random`.

    Parameters
    ----------
    nVertices : int (optional, the default is 250)
        The number of vertices in the graph.
    **kwargs :
        Any parameters to send to :func:`~minimal_random_graph` or
        :func:`~set_types_random`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A graph with a ``pos`` and ``vType`` vertex property and the ``eType``
        edge property.
    """
    g = minimal_random_graph(nVertices, **kwargs)
    g = set_types_random(g, **kwargs)
    return g


def generate_pagerank_graph(nVertices=250, **kwargs) :
    """Creates a random graph where the edge and vertex types are selected 
    using the :func:`~set_types_pagerank` method.

    Calls :func:`~minimal_random_graph` and then calls :func:`~set_types_pagerank`.

    Parameters
    ----------
    nVertices : int (optional, the default is 250)
        The number of vertices in the graph.
    **kwargs :
        Any parameters to send to :func:`~minimal_random_graph` or
        :func:`~set_types_pagerank`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A graph with a ``pos`` and ``vType`` vertex property and the ``eType`` and 
        ``edge_length`` edge property.
    """
    g = minimal_random_graph(nVertices, **kwargs)
    g = set_types_pagerank(g, **kwargs)
    return g


def minimal_random_graph(nVertices, is_directed=True, sfdp=None) :
    """Creates a connected random graph.

    This function first places ``nVertices`` points in the unit square randomly
    and selects a radius ``r``. Then for every vertex ``v`` all other vertices
    with Euclidean distance less or equal to ``r`` are connect by an edge. The
    ``r`` choosen is the smallest number such that the graph ends up connected
    at the end of this process.

    If the number of nodes is greater than 200 and ``sfdp`` is ``None`` (the
    default) then the position of the nodes is altered  using ``graph-tool``'s
    :func:`~graph_tool.draw.sfdp_layout` function (with its default arguments).

    Parameters
    ----------
    nVertices : int
        The number of vertices in the graph.
    is_directed : bool (optional, the default is ``True``)
        Specifies whether the graph is directed or not.
    sfdp : bool or None (optional, the default is ``None``)
        Specifies whether to run ``graph-tool``'s :func:`~graph_tool.draw.sfdp_layout` 
        function on the created graph ``g``.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A graph with a ``pos`` vertex property for the vertex positions.
    """
    points  = np.random.random((nVertices, 2)) * 10
    nEdges  = nVertices * (nVertices - 1) // 2
    edges   = []

    for k in range(nVertices) :
        for j in range(k+1, nVertices) :
            v = points[k] - points[j]
            edges.append( (k, j, v[0]**2 + v[1]**2) )

    cluster = 2
    mytype  = [('n1', int), ('n2', int), ('distance', float)]
    edges   = np.array(edges, dtype=mytype)
    edges   = np.sort(edges, order='distance')
    unionF  = UnionFind([k for k in range(nVertices)])

    for n1, n2, d in edges :
        max_space = d
        unionF.union(n1, n2)
        if unionF.nClusters == cluster - 1 :
            break

    for r in [np.sqrt(max_space) * (1 + 0.1 * k) for k in range(10)] :
        g, pos  = gt.geometric_graph(points, r)
        comp, a = gt.label_components(g)
        if max(comp.a) == 0 :
            break

    if is_directed :
        g.set_directed(True)

    g2  = g.copy()
    for e in g2.edges() :
        e1  = g.add_edge(source=int(e.target()), target=int(e.source()))

    g.reindex_edges()
    if (nVertices > 200 and sfdp is None) or sfdp :
        pos = gt.sfdp_layout(g)
    
    g.vp['pos'] = pos
    return g
