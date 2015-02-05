import graph_tool.all as gt
import numpy          as np
import copy

from .graph_preparation import set_types_random, set_types_pagerank
from .union_find        import UnionFind

def _matrix2list(matrix) :
    """Takes an adjacency matrix and returns an adjacency list."""
    n   = len(matrix)
    adj = [ [] for k in range(n)]
    for k in range(n) :
        for j in range(n) :
            if matrix[k, j] :
                adj[k].append(j)

    return adj


def _dict2list(adj_dict) :
    """Takes a dictionary representation of an adjacency list and returns
    a list based representation.
    """
    vertives = set()
    for key, value in adj_dict.items() :
        vertices.add(key)
        if isinstance(value, int) :
            vertices.add(value)
        else :
            vertices.update(value)

    adj = []
    if min(vertices) == 1 :
        for k in range(1, max(vertices) + 1 ) :
            if k in adj_dict :
                if isinstance(adj_dict[k], int) :
                    adj.append( [adj_dict[k]-1] )
                else :
                    adj.append( [j - 1 for j in adj_dict[k]] )
            else :
                adj.append([])
    else :
        for k in range(max(vertices) + 1 ) :
            if k in adj_dict :
                if isinstance(adj_dict[k], int) :
                    adj.append( [adj_dict[k]] )
                else :
                    adj.append( adj_dict[k] )
            else :
                adj.append([])
    return adj


def _adjacency_adjust(adjacency, edge_types, adjust_type, is_directed) :
    """Takes an adjacency list and returns a (possibly) modified adjacency list."""

    ok_adj = True
    eTypes = []

    for adj in adjacency :
        if len(adj) == 0 :
            ok_adj = False
            break

    if edge_types is None :
        for adj in adjacency :
            eTypes.append([1 for k in adj])
    else :
        for k in range(n) :
            if len(adjacency[k]) != len(eTypes[k]) :
                raise RuntimeError("Supplied edge types must match adjacency list/matrix.")

    if not ok_adj and is_directed :
        if adjust_type == 1 :
            null_nodes = set()

            for k, adj in enumerate(adjacency) :
                if len(adj) == 0 :
                    null_nodes.add(k)

            for k, adj in enumerate(eTypes) :
                eTypes[k] = [0 if j in null_nodes else j for j in adj]

        elif adjust_type == 2 :
            for k, adj in enumerate(adjacency) :
                if len(adj) == 0 :
                    adjacency[k].append(n + 1)
                    eTypes[k].append(0)

            adjacency.append([])
            eTypes.append([])

        else :
            for k, adj in enumerate(adjacency) :
                if len(adj) == 0 :
                    adjacency[k].append(k)
                    eTypes[k].append(0)

    return adjacency, eTypes



def adjacency2graph(adjacency, edge_types=None, edge_lengths=None, adjust_type=0, is_directed=True) :
    """Takes an adjacency list, dict, or matrix and returns a graph.

    The purpose of this function is take an adjacency list (or matrix) and
    return a :class:`~graph_tool.Graph` that can be used with
    :class:`~queueing_tool.network.QueueNetwork`. The Graph returned has a 
    ``vType`` vertex property, and ``eType`` and ``edge_length`` edge 
    properties. If the adjacency is directed and not connected, then the
    adjacency list is altered.

    Parameters
    ----------
    adjacency : list, dict, or numpy.ndarray
        An adjacency list, dict, or matrix.
    edge_types : list, dict, or numpy.ndarray (optional)
        A mapping that corresponds to that edges ``eType``. For example, if
        ``edge_types`` is a matrix, then ``edge_type[u, v]`` is the type of
        queue along the edge between vertices ``u`` and ``v``. If ``edge_types``
        is not supplied then all but terminal edges have type 1.
    edge_lengths : list, dict, or numpy.ndarray (optional)
        A mapping where that corresponds to each edge's length. If not supplied,
        then each edge has length 1.
    adjust_type : int (optional, the default is 0)
        Specifies what to do when the graph has terminal vertices (nodes with no
        out-edges). There are three choices:

            ``adjust_type = 0``: A loop is added to each terminal node in the graph,
            and their ``eType`` of that edge is set to 0.
            ``adjust_type = 1``: All edges leading to terminal nodes have their 
            ``eType`` set to 0.
            ``adjust_type = 2``: A new vertex is created and each node has an edge
            connect to the new vertex. The ``eType`` for each of these new edges is
            set to 0.

        Note that if ``adjust_type`` is not 1 or 2 then it assumed to be 0.
    is_directed : bool (optional, the default is True)
        Sets whether the returned graph is directed or not.

    Returns
    -------
    out : :class:`~graph_tool.Graph`
        A :class:`~graph_tool.Graph` with the ``vType`` vertex property, and 
        ``eType`` and ``edge_length`` edge properties.

    Raises
    ------
    TypeError
        Is raised if ``adjacency``, ``edge_types``, and ``edge_lengths`` are
        not ``list``, ``dict``, or  ``numpy.ndarray`` types (``edge_types`` and 
        ``edge_lengths`` can be ``None``).
    RuntimeError
        A :exc:`~RuntimeError` is raised if ``edge_types`` does not have the 
        same dimensions as ``adjacency``.
    """
    if adjacency is None :
        raise TypeError("The `adjacency` parameter must be a list, dict, or a numpy.ndarray.")

    params = [adjacency, edge_types, edge_lengths]
    string = ['adjacency', 'edge_types', 'edge_lengths']

    for k, param in enumerate(params) :
        if param is not None :
            if isinstance(param, np.ndarray) :
                params[k] = _matrix2list(param)
            elif isinstance(param, dict) :
                params[k] = _dict2list(param)
            elif not isinstance(param, list) :
                raise TypeError("The `%s` parameter must be a list, dict, or a numpy.ndarray." % (string[k]) )

    adjacency, edge_types, edge_lengths = params
    adjacency, edge_types = _adjacency_adjust(adjacency, edge_types, adjust_type, is_directed)

    nV  = len(adjacency)
    g   = gt.Graph()
    vs  = g.add_vertex(nV)

    g.set_directed(is_directed)

    if edge_lengths is None :
        edge_lengths = [0 for k in range(adjacency)]
        for k, adj in enumerate(adjacency) :
            edge_lengths[k] = [1 for j in range(len(adj))]

    if len(edge_lengths) != len(adjacency) :
        edge_lengths.append([])

    for k in range(nV) :
        if len(edge_lengths[k]) != len(adjacency[k]) :
            edge_lengths[k].append(1)

    vType   = g.new_vertex_property("int")
    eType   = g.new_edge_property("int")
    elength = g.new_edge_property("double")
    vType.a = 1

    for u, adj in enumerate(adjacency) :
        for j, v in enumerate(adj) :
            e = g.edge(u, v) 
            eType[e]    = edge_types[u][j]
            elength[e]  = edge_lengths[u][j]
            if u == v :
                vType[e.source()] = edge_types[u][j]

    g.vp['vType'] = vType
    g.ep['eType'] = eType
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
        Any parameters to send to :func:`~minimal_random_graph` or :func:`~set_types_random`.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        A graph with a ``pos`` and ``vType`` vertex property and the ``eType`` and 
        ``edge_length`` edge property.
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
        Any parameters to send to :func:`~minimal_random_graph` or :func:`~set_types_pagerank`.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
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
    g : :class:`~graph_tool.Graph`
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
