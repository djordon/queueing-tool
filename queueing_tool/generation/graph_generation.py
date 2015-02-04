import graph_tool.all as gt
import numpy          as np
import copy

from .graph_preparation import set_types_random, set_types_pagerank
from .union_find        import UnionFind

def matrix2list(matrix) :
    """Takes an adjacency matrix and returns an adjacency list.

    """
    n   = len(matrix)
    adj = [ [] for k in range(n)]
    for k in range(n) :
        for j in range(n) :
            if matrix[k, j] :
                adj[k].append(j)

    return adj


def ematrix2list(adjacency, matrix) :
    """Takes an adjacency matrix and returns an adjacency list.

    """
    n   = len(matrix)
    adj = [ [] for k in range(n)]
    for k in range(n) :
        for j in adjacency[k] :
            adj[k].append(int(matrix[k, j]))

    return adj


def adjacency2edgetype(adjacency) :
    """Takes an adjacency list and returns a list of types.

    """
    n   = len(adjacency)
    ety = [ [] for k in range(n)]
    if not isinstance(adjacency, list) :
        adj = [ [] for k in range(n)]
        for k in range(n) :
            for j in range(n) :
                if adjacency[k, j] :
                    adj[k].append(j)
    else :
        adj = adjacency

    for k in range(n) :
        for j in adj[k] :
            ety[k].append( 1 if len(adj[j]) else 0 )

    return ety


def adjacency2graph(adjacency, edge_types=None, edge_lengths=None) :
    """Takes an adjacency list and returns a graph.

    """
    nV  = len(adjacency)
    g   = gt.Graph()
    vs  = g.add_vertex(nV)

    if not isinstance(adjacency, list) :
        adjacency = matrix2list(adjacency)

    for k in range(nV) :
        adj = adjacency[k]
        for j in adj :
            e = g.add_edge(k, j)

    if edge_types is not None and not isinstance(edge_types, list) :
        edge_types  = ematrix2list(adjacency, edge_types)

    if edge_types is None :
        edge_types = copy.deepcopy(adjacency)
        for k, adj in enumerate(adjacency) :
            edge_types[k] = [1 for j in range(len(adj))]


    if edge_lengths is None :
        edge_lengths = copy.deepcopy(adjacency)
        for k, adj in enumerate(adjacency) :
            edge_lengths[k] = [1 for j in range(len(adj))]

    vType   = g.new_vertex_property("int")
    eType   = g.new_edge_property("int")
    elength = g.new_edge_property("double")

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
