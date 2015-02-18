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
        raise TypeError('eType must by either a dict, list, or numpy.ndarray')

    return other_dict


def _adjacency_adjust(adjacency, eType, adjust, is_directed) :
    """Takes an adjacency list and returns a (possibly) modified adjacency list."""

    if eType is None :
        for adj in adjacency.values() :
            eType.append([1 for k in adj])
    else :
        if len(adjacency) != len(eType) :
            raise RuntimeError("Graph for edge types must match graph from adjacency list/matrix.")
        for k in adjacency.keys() :
            if len(adjacency[k]) != len(eType[k]) :
                raise RuntimeError("Graph for edge types must match graph from adjacency list/matrix.")

    ok_adj = True

    for v, adj in adjacency.items() :
        if len(adj) == 0 :
            if eType is None :
                ok_adj = False
                break
            else :
                for u, adj2 in adjacency.items() :
                    for k, w in enumerate(adj2) :
                        if v == w and eType[u][k] != 0 :
                            ok_adj = False
                            break

    if not ok_adj and is_directed :
        if adjust == 1 :
            null_nodes = set()

            for k, adj in adjacency.items() :
                if len(adj) == 0 :
                    null_nodes.add(k)

            for k, adj in adjacency.items() :
                et = eType[k].copy()
                eType[k] = [0 if v in null_nodes else et[j] for j, v in enumerate(adj)]

        else :
            for k, adj in adjacency.items() :
                if len(adj) == 0 :
                    adjacency[k].append(k)
                    eType[k].append(0)

    return adjacency, eType


def adjacency2graph(adjacency, eType=None, adjust=0, is_directed=True) :
    """Takes an adjacency list, dict, or matrix and returns a graph.

    The purpose of this function is take an adjacency list (or matrix) and
    return a :class:`~graph_tool.Graph` that can be used with
    :class:`.QueueNetwork`. The Graph returned has an ``eType`` edge property.
    If the adjacency is directed and not connected, then the adjacency list is
    altered.

    Parameters
    ----------
    adjacency : list, dict, or numpy.ndarray
        An adjacency list, dict, or matrix.
    eType : list, dict, or numpy.ndarray (optional)
        A mapping that corresponds to that edges ``eType``. For example, if
        ``eType`` is a matrix, then ``eType[u, v]`` is the type of
        queue along the edge between vertices ``u`` and ``v``. If ``eType``
        is not supplied then all but terminal edges have type 1.
    adjust : int (optional, the default is 0)
        Specifies what to do when the graph has terminal vertices (nodes with
        no out-edges). There are three choices:

            ``adjust = 0``: A loop is added to each terminal node in the
            graph, and their ``eType`` of that edge is set to 0.
            ``adjust = 1``: All edges leading to terminal nodes have their
            ``eType`` set to 0.

        Note that if ``adjust`` is not 1 or 2 then it assumed to be 0.
    is_directed : bool (optional, the default is True)
        Sets whether the returned graph is directed or not.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A :class:`~graph_tool.Graph` with the ``eType`` edge property.

    Raises
    ------
    TypeError
        Is raised if ``adjacency`` or ``eType`` is not a :class:`.list`\,
        :class:`.dict`\, :class:`~numpy.ndarray` the (``eType`` can be 
        ``None``\).
    RuntimeError
        A :exc:`~RuntimeError` is raised if ``eType`` does not have the 
        same dimensions as ``adjacency``\.

    Examples
    --------
    If terminal nodes are such that all in-edges have edge type 0 then nothing
    is changed

    >>> adj = { 0 : [1], 1 : [2], 2 : [3, 4], 3 : [2] }
    >>> eTy = { 0 : [1], 1 : [2], 2 : [4, 0], 3 : [3] }
    >>> g = qt.adjacency2graph(adj, eType=eTy)
    >>> ans = qt.graph2dict(g)
    >>> ans[0]
    {0: [1], 1: [2], 2: [3, 4], 3: [2], 4: []}
    >>> ans[1]
    {0: [1], 1: [2], 2: [4, 0], 3: [3], 4: []}

    If this is not the case, then the graph is adjusted by adding a loop with
    eType 0 to the terminal edge:

    >>> eTy = { 0 : [1], 1 : [2], 2 : [4, 5], 3 : [3] }
    >>> g = qt.adjacency2graph(adj, eType=eTy)
    >>> ans = qt.graph2dict(g)
    >>> ans[0]
    {0: [1], 1: [2], 2: [3, 4], 3: [2], 4: [4]}
    >>> ans[1]
    {0: [1], 1: [2], 2: [4, 5], 3: [3], 4: [0]}

    Alternatively, you could have this function adjust the edges that lead to
    terminal vertices by changing their edge type to 0:

    >>> eTy = { 0 : [1], 1 : [2], 2 : [4, 5], 3 : [3] }
    >>> g = qt.adjacency2graph(adj, eType=eTy, adjust=1)
    >>> ans = qt.graph2dict(g)
    >>> ans[0]
    {0: [1], 1: [2], 2: [3, 4], 3: [2], 4: [4]}
    >>> ans[1]
    {0: [1], 1: [2], 2: [4, 0], 3: [3], 4: []}
    """
    if isinstance(adjacency, np.ndarray) :
        adjacency = _matrix2dict(adjacency)
    elif isinstance(adjacency, dict) :
        adjacency = _dict2dict(adjacency)
    elif isinstance(adjacency, list) :
        adjacency = _list2dict(adjacency)
    else :
        raise TypeError("If the adjacency parameter is supplied it must be a list, dict, or a numpy.ndarray.")

    if eType is not None :
        eType = _other2dict(adjacency, eType)

    adjacency, eType = _adjacency_adjust(adjacency, eType, adjust, is_directed)

    nV  = len(adjacency)
    g   = gt.Graph()
    vs  = g.add_vertex(nV)

    g.set_directed(is_directed)

    eT = g.new_edge_property("int")

    for u, adj in adjacency.items() :
        for j, v in enumerate(adj) :
            e = g.add_edge(u, v)
            eT[e] = eType[u][j]

    g.ep['eType'] = eT
    return g


def generate_transition_matrix(g, seed=None) :
    """Generates a random transition matrix for the graph g.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
    seed : int (optional)
        An integer used to initialize ``numpy``\'s psuedorandom number
        generators.

    Returns
    -------
    mat : :class:`~numpy.ndarray`
        Returns a transition matrix where ``mat[i,j]`` is the probability of
        transitioning from vertex ``i`` to vertex ``j``\. If there is no edge
        connecting vertex ``i`` to vertex ``j`` then ``mat[i,j] = 0``\.
    """
    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)

    nV  = g.num_vertices()
    mat = np.zeros( (nV, nV) )

    for v in g.vertices() :
        vi  = int(v)
        ind = [int(e.target()) for e in v.out_edges()]
        deg = len(ind)
        if deg == 1 :
            mat[vi, ind] = 1
        elif deg > 1 :
            probs = np.ceil(np.random.rand(deg) * 100) / 100
            if np.isclose(np.sum(probs), 0) :
                probs[np.random.randint(deg)]  = 1

            mat[vi, ind] = probs / np.sum(probs)

    return mat
            


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
        A graph with a ``pos`` vertex property and the ``eType`` edge property.

    Examples
    --------
    The following generates a directed graph with 50 vertices where half the
    edges are type 1 and 1/4th are type 2 and 1/4th are type 3:

    >>> g = qt.generate_random_graph(nVertices=50, pTypes={1: 0.5, 2: 0.25, 3: 0.25})

    To make an undirected graph with 25 vertices where there are 4 different
    edge types with random proportions:

    >>> p = np.random.rand(4)
    >>> p = {k + 1: p[k] / sum(p) for k in range(4)}
    >>> g = qt.generate_random_graph(nVertices=25, is_directed=False, pTypes=p)
    """
    g = minimal_random_graph(nVertices, **kwargs)
    g = set_types_random(g, **kwargs)
    return g


def generate_pagerank_graph(nVertices=250, **kwargs) :
    """Creates a random graph where the edge and vertex types are selected 
    using the :func:`.set_types_pagerank` method.

    Calls :func:`.minimal_random_graph` and then calls :func:`.set_types_pagerank`.

    Parameters
    ----------
    nVertices : int (optional, the default is 250)
        The number of vertices in the graph.
    **kwargs :
        Any parameters to send to :func:`.minimal_random_graph` or
        :func:`.set_types_pagerank`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A graph with a ``pos`` vertex property and the ``eType`` edge property.
    """
    g = minimal_random_graph(nVertices, **kwargs)
    g = set_types_pagerank(g, **kwargs)
    return g


def minimal_random_graph(nVertices, is_directed=True, sfdp=None, seed=None, **kwargs) :
    """Creates a connected random graph.

    This function first places ``nVertices`` points in the unit square
    randomly. Then, for every vertex ``v``, all other vertices with Euclidean
    distance less or equal to ``r`` are connect by an edge --- where ``r`` is
    the smallest number such that the graph ends up connected at the end of
    this process.

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
        Specifies whether to run ``graph-tool``'s :
        func:`~graph_tool.draw.sfdp_layout` function on the graph ``g``. If
        ``True``, the vertex positions returned by
        func:`~graph_tool.draw.sfdp_layout` are used to set the ``pos`` vertex
        property.
    seed : int (optional)
        An integer used to initialize ``numpy``\'s and ``graph-tool``\'s
        psuedorandom number generators.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A graph with a ``pos`` vertex property for the vertex positions.
    """
    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)
        gt.seed_rng(seed)

    points  = np.random.random((nVertices, 2)) * 10
    nEdges  = nVertices * (nVertices - 1) // 2
    edges   = []

    for k in range(nVertices-1) :
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
