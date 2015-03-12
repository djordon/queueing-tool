import graph_tool.all as gt
import numpy   as np
import numbers
import copy

from .union_find        import UnionFind


def _test_graph(g) :
    """A function that makes sure ``g`` is either a :class:`~graph_tool.Graph` or 
     a string or file object to one.

    Parameters
    ----------
    g : A **str** or a :class:`~graph_tool.Graph`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        If ``g`` is a string or a file object then the output given by
        ``graph_tool.load_graph(g, fmt='xml')``, if ``g`` is aready a 
        :class:`~graph_tool.Graph` then it is returned unaltered.

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`\.
    """
    if isinstance(g, str) :
        g = gt.load_graph(g, fmt='xml')
    elif not isinstance(g, gt.Graph) :
        raise TypeError("Need to supply a graph-tool graph or the location of a graph")
    return g


def _calculate_distance(latlon1, latlon2) :
    """Calculates the distance between two points on earth.
    """
    lat1, lon1  = latlon1
    lat2, lon2  = latlon2
    R     = 6371          # radius of the earth in kilometers
    dlon  = lon2 - lon1
    dlat  = lat2 - lat1
    a     = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c     = 2 * np.pi * R * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) / 180
    return c


def _matrix2dict(matrix) :
    """Takes an adjacency matrix and returns an adjacency list."""
    n   = len(matrix)
    adj = {k : [] for k in range(n)}
    for k in range(n) :
        for j in range(n) :
            if matrix[k, j] :
                adj[k].extend([j for i in range(int(matrix[k,j]))])
    
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

    adjacency = {}
    vertices  = list(vertices)
    vertices.sort()

    vs = {v : k for k, v in enumerate(vertices)}

    for key, value in adj_dict.items() :
        if not hasattr(value, '__iter__') :
            adjacency[vs[key]] = [vs[value]]
        else :
            adjacency[vs[key]] = [vs[v] for v in value]

    for v in vertices :
        if vs[v] not in adjacency :
            adjacency[vs[v]] = []

    return adjacency


def _list2dict(adj_list) :
    """Takes a dictionary representation of an adjacency list and returns
    a list based representation.
    """
    adj_dict = {}
    for key, value in enumerate(adj_list) :
        adj_dict[key] = value

    return _dict2dict(adj_dict)


def _other2dict(adj_dict, other) :
    other_dict = {}

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
    elif isinstance(other, np.ndarray) :
        other_dict = copy.deepcopy(adj_dict)
        for k, value in adj_dict.items() :
            for i, j in enumerate(value) :
                other_dict[k][i] = other[k, j]
    else :
        raise TypeError('eType must by either a dict, list, or numpy.ndarray')

    tmp = copy.deepcopy(other_dict)
    for key, value in tmp.items() :
        if not hasattr(value, '__iter__') :
            other_dict[key] = [value]

    return other_dict


def _adjacency_adjust(adjacency, eType, adjust, is_directed) :
    """Takes an adjacency list and returns a (possibly) modified adjacency list."""

    if eType is None :
        eType = {}
        for v, adj in adjacency.items() :
            eType[v] = [1 for k in adj]
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
                #et = eType[k]
                eType[k] = [0 if v in null_nodes else eType[k][j] for j, v in enumerate(adj)]

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
    If the adjacency is directed and not connected, then the adjacency list 
    may be altered.

    Parameters
    ----------
    adjacency : list, dict, or :class:`~numpy.ndarray`
        An adjacency list, dict, or matrix.
    eType : list, dict, or :class:`~numpy.ndarray` (optional)
        A mapping that identifies each edge's ``eType``. For example, if
        ``eType`` is a matrix, then ``eType[u, v]`` is the type of
        queue that lays along the edge between vertices ``u`` and ``v``. If
        ``eType`` is not supplied then all but terminal edges have type 1,
        terminal edges will have type 0.
    adjust : int ``{0, 1}`` (optional, the default is 0)
        Specifies what to do when the graph has terminal vertices (nodes with
        no out-edges). Note that if ``adjust`` is not 0 or 1 then it assumed
        to be 0. There are three choices:

            ``adjust = 0``
                A loop is added to each terminal node in the graph, and their
                ``eType`` of that loop is set to 0.
            ``adjust = 1``
                All edges leading to terminal nodes have their ``eType`` set
                to 0.
        
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
        A :exc:`~RuntimeError` is raised if, when passed, the ``eType``
        parameter does not have the same dimensions as ``adjacency``\.

    Examples
    --------
    If terminal nodes are such that all in-edges have edge type ``0`` then
    nothing is changed

    >>> adj = { 0 : [1], 1 : [2], 2 : [3, 4], 3 : [2] }
    >>> eTy = { 0 : [1], 1 : [2], 2 : [4, 0], 3 : [3] }
    >>> g = qt.adjacency2graph(adj, eType=eTy)
    >>> ans = qt.graph2dict(g)
    >>> ans[0]    # This is the adjacency list
    {0: [1], 1: [2], 2: [3, 4], 3: [2], 4: []}
    >>> ans[1]    # This is the edge types
    {0: [1], 1: [2], 2: [4, 0], 3: [3], 4: []}

    If this is not the case, then the graph is adjusted by adding a loop with
    eType 0 to terminal vertices. In this case, vertex 4 is terminal since it
    does not have any out edges:

    >>> eTy = { 0 : [1], 1 : [2], 2 : [4, 5], 3 : [3] }
    >>> g = qt.adjacency2graph(adj, eType=eTy)
    >>> ans = qt.graph2dict(g)
    >>> ans[0]    # A loop was added to vertex 4
    {0: [1], 1: [2], 2: [3, 4], 3: [2], 4: [4]}
    >>> ans[1]    # The added loop has edge type 0
    {0: [1], 1: [2], 2: [4, 5], 3: [3], 4: [0]}

    Alternatively, you could have this function adjust the edges that lead to
    terminal vertices by changing their edge type to 0:

    >>> eTy = { 0 : [1], 1 : [2], 2 : [4, 5], 3 : [3] }
    >>> g = qt.adjacency2graph(adj, eType=eTy, adjust=1)
    >>> ans = qt.graph2dict(g)
    >>> ans[0]    # The graph is unaltered
    {0: [1], 1: [2], 2: [3, 4], 3: [2], 4: []}
    >>> ans[1]    # The terminal edge's edge type was changed to 0
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
        if is_directed :
            for j, v in enumerate(adj) :
                e = g.add_edge(u, v)
                eT[e] = eType[u][j]
        else :
            for j, v in enumerate(adj) :
                if len(g.edge(u,v,True)) < adj.count(v) :
                    e = g.add_edge(u, v)
                    eT[e] = eType[u][j]

    g.ep['eType'] = eT
    return g


def generate_transition_matrix(g, seed=None) :
    """Generates a random transition matrix for the graph ``g``\.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number generator.

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
        A graph with a ``pos`` vertex property (these are the vertex positions)
        and the ``eType`` edge property.

    Examples
    --------
    The following generates a directed graph with 50 vertices where half the
    edges are type 1 and 1/4th are type 2 and 1/4th are type 3:

    >>> g = qt.generate_random_graph(50, pTypes={1: 0.5, 2: 0.25, 3: 0.25}, seed=15)
    >>> np.sum(g.ep['eType'].a == 1) / g.num_edges()
    0.5
    >>> np.sum(g.ep['eType'].a == 2) / g.num_edges()
    0.25147928994082841
    >>> np.sum(g.ep['eType'].a == 3) / g.num_edges()
    0.24852071005917159

    To make an undirected graph with 25 vertices where there are 4 different
    edge types with random proportions:

    >>> p = np.random.rand(4)
    >>> p = {k + 1: p[k] / sum(p) for k in range(4)}
    >>> g = qt.generate_random_graph(nVertices=25, is_directed=False, pTypes=p)

    Note that none of the edge types in the above example are 0. It is
    recommended let use edge type indices starting at 1, since 0 is typically
    used for terminal edges.
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
    """Creates a connected graph by selecting vertex locations graphly.

    Parameters
    ----------
    nVertices : int
        The number of vertices in the graph.
    is_directed : bool (optional, the default is ``True``)
        Specifies whether the graph is directed or not.
    sfdp : bool or None (optional, the default is ``None``)
        Specifies whether to run graph-tool's
        :func:`~graph_tool.draw.sfdp_layout` function on the graph ``g``.
        If ``True``, the vertex positions returned by
        :func:`~graph_tool.draw.sfdp_layout` are used to set the ``pos``
        vertex property.
    seed : int (optional)
        An integer used to initialize numpy's and graph-tool's psuedorandom
        number generators.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`~graph_tool.Graph`
        A graph with a ``pos`` vertex property for the vertex positions.

    Notes
    -----
    This function first places ``nVertices`` points in the unit square
    randomly. Then, for every vertex ``v``, all other vertices with Euclidean
    distance less or equal to ``r`` are connect by an edge --- where ``r`` is
    the smallest number such that the graph ends up connected at the end of
    this process.

    If the number of nodes is greater than 200 and ``sfdp`` is ``None`` (the
    default) then the position of the nodes is altered  using graph-tool's
    :func:`~graph_tool.draw.sfdp_layout` function (with ``max_iter=10000`` and
    all other parameters set to their default value).
    """
    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)
        gt.seed_rng(seed)

    points  = np.random.random((nVertices, 2)) * np.float(10)
    nEdges  = nVertices * (nVertices - 1) // 2
    edges   = []

    for k in range(nVertices-1) :
        for j in range(k+1, nVertices) :
            v = points[k] - points[j]
            edges.append( (k, j, v[0]**2 + v[1]**2) )

    cluster = 2
    mytype  = [('n1', int), ('n2', int), ('distance', np.float)]
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
        pos = gt.sfdp_layout(g, max_iter=10000)
    
    g.vp['pos'] = pos
    return g


def set_types_random(g, pTypes=None, seed=None, **kwargs) :
    """Randomly sets ``eType`` (edge type) properties of the graph.

    This function randomly assigns each edge a type. The probability of an edge being 
    a specific type is proscribed in the ``pTypes`` variable.

    Parameters
    ----------
    g : A string or a :class:`~graph_tool.Graph`.
    pTypes : dict (optional)
        A dictionary of types and proportions, where the keys are the types
        and the values are the proportion of edges that are expected to be of
        that type. The values can be either proportions (that add to one) or
        the exact number of edges that be set to a type. In the later case, the
        sum of all the values must equal the total number of edges in the
        :class:`~graph_tool.Graph`\.
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number generator.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`~graph_tool.Graph`
        Returns the :class:`~graph_tool.Graph` ``g`` with an ``eType`` edge property.

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`\.

    RuntimeError
        Raises a :exc:`~RuntimeError` if the ``pType`` values do not sum to one
        or does not sum to the number of edges in the graph.
    
    Notes
    -----
    If ``pTypes`` is not explicitly specified in the arguments, then it defaults to three
    types in the graph (types 1, 2, and 3) and sets their proportions to be 1/3 each.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)

    if pTypes is None :
        pTypes = {k : 1.0/3 for k in range(1,4)}

    nEdges  = g.num_edges() 
    edges   = [k for k in range(nEdges)]
    cut_off = np.cumsum( np.array(list(pTypes.values())) )

    if np.isclose(cut_off[-1], 1.0) :
        cut_off = np.array(np.round(cut_off * nEdges)).astype(int)
    elif cut_off[-1] != nEdges :
        raise RuntimeError("pTypes must sum to one, or sum to the number of edges in the graph")

    np.random.shuffle(edges)
    eTypes  = {}
    for k, key in enumerate(pTypes.keys()) :
        if k == 0 :
            for ei in edges[:cut_off[k]] :
                eTypes[ei] = key
        else :
            for ei in edges[cut_off[k-1]:cut_off[k]] :
                eTypes[ei] = key

    eType = g.new_edge_property("int")

    for k, e in enumerate(g.edges()) :
        eType[e] = eTypes[k]
    
    g.ep['eType'] = eType
    return g


def set_types_pagerank(g, pType2=0.1, pType3=0.1, seed=None, **kwargs) :
    """Creates a stylized graph. Sets edge and types using `pagerank`_.

    This function sets the edge types of a graph to be either 1, 2, or 3.
    It sets the vertices to type 2 by selecting the top
    ``pType2 * g.num_vertices()`` vertices given by the
    :func:`~graph_tool.centrality.pagerank` of the graph. A loop is added
    to all vertices identified this way (if one does not exist already). It
    then randomly sets vertices close to the type 2 vertices as type 3, and
    adds loops to these vertices as well. These loops then have edge types the
    correspond to the vertices type. The rest of the edges are set to type 1.

    .. _pagerank: http://en.wikipedia.org/wiki/PageRank

    Parameters
    ----------
    g : A string or a :class:`~graph_tool.Graph`.
    pType2 : float (optional, the default is 0.1)
        Specifies the proportion of vertices that will be of type 2.
    pType3 : float (optional, the default is 0.1)
        Specifies the proportion of vertices that will be of type 3 and that
        are near pType2 vertices.
    seed : int (optional)
        An integer used to initialize numpy's and graph-tool's psuedorandom
        number generators.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`~graph_tool.Graph`
        Returns the :class:`~graph_tool.Graph` ``g`` with the ``eType`` edge
        property.

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`\.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)
        gt.seed_rng(seed)

    pagerank    = gt.pagerank(g)
    tmp         = np.sort(np.array(pagerank.a))
    nDests      = int(np.ceil(g.num_vertices() * pType2))
    dests       = np.where(pagerank.a >= tmp[-nDests])[0]

    if 'pos' not in g.vp :
        pos = gt.sfdp_layout(g, max_iter=10000)
        g.vp['pos'] = pos

    dest_pos    = np.array([g.vp['pos'][g.vertex(k)] for k in dests])
    nFCQ        = int(pType3 * g.num_vertices())
    min_g_dist  = np.ones(nFCQ) * np.infty
    ind_g_dist  = np.ones(nFCQ, int)
    
    r, theta    = np.random.random(nFCQ) / 500, np.random.random(nFCQ) * 360
    xy_pos      = np.array([r * np.cos(theta), r * np.sin(theta)]).transpose()
    g_pos       = xy_pos + dest_pos[np.array( np.mod(np.arange(nFCQ), nDests), int)]
    
    for v in g.vertices() :
        if int(v) not in dests :
            tmp = np.array([_calculate_distance(g.vp['pos'][v], g_pos[k, :]) for k in range(nFCQ)])
            min_g_dist = np.min((tmp, min_g_dist), 0)
            ind_g_dist[min_g_dist == tmp] = int(v)
    
    ind_g_dist  = np.unique(ind_g_dist)
    fcqs        = ind_g_dist[:min( (nFCQ, len(ind_g_dist)) )]
    loop_type   = g.new_vertex_property("int")

    for v in g.vertices() :
        if int(v) in dests :
            loop_type[v] = 3
            if not isinstance(g.edge(v, v), gt.Edge) :
                e = g.add_edge(source=v, target=v)
        elif int(v) in fcqs :
            loop_type[v] = 2
            if not isinstance(g.edge(v, v), gt.Edge) :
                e = g.add_edge(source=v, target=v)
    
    g.reindex_edges()
    eType     = g.new_edge_property("int")
    eType.a  += 1

    for v in g.vertices() :
        if loop_type[v] in [2, 3] :
            e = g.edge(v, v)
            if loop_type[v] == 2 :
                eType[e] = 2
            else :
                eType[e] = 3
    
    g.ep['eType'] = eType
    return g
