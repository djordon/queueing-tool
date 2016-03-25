import numbers
import copy

import networkx as nx
import numpy as np

from queueing_tool.graph.graph_functions import _test_graph
from queueing_tool.graph.graph_wrapper import QueueNetworkDiGraph
from queueing_tool.union_find import UnionFind


def _calculate_distance(latlon1, latlon2):
    """Calculates the distance between two points on earth.
    """
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2
    R     = 6371          # radius of the earth in kilometers
    dlon  = lon2 - lon1
    dlat  = lat2 - lat1
    a = np.sin(dlat / 2.)**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.))**2
    c = 2 * np.pi * R * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) / 180.
    return c


def generate_transition_matrix(g, seed=None):
    """Generates a random transition matrix for the graph ``g``.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :class:`numpy.ndarray`, dict, etc.
        Any object that :any:`DiGraph<networkx.DiGraph>` accepts.
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number
        generator.

    Returns
    -------
    mat : :class:`~numpy.ndarray`
        Returns a transition matrix where ``mat[i, j]`` is the
        probability of transitioning from vertex ``i`` to vertex ``j``.
        If there is no edge connecting vertex ``i`` to vertex ``j``
        then ``mat[i, j] = 0``.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    nV  = g.number_of_nodes()
    mat = np.zeros((nV, nV))

    for v in g.nodes():
        ind = [e[1] for e in g.out_edges(v)]
        deg = len(ind)
        if deg == 1:
            mat[v, ind] = 1
        elif deg > 1:
            probs = np.ceil(np.random.rand(deg) * 100) / 100.
            if np.isclose(np.sum(probs), 0):
                probs[np.random.randint(deg)]  = 1

            mat[v, ind] = probs / np.sum(probs)

    return mat



def generate_random_graph(nVertices=250, **kwargs):
    """Creates a random graph where the edge and with different
    vertex types.

    The vertex types are selected using the
    :func:`.set_types_random` method. This method calls
    :func:`.minimal_random_graph` and then calls
    :func:`.set_types_random`.

    Parameters
    ----------
    nVertices : int (optional, default: 250)
        The number of vertices in the graph.
    **kwargs :
        Any parameters to send to :func:`.minimal_random_graph` or
        :func:`.set_types_random`.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        A graph with the position of the vertex set as a property.
        The position property is called ``pos``. Also, the ``eType``
        edge property is set for each edge.

    Examples
    --------
    The following generates a directed graph with 50 vertices where half
    the edges are type 1 and 1/4th are type 2 and 1/4th are type 3:

    >>> import queueing_tool as qt
    >>> pTypes = {1: 0.5, 2: 0.25, 3: 0.25}
    >>> g = qt.generate_random_graph(50, pTypes=pTypes, seed=15)
    >>> p1 = np.sum([g.ep(e, 'eType') == 1 for e in g.edges()])
    >>> float(p1) / g.number_of_edges()
    0.5
    >>> p2 = np.sum([g.ep(e, 'eType') == 2 for e in g.edges()])
    >>> float(p2) / g.number_of_edges() # doctest: +ELLIPSIS
    0.251...
    >>> p3 = np.sum([g.ep(e, 'eType') == 3 for e in g.edges()])
    >>> float(p3) / g.number_of_edges() # doctest: +ELLIPSIS
    0.248...

    To make an undirected graph with 25 vertices where there are 4
    different edge types with random proportions:

    >>> p = np.random.rand(4)
    >>> p = {k + 1: p[k] / sum(p) for k in range(4)}
    >>> g = qt.generate_random_graph(nVertices=25, is_directed=False, pTypes=p)

    Note that none of the edge types in the above example are 0. It is
    recommended let use edge type indices starting at 1, since 0 is
    typically used for terminal edges.
    """
    g = minimal_random_graph(nVertices, **kwargs)
    g = set_types_random(g, **kwargs)
    return g


def generate_pagerank_graph(nVertices=250, **kwargs):
    """Creates a random graph where the edge and vertex types are
    selected using the :func:`.set_types_rank` method.

    Calls :func:`.minimal_random_graph` and then calls
    :func:`.set_types_rank`.

    This function sets the edge types of a graph to be either 1, 2, or
    3. It sets the vertices to type 2 by selecting the top
    ``pType2 * g.number_of_nodes()`` vertices given by the
    :func:`~networkx.pagerank` of the graph. A loop is added
    to all vertices identified this way (if one does not exist
    already). It then randomly sets vertices close to the type 2
    vertices as type 3, and adds loops to these vertices as well. These
    loops then have edge types the correspond to the vertices type. The
    rest of the edges are set to type 1.

    Parameters
    ----------
    nVertices : int (optional, the default is 250)
        The number of vertices in the graph.
    **kwargs :
        Any parameters to send to :func:`.minimal_random_graph` or
        :func:`.set_types_rank`.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        A graph with a ``pos`` vertex property and the ``eType`` edge
        property.
    """
    g = minimal_random_graph(nVertices, **kwargs)
    r = np.zeros(nVertices)
    for k, pr in nx.pagerank(g).items():
        r[k] = pr
    g = set_types_rank(g, rank=r, **kwargs)
    return g


def minimal_random_graph(nVertices, seed=None, **kwargs):
    """Creates a connected graph by selecting vertex locations.

    Parameters
    ----------
    nVertices : int
        The number of vertices in the graph.
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number
        generators.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        A graph with a ``pos`` vertex property for each vertex's
        position.

    Notes
    -----
    This function first places ``nVertices`` points in the unit square
    randomly. Then, for every vertex ``v``, all other vertices with
    Euclidean distance less or equal to ``r`` are connect by an edge
    --- where ``r`` is the smallest number such that the graph ends up
    connected at the end of this process.
    """
    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    points = np.random.random((nVertices, 2)) * np.float(10)
    nEdges = nVertices * (nVertices - 1) // 2
    edges  = []

    for k in range(nVertices-1):
        for j in range(k+1, nVertices):
            v = points[k] - points[j]
            edges.append( (k, j, v[0]**2 + v[1]**2) )

    mytype = [('n1', int), ('n2', int), ('distance', np.float)]
    edges  = np.array(edges, dtype=mytype)
    edges  = np.sort(edges, order='distance')
    unionF = UnionFind([k for k in range(nVertices)])

    g = nx.Graph()

    for n1, n2, d in edges:
        max_space = d
        unionF.union(n1, n2)
        g.add_edge(n1, n2)
        if unionF.nClusters == 1:
            break

    pos = {j: p for j, p in enumerate(points)}
    g = QueueNetworkDiGraph(g.to_directed())
    g.set_pos(pos)

    #max_space = np.sqrt(max_space)

    #pos = {k: p for k, p in enumerate(points)}
    #for k in range(20):
    #    r = max_space * (1 + 0.1 * k)
    #    g = nx.random_geometric_graph(nVertices, r, pos=pos)
    #    nCC = nx.number_connected_components(g)
    #    if nCC == 1:
    #        break

    return g#QueueNetworkDiGraph(g.to_directed())


def set_types_random(g, pTypes=None, seed=None, **kwargs):
    """Randomly sets ``eType`` (edge type) properties of the graph.

    This function randomly assigns each edge a type. The probability of
    an edge being a specific type is proscribed in the ``pTypes``
    variable.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :class:`numpy.ndarray`, dict, etc.
        Any object that :any:`DiGraph<networkx.DiGraph>` accepts.
    pTypes : dict (optional)
        A dictionary of types and proportions, where the keys are the
        types and the values are the proportion of edges that are
        expected to be of that type. The values can must be proportions
        (that add to one).
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number
        generator.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        Returns the a graph with an ``eType`` edge property.

    Raises
    ------
    TypeError
        Raised when the parameter ``g`` is not of a type that can be
        made into a :any:`networkx.DiGraph`.

    ValueError
        Raises a :exc:`~ValueError` if the ``pType`` values do not sum
        to one.

    Notes
    -----
    If ``pTypes`` is not explicitly specified in the arguments, then it
    defaults to three types in the graph (types 1, 2, and 3) and sets
    their proportions to be 1/3 each.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    if pTypes is None:
        pTypes = {k : 1.0 / 3 for k in range(1, 4)}

    nEdges  = g.number_of_edges()
    edges   = [e for e in g.edges()]
    cut_off = np.cumsum( np.array(list(pTypes.values())) )

    if np.isclose(cut_off[-1], 1.0):
        cut_off = np.array(np.round(cut_off * nEdges)).astype(int)
    else:
        msg = "pTypes values must sum to one."
        raise ValueError("pTypes values must sum to one.")

    np.random.shuffle(edges)
    eTypes = {}
    for k, key in enumerate(pTypes.keys()):
        if k == 0:
            for e in edges[:cut_off[k]]:
                eTypes[e] = key
        else:
            for e in edges[cut_off[k-1]:cut_off[k]]:
                eTypes[e] = key

    g.new_edge_property('eType')
    for e in g.edges():
        g.set_ep(e, 'eType', eTypes[e])

    return g


def set_types_rank(g, rank, pType2=0.1, pType3=0.1, seed=None, **kwargs):
    """Creates a stylized graph. Sets edge and types using `pagerank`_.

    This function sets the edge types of a graph to be either 1, 2, or
    3. It sets the vertices to type 2 by selecting the top
    ``pType2 * g.number_of_nodes()`` vertices given by the
    :func:`~networkx.pagerank` of the graph. A loop is added
    to all vertices identified this way (if one does not exist
    already). It then randomly sets vertices close to the type 2
    vertices as type 3, and adds loops to these vertices as well. These
    loops then have edge types the correspond to the vertices type. The
    rest of the edges are set to type 1.

    .. _pagerank: http://en.wikipedia.org/wiki/PageRank

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :class:`~numpy.ndarray`, dict, etc.
        Any object that :any:`DiGraph<networkx.DiGraph>` accepts.
    pType2 : float (optional, default: 0.1)
        Specifies the proportion of vertices that will be of type 2.
    pType3 : float (optional, default: 0.1)
        Specifies the proportion of vertices that will be of type 3 and
        that are near pType2 vertices.
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number
        generators.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        Returns the a graph with an ``eType`` edge property.

    Raises
    ------
    TypeError
        Raised when the parameter ``g`` is not of a type that can be
        made into a :any:`DiGraph<networkx.DiGraph>`.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    tmp    = np.sort(np.array(rank))
    nDests = int(np.ceil(g.number_of_nodes() * pType2))
    dests  = np.where(rank >= tmp[-nDests])[0]

    if 'pos' not in g.vertex_properties():
        g.set_pos()

    dest_pos   = np.array([g.vp(v, 'pos') for v in dests])
    nFCQ       = int(pType3 * g.number_of_nodes())
    min_g_dist = np.ones(nFCQ) * np.infty
    ind_g_dist = np.ones(nFCQ, int)

    r, theta = np.random.random(nFCQ) / 500., np.random.random(nFCQ) * 360.
    xy_pos   = np.array([r * np.cos(theta), r * np.sin(theta)]).transpose()
    g_pos    = xy_pos + dest_pos[np.array(np.mod(np.arange(nFCQ), nDests), int)]

    for v in g.nodes():
        if v not in dests:
            tmp = np.array([_calculate_distance(g.vp(v, 'pos'), g_pos[k, :]) for k in range(nFCQ)])
            min_g_dist = np.min((tmp, min_g_dist), 0)
            ind_g_dist[min_g_dist == tmp] = v

    ind_g_dist = np.unique(ind_g_dist)
    fcqs  = set(ind_g_dist[:min( (nFCQ, len(ind_g_dist)) )])
    dests = set(dests)
    g.new_vertex_property('loop_type')

    for v in g.nodes():
        if v in dests:
            g.set_vp(v, 'loop_type', 3)
            if not g.is_edge((v, v)):
                g.add_edge(v, v)
        elif v in fcqs:
            g.set_vp(v, 'loop_type', 2)
            if not g.is_edge((v, v)):
                g.add_edge(v, v)

    g.new_edge_property('eType')
    for e in g.edges():
        g.set_ep(e, 'eType', 1)

    for v in g.nodes():
        if g.vp(v, 'loop_type') in [2, 3]:
            e = (v, v)
            if g.vp(v, 'loop_type') == 2:
                g.set_ep(e, 'eType', 2)
            else:
                g.set_ep(e, 'eType', 3)

    return g
