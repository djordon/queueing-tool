import numbers
import warnings

import networkx as nx
import numpy as np

from queueing_tool.graph.graph_functions import _test_graph, _calculate_distance
from queueing_tool.graph.graph_wrapper import QueueNetworkDiGraph
from queueing_tool.union_find import UnionFind


def generate_transition_matrix(g, seed=None):
    """Generates a random transition matrix for the graph ``g``.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :class:`numpy.ndarray`, dict, etc.
        Any object that :any:`DiGraph<networkx.DiGraph>` accepts.
    seed : int (optional)
        An integer used to initialize numpy's psuedo-random number
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

    nV = g.number_of_nodes()
    mat = np.zeros((nV, nV))

    for v in g.nodes():
        ind = [e[1] for e in sorted(g.out_edges(v))]
        deg = len(ind)
        if deg == 1:
            mat[v, ind] = 1
        elif deg > 1:
            probs = np.ceil(np.random.rand(deg) * 100) / 100.
            if np.isclose(np.sum(probs), 0):
                probs[np.random.randint(deg)] = 1

            mat[v, ind] = probs / np.sum(probs)

    return mat


def generate_random_graph(num_vertices=250, prob_loop=0.5, **kwargs):
    """Creates a random graph where the edges have different types.

    This method calls :func:`.minimal_random_graph`, and then adds
    a loop to each vertex with ``prob_loop`` probability. It then
    calls :func:`.set_types_random` on the resulting graph.

    Parameters
    ----------
    num_vertices : int (optional, default: 250)
        The number of vertices in the graph.
    prob_loop : float (optional, default: 0.5)
        The probability that a loop gets added to a vertex.
    **kwargs :
        Any parameters to send to :func:`.minimal_random_graph` or
        :func:`.set_types_random`.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        A graph with the position of the vertex set as a property.
        The position property is called ``pos``. Also, the ``edge_type``
        edge property is set for each edge.

    Examples
    --------
    The following generates a directed graph with 50 vertices where half
    the edges are type 1 and 1/4th are type 2 and 1/4th are type 3:

    >>> import queueing_tool as qt
    >>> pTypes = {1: 0.5, 2: 0.25, 3: 0.25}
    >>> g = qt.generate_random_graph(100, proportions=pTypes, seed=17)
    >>> non_loops = [e for e in g.edges() if e[0] != e[1]]
    >>> p1 = np.sum([g.ep(e, 'edge_type') == 1 for e in non_loops])
    >>> float(p1) / len(non_loops) # doctest: +ELLIPSIS
    0.486...
    >>> p2 = np.sum([g.ep(e, 'edge_type') == 2 for e in non_loops])
    >>> float(p2) / len(non_loops) # doctest: +ELLIPSIS
    0.249...
    >>> p3 = np.sum([g.ep(e, 'edge_type') == 3 for e in non_loops])
    >>> float(p3) / len(non_loops) # doctest: +ELLIPSIS
    0.264...

    To make an undirected graph with 25 vertices where there are 4
    different edge types with random proportions:

    >>> p = np.random.rand(4)
    >>> p = p / sum(p)
    >>> p = {k + 1: p[k] for k in range(4)}
    >>> g = qt.generate_random_graph(num_vertices=25, is_directed=False, proportions=p)

    Note that none of the edge types in the above example are 0. It is
    recommended use edge type indices starting at 1, since 0 is
    typically used for terminal edges.
    """
    g = minimal_random_graph(num_vertices, **kwargs)
    for v in g.nodes():
        e = (v, v)
        if not g.is_edge(e):
            if np.random.uniform() < prob_loop:
                g.add_edge(*e)
    g = set_types_random(g, **kwargs)
    return g


def generate_pagerank_graph(num_vertices=250, **kwargs):
    """Creates a random graph where the vertex types are
    selected using their pagerank.

    Calls :func:`.minimal_random_graph` and then
    :func:`.set_types_rank` where the ``rank`` keyword argument
    is given by :func:`networkx.pagerank`.

    Parameters
    ----------
    num_vertices : int (optional, the default is 250)
        The number of vertices in the graph.
    **kwargs :
        Any parameters to send to :func:`.minimal_random_graph` or
        :func:`.set_types_rank`.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        A graph with a ``pos`` vertex property and the ``edge_type``
        edge property.

    Notes
    -----
    This function sets the edge types of a graph to be either 1, 2, or
    3. It sets the vertices to type 2 by selecting the top
    ``pType2 * g.number_of_nodes()`` vertices given by the
    :func:`~networkx.pagerank` of the graph. A loop is added
    to all vertices identified this way (if one does not exist
    already). It then randomly sets vertices close to the type 2
    vertices as type 3, and adds loops to these vertices as well. These
    loops then have edge types that correspond to the vertices type.
    The rest of the edges are set to type 1.
    """
    g = minimal_random_graph(num_vertices, **kwargs)
    r = np.zeros(num_vertices)

    # networkx 2.8.6 throws a warning with all pagerank functions except
    # _pagerank_python. We would need to ignore the warning even if we used
    # the recommended networkx.pagerank function.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # In networkx 2.8.6, this function requires scipy, which isn't
            # a requirement of either networkx or queueing-tool. But the
            # other pagerank_* functions are deprecated so we'll only try
            # those if the recommended one fails.
            page_rank = nx.pagerank(g)
        except ImportError as exe:
            try:
                # This function is deprecated and is supposed to be removed
                # in networkx 3.0.
                page_rank = nx.pagerank_numpy(g)
            except:
                raise exe
 
    for k, pr in page_rank.items():
        r[k] = pr
    g = set_types_rank(g, rank=r, **kwargs)
    return g


def minimal_random_graph(num_vertices, seed=None, **kwargs):
    """Creates a connected graph with random vertex locations.

    Parameters
    ----------
    num_vertices : int
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
    This function first places ``num_vertices`` points in the unit square
    randomly (using the uniform distribution). Then, for every vertex
    ``v``, all other vertices with Euclidean distance less or equal to
    ``r`` are connect by an edge --- where ``r`` is the smallest number
    such that the graph ends up connected.
    """
    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    points = np.random.random((num_vertices, 2)) * 10
    edges = []

    for k in range(num_vertices - 1):
        for j in range(k + 1, num_vertices):
            v = points[k] - points[j]
            edges.append((k, j, v[0]**2 + v[1]**2))

    mytype = [('n1', int), ('n2', int), ('distance', float)]
    edges = np.array(edges, dtype=mytype)
    edges = np.sort(edges, order='distance')
    unionF = UnionFind([k for k in range(num_vertices)])

    g = nx.Graph()

    for n1, n2, dummy in edges:
        unionF.union(n1, n2)
        g.add_edge(n1, n2)
        if unionF.nClusters == 1:
            break

    pos = {j: p for j, p in enumerate(points)}
    g = QueueNetworkDiGraph(g.to_directed())
    g.set_pos(pos)
    return g


def set_types_random(g, proportions=None, loop_proportions=None, seed=None,
                     **kwargs):
    """Randomly sets ``edge_type`` (edge type) properties of the graph.

    This function randomly assigns each edge a type. The probability of
    an edge being a specific type is proscribed in the
    ``proportions``, ``loop_proportions`` variables.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :class:`numpy.ndarray`, dict, etc.
        Any object that :any:`DiGraph<networkx.DiGraph>` accepts.
    proportions : dict (optional, default: ``{k: 0.25 for k in range(1, 4)}``)
        A dictionary of edge types and proportions, where the keys are
        the types and the values are the proportion of non-loop edges
        that are expected to be of that type. The values can must sum
        to one.
    loop_proportions : dict (optional, default: ``{k: 0.25 for k in range(4)}``)
        A dictionary of edge types and proportions, where the keys are
        the types and the values are the proportion of loop edges
        that are expected to be of that type. The values can must sum
        to one.
    seed : int (optional)
        An integer used to initialize numpy's psuedorandom number
        generator.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        Returns the a graph with an ``edge_type`` edge property.

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
    defaults to four types in the graph (types 0, 1, 2, and 3). It sets
    non-loop edges to be either 1, 2, or 3 33% chance, and loops are
    types 0, 1, 2, 3 with 25% chance.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    if proportions is None:
        proportions = {k: 1. / 3 for k in range(1, 4)}

    if loop_proportions is None:
        loop_proportions = {k: 1. / 4 for k in range(4)}

    edges = [e for e in g.edges() if e[0] != e[1]]
    loops = [e for e in g.edges() if e[0] == e[1]]
    props = list(proportions.values())
    lprops = list(loop_proportions.values())

    if not np.isclose(sum(props), 1.0):
        raise ValueError("proportions values must sum to one.")
    if not np.isclose(sum(lprops), 1.0):
        raise ValueError("loop_proportions values must sum to one.")

    eTypes = {}
    types = list(proportions.keys())
    values = np.random.choice(types, size=len(edges), replace=True, p=props)

    for k, e in enumerate(edges):
        eTypes[e] = values[k]

    types = list(loop_proportions.keys())
    values = np.random.choice(types, size=len(loops), replace=True, p=lprops)

    for k, e in enumerate(loops):
        eTypes[e] = values[k]

    g.new_edge_property('edge_type')
    for e in g.edges():
        g.set_ep(e, 'edge_type', eTypes[e])

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
    rank : :class:`numpy.ndarray`
        An ordering of the vertices.
    pType2 : float (optional, default: 0.1)
        Specifies the proportion of vertices that will be of type 2.
    pType3 : float (optional, default: 0.1)
        Specifies the proportion of vertices that will be of type 3 and
        that are near pType2 vertices.
    seed : int (optional)
        An integer used to initialize numpy's psuedo-random number
        generator.
    **kwargs :
        Unused.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        Returns the a graph with an ``edge_type`` edge property.

    Raises
    ------
    TypeError
        Raised when the parameter ``g`` is not of a type that can be
        made into a :any:`DiGraph<networkx.DiGraph>`.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral):
        np.random.seed(seed)

    tmp = np.sort(np.array(rank))
    nDests = int(np.ceil(g.number_of_nodes() * pType2))
    dests = np.where(rank >= tmp[-nDests])[0]

    if 'pos' not in g.vertex_properties():
        g.set_pos()

    dest_pos = np.array([g.vp(v, 'pos') for v in dests])
    nFCQ = int(pType3 * g.number_of_nodes())
    min_g_dist = np.ones(nFCQ) * np.infty
    ind_g_dist = np.ones(nFCQ, int)

    r, theta = np.random.random(nFCQ) / 500., np.random.random(nFCQ) * 360.
    xy_pos = np.array([r * np.cos(theta), r * np.sin(theta)]).transpose()
    g_pos = xy_pos + dest_pos[np.array(np.mod(np.arange(nFCQ), nDests), int)]

    for v in g.nodes():
        if v not in dests:
            tmp = np.array([_calculate_distance(g.vp(v, 'pos'), g_pos[k, :]) for k in range(nFCQ)])
            min_g_dist = np.min((tmp, min_g_dist), 0)
            ind_g_dist[min_g_dist == tmp] = v

    ind_g_dist = np.unique(ind_g_dist)
    fcqs = set(ind_g_dist[:min(nFCQ, len(ind_g_dist))])
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

    g.new_edge_property('edge_type')
    for e in g.edges():
        g.set_ep(e, 'edge_type', 1)

    for v in g.nodes():
        if g.vp(v, 'loop_type') in [2, 3]:
            e = (v, v)
            if g.vp(v, 'loop_type') == 2:
                g.set_ep(e, 'edge_type', 2)
            else:
                g.set_ep(e, 'edge_type', 3)

    return g
