import graph_tool.all as gt
import numpy   as np
import numbers

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


def osm_edge_types(g) :
    """A function that takes graphs created using open street maps and formsts
    them for use with the :class:`.QueueNetwork` class.

    Made specifically for a :class:`~graph_tool.Graph` created using data from
    `openstreetmaps <www.openstreetmaps.org>`_. Graphs from openstreetmaps
    sometimes have tags for certain nodes (like the latitude and longitude),
    or whether a location is an attraction. This function uses some of that
    information to set a :class:`~graph_tool.Graph`'s ``eType`` and
    ``edge_length`` edge properties.

    Parameters
    ----------
    g : A **str** or a :class:`~graph_tool.Graph`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        Returns the :class:`~graph_tool.Graph` ``g`` with the ``eType`` and
        ``edge_length`` edge properties.

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`\.
    """
    g = _test_graph(g)

    g.reindex_edges()
    vertex_props = set()
    for key in g.vertex_properties.keys() :
        vertex_props.add(key)

    edge_props = set()
    for key in g.edge_properties.keys() :
        edge_props.add(key)

    has_garage  = 'garage' in vertex_props
    has_destin  = 'destination' in vertex_props
    has_light   = 'light' in vertex_props
    has_egarage = 'garage' in edge_props
    has_edestin = 'destination' in edge_props
    has_elight  = 'light' in edge_props

    eType   = g.new_edge_property("int")
    for v in g.vertices() :
        if has_garage and g.vp['garage'][v] :
            e = g.edge(v,v)
            if isinstance(e, gt.Edge) :
                eType[e]  = 1
        if has_destin and g.vp['destination'][v] :
            e = g.edge(v,v)
            if isinstance(e, gt.Edge) :
                eType[e]  = 2
        if has_light and g.vp['light'][v] :
            e = g.edge(v,v)
            if isinstance(e, gt.Edge) :
                eType[e]  = 3

    for e in g.edges() :
        if has_egarage and g.ep['garage'][e] :
            eType[e]  = 1
        if has_edestin and g.ep['destination'][e] :
            eType[e]  = 2
        if has_elight and g.ep['light'][e] :
            eType[e]  = 3

    g.ep['eType'].a = eType.a + 1
    return add_edge_lengths(g)


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


def add_edge_lengths(g) :
    """Add add the edge lengths as a :class:`~graph_tool.PropertyMap` for the graph.

    Uses the ``pos`` vertex property to get the location of each vertex. These
    are then used to calculate the length of an edge between two vertices.

    Parameters
    ----------
    g : A string or a :class:`~graph_tool.Graph`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        Returns the :class:`~graph_tool.Graph` ``g`` with the ``edge_length``
        edge property.

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`\.

    """
    g = _test_graph(g)
    elength   = g.new_edge_property("double")

    for e in g.edges() :
        latlon1     = g.vp['pos'][e.target()]
        latlon2     = g.vp['pos'][e.source()]
        elength[e]  = np.round(_calculate_distance(latlon1, latlon2), 3)
    
    g.ep['edge_length'] = elength
    return g


def set_types_random(g, pTypes=None, seed=None, **kwargs) :
    """Randomly sets ``eType`` (edge type) properties of the graph.

    This function randomly assigns each edge a type. The probability of an edge being 
    a specific type is proscribed in the ``pTypes`` variable. The vertex type is set
    to the same type as an edge that is a loop, and 1 otherwise.

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
        An integer used to initialize ``numpy``\'s psuedorandom number
        generator.
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
        and do not sum to the number of edges in the graph.
    
    Notes
    -----
    If ``pTypes`` is not explicitly specified in the arguments, then it defaults to three
    types in the graph (types 1, 2, and 3) and sets their proportions to be 1/3 each.
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)

    if pTypes is None :
        pTypes = {k : 1/3 for k in range(1,4)}

    nEdges  = g.num_edges()
    edges   = [k for k in range(nEdges)]
    cut_off = np.cumsum( list(pTypes.values()) )

    if np.isclose(cut_off[-1], 1) :
        cut_off = np.round(cut_off * nEdges, out=np.zeros(len(pTypes), int))
    elif cut_off != nEdges :
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

    for e in g.edges() :
        eType[e] = eTypes[g.edge_index[e]]
    
    g.ep['eType'] = eType
    return g


def set_types_pagerank(g, pType2=0.1, pType3=0.1, seed=None, **kwargs) :
    """Creates a stylized graph. Sets edge and types using `pagerank`_.

    This function sets the edge types of a graph to be either 1, 2, or 3.
    It sets the vertices to type 2 by selecting the top
    ``pType2 * g.num_vertices()`` vertices given by the
    :func:`~graph_tool.centrality.pagerank` of the graph. These a loop is added
    to all vertices identified this way (if one does not exist already). It
    then randomly sets vertices close to type 2 vertices as type 3, and adds
    loops to these vertices as well. These loops then have edge types the
    correspond to the vertices type. The rest of the edges are set to type 1.

    Parameters
    ----------
    g : A string or a :class:`~graph_tool.Graph`.
    pType2 : float (optional, the default is 0.1)
        Specifies the proportion of vertices that will be of type 2.
    pType3 : float (optional, the default is 0.1)
        Specifies the proportion of vertices that will be of type 3 and that
        are near pType2 vertices.
    seed : int (optional)
        An integer used to initialize ``numpy``\'s and ``graph-tool``\'s
        psuedorandom number generators.
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

    .. _pagerank: http://en.wikipedia.org/wiki/PageRank
    """
    g = _test_graph(g)

    if isinstance(seed, numbers.Integral) :
        np.random.seed(seed)
        gt.seed_rng(seed)

    pagerank    = gt.pagerank(g)
    tmp         = np.sort(np.array(pagerank.a))
    nDests      = int(np.ceil(g.num_vertices() * pType2))
    dests       = np.where(pagerank.a >= tmp[-nDests])[0]
    
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


def prepare_graph(g, g_colors, q_cls, q_arg) :
    """Prepares a graph for use in :class:`.QueueNetwork`.

    This function is called by ``__init__`` in the :class:`.QueueNetwork` class.
    It creates the :class:`.QueueServer` instances that sit on the edges, and
    sets various :class:`~graph_tool.PropertyMap`\s that are used when drawing
    the graph.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
    g_colors : :class:`.dict`
        A dictionary of colors. The specific keys used are ``vertex_color`` and
        ``vertex_fill_color`` for vertices that do not have any loops. Set
        :class:`.QueueNetwork` for the default values passed.
    q_cls : :class:`.dict`
        A dictionary where the keys are integers that represent an edge type,
        and the values are :class:`.QueueServer` classes.
    q_args : :class:`.dict`
        A dictionary where the keys are integers that represent an edge type,
        and the values are the arguments that are used when creating an
        instance of that :class:`.QueueServer` class.

    Returns
    -------
    g : :class:`~graph_tool.Graph`
        The same graph, but with the addiction of various
        :class:`~graph_tool.PropertyMap`\s.
    queues : :class:`.list`
        A list of :class:`.QueueServer`\s where ``queues[k]`` is the
        ``QueueServer`` that sets on the edge with edge index ``k``.
    
    Notes
    -----
    The graph ``g`` should have the ``eType`` edge property map. If it does not
    then an ``eType`` edge property is created and set to 1.

    The following properties are set by each queue: ``vertex_color``,
    ``vertex_fill_color``, ``vertex_fill_color``, ``edge_color``.
    See :class:`.QueueServer` for more on setting these values.

    The following properties are assigned as a :class:`~graph_tool.PropertyMap`
    to the graph; their default values for each edge or vertex is shown:
        
        * ``vertex_pen_width``: ``1.1``,
        * ``vertex_size``: ``8``,
        * ``edge_control_points``: ``[0, 0, 0, 0]``
        * ``edge_marker_size``: ``8``
        * ``edge_pen_width``: ``1.25``
        
    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`.
    """
    g = _test_graph(g)

    g.reindex_edges()
    vertex_color        = g.new_vertex_property("vector<double>")
    vertex_fill_color   = g.new_vertex_property("vector<double>")
    vertex_pen_width    = g.new_vertex_property("double")
    vertex_size         = g.new_vertex_property("double")

    edge_control_points = g.new_edge_property("vector<double>")
    edge_color          = g.new_edge_property("vector<double>")
    edge_marker_size    = g.new_edge_property("double")
    edge_pen_width      = g.new_edge_property("double")

    vertex_props = set()
    for key in g.vertex_properties.keys() :
        vertex_props.add(key)

    edge_props = set()
    for key in g.edge_properties.keys() :
        edge_props.add(key)

    if 'eType' not in edge_props :
        eType   = g.new_edge_property("int")
        eType.a = 1
        g.ep['eType'] = eType

    props   = vertex_props.union(edge_props)
    queues  = _set_queues(g, q_cls, q_arg, 'cap' in vertex_props)

    if 'pos' not in vertex_props :
        g.vp['pos'] = gt.sfdp_layout(g, epsilon=1e-2, cooling_step=0.95)

    for k, e in enumerate(g.edges()) :
        if e.target() == e.source() :
            edge_color[e] = queues[k].colors['edge_loop_color']
        else :
            edge_color[e] = queues[k].colors['edge_color']

    for v in g.vertices() :
        e = g.edge(v, v)
        if isinstance(e, gt.Edge) :
            vertex_color[v]       = queues[g.edge_index[e]]._current_color(2)
            vertex_fill_color[v]  = queues[g.edge_index[e]]._current_color()
        else :
            vertex_color[v]       = g_colors['vertex_color']
            vertex_fill_color[v]  = g_colors['vertex_fill_color'] 

    edge_pen_width.a      = 1.25
    edge_marker_size.a    = 8
    vertex_pen_width.a    = 1.1
    vertex_size.a         = 8

    properties = {
        'vertex_fill_color' : vertex_fill_color,
        'vertex_pen_width' : vertex_pen_width,
        'vertex_color' : vertex_color,
        'vertex_size' : vertex_size,
        'edge_control_points' : edge_control_points,
        'edge_marker_size' : edge_marker_size,
        'edge_pen_width' : edge_pen_width,
        'edge_color' : edge_color}

    for key, value in properties.items() :
        if key not in props :
            if key[:4] == 'edge' :
                g.ep[key] = value
            else :
                g.vp[key] = value

    return g, queues


def _set_queues(g, q_cls, q_arg, has_cap) :
    queues = [0 for k in range(g.num_edges())]

    for e in g.edges() :
        eType = g.ep['eType'][e]
        qedge = (int(e.source()), int(e.target()), g.edge_index[e], eType)

        if has_cap and 'nServers' not in q_arg[eType] :
            q_arg[eType]['nServers'] = max(g.vp['cap'][e.target()] // 2, 1)

        queues[qedge[2]] = q_cls[eType](edge=qedge, **q_arg[eType])

    return queues
