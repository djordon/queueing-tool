import networkx as nx
import numpy as np
import numbers

from .. graph import GraphWrapper
from .graph_generation import adjacency2graph
from .graph_functions  import graph2dict


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
    if not isinstance(g, GraphWrapper):
        if not isinstance(g, nx.DiGraph):
            try:
                import graph_tool.all as gt
            except ImportError:
                msg = ("Graph given was not a networkx DiGraph or graph_tool "
                       "graph.")
                raise ImportError(msg)
            if not isinstance(g, gt.Graph) :
                msg = "Need to supply a graph-tool Graph or networkx DiGraph"
                raise TypeError(msg)

        g = GraphWrapper(g)
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
    g.new_edge_property('elength', 'double')

    for e in g.edges() :
        latlon1 = g.vp(e[1], 'pos')
        latlon2 = g.vp(e[0], 'pos')
        g.set_ep(e, 'elength', np.round(_calculate_distance(latlon1, latlon2), 3))
    
    return g



def _prepare_graph(g, g_colors, q_cls, q_arg) :
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
        The same graph, but with the addition of various
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
        * ``edge_control_points``: ``[]``
        * ``edge_marker_size``: ``8``
        * ``edge_pen_width``: ``1.25``
        
    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`.
    """
    g = _test_graph(g)

    g.new_vertex_property('vertex_color', 'vector<double>')
    g.new_vertex_property('vertex_fill_color', 'vector<double>')
    g.new_vertex_property('vertex_pen_width', 'double')
    g.new_vertex_property('vertex_size', 'double')

    g.new_edge_property('edge_control_points', 'vector<double>')
    g.new_edge_property('edge_color', 'vector<double>')
    g.new_edge_property('edge_marker_size', 'double')
    g.new_edge_property('edge_pen_width', 'double')

    if 'eType' not in g.edge_properties :
        ans = g.graph2dict()
        g   = adjacency2graph(ans, adjust=1, is_directed=g.is_directed())

    queues = _set_queues(g, q_cls, q_arg, 'cap' in g.vertex_properties)

    if 'pos' not in g.vertex_properties:
        g.set_pos()

    for k, e in enumerate(g.edges()):
        g.set_ep(e, 'edge_pen_width', 1.25)
        g.set_ep(e, 'edge_marker_size', 8)
        if e[0] == e[1]:
            g.set_ep(e, 'edge_color', queues[k].colors['edge_loop_color'])
        else :
            g.set_ep(e, 'edge_color', queues[k].colors['edge_color'])

    for v in g.vertices() :
        g.set_vp(v, 'vertex_pen_width', 1.1)
        g.set_vp(v, 'vertex_size', 8)
        e = (v, v)
        if g.is_edge(e):
            g.set_vp(v, 'vertex_color', queues[g.edge_index[e]]._current_color(2))
            g.set_vp(v, 'vertex_fill_color', queues[g.edge_index[e]]._current_color())
        else:
            g.set_vp(v, 'vertex_color', g_colors['vertex_color'])
            g.set_vp(v, 'vertex_fill_color', g_colors['vertex_fill_color'])

    return g, queues


def _set_queues(g, q_cls, q_arg, has_cap):
    queues = [0 for k in range(g.num_edges())]

    for e in g.edges():
        eType = g.ep(e, 'eType')
        qedge = (e[0], e[1], g.edge_index[e], eType)

        if has_cap and 'nServers' not in q_arg[eType]:
            cap = g.vp(e[1], 'cap') if g.vp(e[1], 'cap') is not None else 0
            q_arg[eType]['nServers'] = max(cap, 1)

        queues[qedge[2]] = q_cls[eType](edge=qedge, **q_arg[eType])

    return queues
