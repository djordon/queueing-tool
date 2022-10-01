import networkx as nx
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.style.use('ggplot')
    HAS_MATPLOTLIB = True

except ImportError:
    HAS_MATPLOTLIB = False


def _matrix2dict(matrix, etype=False):
    """Takes an adjacency matrix and returns an adjacency list."""
    n = len(matrix)
    adj = {k: {} for k in range(n)}
    for k in range(n):
        for j in range(n):
            if matrix[k, j] != 0:
                adj[k][j] = {} if not etype else matrix[k, j]

    return adj


def _dict2dict(adj_dict):
    """Takes a dictionary based representation of an adjacency list
    and returns a dict of dicts based representation.
    """
    item = adj_dict.popitem()
    adj_dict[item[0]] = item[1]
    if not isinstance(item[1], dict):
        new_dict = {}
        for key, value in adj_dict.items():
            new_dict[key] = {v: {} for v in value}

        adj_dict = new_dict
    return adj_dict


def _adjacency_adjust(adjacency, adjust, is_directed):
    """Takes an adjacency list and returns a (possibly) modified
    adjacency list.
    """

    for v, adj in adjacency.items():
        for properties in adj.values():
            if properties.get('edge_type') is None:
                properties['edge_type'] = 1

    if is_directed:
        if adjust == 2:
            null_nodes = set()

            for k, adj in adjacency.items():
                if len(adj) == 0:
                    null_nodes.add(k)

            for k, adj in adjacency.items():
                for v in adj.keys():
                    if v in null_nodes:
                        adj[v]['edge_type'] = 0

        else:
            for k, adj in adjacency.items():
                if len(adj) == 0:
                    adj[k] = {'edge_type': 0}

    return adjacency


def adjacency2graph(adjacency, edge_type=None, adjust=1, **kwargs):
    """Takes an adjacency list, dict, or matrix and returns a graph.

    The purpose of this function is take an adjacency list (or matrix)
    and return a :class:`.QueueNetworkDiGraph` that can be used with a
    :class:`.QueueNetwork` instance. The Graph returned has the
    ``edge_type`` edge property set for each edge. Note that the graph may
    be altered.

    Parameters
    ----------
    adjacency : dict or :class:`~numpy.ndarray`
        An adjacency list as either a dict, or an adjacency matrix.
    adjust : int ``{1, 2}`` (optional, default: 1)
        Specifies what to do when the graph has terminal vertices
        (nodes with no out-edges). Note that if ``adjust`` is not 2
        then it is assumed to be 1. There are two choices:

        * ``adjust = 1``: A loop is added to each terminal node in the
          graph, and their ``edge_type`` of that loop is set to 0.
        * ``adjust = 2``: All edges leading to terminal nodes have
          their ``edge_type`` set to 0.

    **kwargs :
        Unused.

    Returns
    -------
    out : :any:`networkx.DiGraph`
        A directed graph with the ``edge_type`` edge property.

    Raises
    ------
    TypeError
        Is raised if ``adjacency`` is not a dict or
        :class:`~numpy.ndarray`.

    Examples
    --------
    If terminal nodes are such that all in-edges have edge type ``0``
    then nothing is changed. However, if a node is a terminal node then
    a loop is added with edge type 0.

    >>> import queueing_tool as qt
    >>> adj = {
    ...     0: {1: {}},
    ...     1: {2: {},
    ...         3: {}},
    ...     3: {0: {}}}
    >>> eTy = {0: {1: 1}, 1: {2: 2, 3: 4}, 3: {0: 1}}
    >>> # A loop will be added to vertex 2
    >>> g = qt.adjacency2graph(adj, edge_type=eTy)
    >>> ans = qt.graph2dict(g)
    >>> sorted(ans.items())     # doctest: +NORMALIZE_WHITESPACE
    [(0, {1: {'edge_type': 1}}),
     (1, {2: {'edge_type': 2}, 3: {'edge_type': 4}}), 
     (2, {2: {'edge_type': 0}}),
     (3, {0: {'edge_type': 1}})]

    You can use a dict of lists to represent the adjacency list.

    >>> adj = {0 : [1], 1: [2, 3], 3: [0]}
    >>> g = qt.adjacency2graph(adj, edge_type=eTy)
    >>> ans = qt.graph2dict(g)
    >>> sorted(ans.items())     # doctest: +NORMALIZE_WHITESPACE
    [(0, {1: {'edge_type': 1}}),
     (1, {2: {'edge_type': 2}, 3: {'edge_type': 4}}),
     (2, {2: {'edge_type': 0}}),
     (3, {0: {'edge_type': 1}})]

    Alternatively, you could have this function adjust the edges that
    lead to terminal vertices by changing their edge type to 0:

    >>> # The graph is unaltered
    >>> g = qt.adjacency2graph(adj, edge_type=eTy, adjust=2)
    >>> ans = qt.graph2dict(g)
    >>> sorted(ans.items())     # doctest: +NORMALIZE_WHITESPACE
    [(0, {1: {'edge_type': 1}}),
     (1, {2: {'edge_type': 0}, 3: {'edge_type': 4}}),
     (2, {}),
     (3, {0: {'edge_type': 1}})]
    """

    if isinstance(adjacency, np.ndarray):
        adjacency = _matrix2dict(adjacency)
    elif isinstance(adjacency, dict):
        adjacency = _dict2dict(adjacency)
    else:
        msg = ("If the adjacency parameter is supplied it must be a "
               "dict, or a numpy.ndarray.")
        raise TypeError(msg)

    if edge_type is None:
        edge_type = {}
    else:
        if isinstance(edge_type, np.ndarray):
            edge_type = _matrix2dict(edge_type, etype=True)
        elif isinstance(edge_type, dict):
            edge_type = _dict2dict(edge_type)

    for u, ty in edge_type.items():
        for v, et in ty.items():
            adjacency[u][v]['edge_type'] = et

    g = nx.from_dict_of_dicts(adjacency, create_using=nx.DiGraph())
    adjacency = nx.to_dict_of_dicts(g)
    adjacency = _adjacency_adjust(adjacency, adjust, True)

    return nx.from_dict_of_dicts(adjacency, create_using=nx.DiGraph())


SAVEFIG_KWARGS = set([
    'dpi',
    'facecolorw',
    'edgecolorw',
    'orientationportrait',
    'papertype',
    'format',
    'transparent',
    'bbox_inches',
    'pad_inches',
    'frameon'
])


class QueueNetworkDiGraph(nx.DiGraph):
    """A directed graph class built to work with a\
    :class:`.QueueNetwork`

    If data is a dict then :func:`.adjacency2graph` is called first.

    Parameters
    ----------
    data : :any:`networkx.DiGraph`, :class:`numpy.ndarray`, dict, etc.
        Any object that networkx can turn into a
        :any:`DiGraph<networkx.DiGraph>`.
    kwargs :
        Any additional arguments for :any:`networkx.DiGraph`.

    Attributes
    ----------
    pos : :class:`~numpy.ndarray` or ``None``
        An ``(V, 2)`` array for the position for each vertex
        (``V`` is the number of vertices). By default this is ``None``.
    edge_color : :class:`~numpy.ndarray` or ``None``
        An ``(E, 4)`` array for the RGBA colors for each edge.
        (``E`` is the number of edges). By default this is ``None``.
    vertex_color : :class:`~numpy.ndarray` or ``None``
        An ``(V, 4)`` array for the RGBA colors for each vertex
        border. By default this is ``None``.
    vertex_fill_color : :class:`~numpy.ndarray` or ``None``
        An ``(V, 4)`` array for the RGBA colors for the body of each
        vertex. By default this is ``None``.

    Notes
    -----
    Not suitable for stand alone use; only use with a
    :class:`.QueueNetwork`.
    """
    def __init__(self, data=None, **kwargs):
        if isinstance(data, dict):
            data = adjacency2graph(data, **kwargs)

        super(QueueNetworkDiGraph, self).__init__(data, **kwargs)
        edges = sorted(self.edges())

        self.edge_index = {e: k for k, e in enumerate(edges)}

        pos = nx.get_node_attributes(self, name='pos')
        if len(pos) == self.number_of_nodes():
            self.pos = np.array([pos[v] for v in self.nodes()])
        else:
            self.pos = None

        self.edge_color = None
        self.vertex_color = None
        self.vertex_fill_color = None
        self._nE = self.number_of_edges()

    def freeze(self):
        nx.freeze(self)

    def is_edge(self, e):
        return e in self.edge_index

    def add_edge(self, *args, **kwargs):
        super(QueueNetworkDiGraph, self).add_edge(*args, **kwargs)
        e = (args[0], args[1])
        if e not in self.edge_index:
            self.edge_index[e] = self._nE
            self._nE += 1

    def out_neighbours(self, v):
        return [e[1] for e in self.out_edges(v)]

    def ep(self, e, edge_property):
        return self.adj[e[0]][e[1]].get(edge_property)

    def vp(self, v, vertex_property):
        return self.nodes[v].get(vertex_property)

    def set_ep(self, e, edge_property, value):
        self.adj[e[0]][e[1]][edge_property] = value
        if hasattr(self, edge_property):
            attr = getattr(self, edge_property)
            attr[self.edge_index[e]] = value

    def set_vp(self, v, vertex_property, value):
        self.nodes[v][vertex_property] = value
        if hasattr(self, vertex_property):
            attr = getattr(self, vertex_property)
            attr[v] = value

    def vertex_properties(self):
        props = set()
        for v in self.nodes():
            props.update(self.nodes[v].keys())
        return props

    def edge_properties(self):
        props = set()
        for e in self.edges():
            props.update(self.adj[e[0]][e[1]].keys())
        return props

    def new_vertex_property(self, name):
        values = {v: None for v in self.nodes()}
        nx.set_node_attributes(self, name=name, values=values)
        if name == 'vertex_color':
            self.vertex_color = [0 for v in range(self.number_of_nodes())]
        if name == 'vertex_fill_color':
            self.vertex_fill_color = [0 for v in range(self.number_of_nodes())]

    def new_edge_property(self, name):
        values = {e: None for e in self.edges()}
        nx.set_edge_attributes(self, name=name, values=values)
        if name == 'edge_color':
            self.edge_color = np.zeros((self.number_of_edges(), 4))

    def set_pos(self, pos=None):
        if pos is None:
            pos = nx.spring_layout(self)
        nx.set_node_attributes(self, name='pos', values=pos)
        self.pos = np.array([pos[v] for v in self.nodes()])

    def get_edge_type(self, edge_type):
        """Returns all edges with the specified edge type.

        Parameters
        ----------
        edge_type : int
            An integer specifying what type of edges to return.

        Returns
        -------
        out : list of 2-tuples
            A list of 2-tuples representing the edges in the graph
            with the specified edge type.

        Examples
        --------
        Lets get type 2 edges from the following graph

        >>> import queueing_tool as qt
        >>> adjacency = {
        ...     0: {1: {'edge_type': 2}},
        ...     1: {2: {'edge_type': 1},
        ...         3: {'edge_type': 4}},
        ...     2: {0: {'edge_type': 2}},
        ...     3: {3: {'edge_type': 0}}
        ... }
        >>> G = qt.QueueNetworkDiGraph(adjacency)
        >>> ans = G.get_edge_type(2)
        >>> ans.sort()
        >>> ans
        [(0, 1), (2, 0)]
        """
        edges = []
        for e in self.edges():
            if self.adj[e[0]][e[1]].get('edge_type') == edge_type:
                edges.append(e)
        return edges

    def draw_graph(self, line_kwargs=None, scatter_kwargs=None, **kwargs):
        """Draws the graph.

        Uses matplotlib, specifically
        :class:`~matplotlib.collections.LineCollection` and
        :meth:`~matplotlib.axes.Axes.scatter`. Gets the default
        keyword arguments for both methods by calling
        :meth:`~.QueueNetworkDiGraph.lines_scatter_args` first.

        Parameters
        ----------
        line_kwargs : dict (optional, default: ``None``)
            Any keyword arguments accepted by
            :class:`~matplotlib.collections.LineCollection`
        scatter_kwargs : dict (optional, default: ``None``)
            Any keyword arguments accepted by
            :meth:`~matplotlib.axes.Axes.scatter`.
        bgcolor : list (optional, keyword only)
            A list with 4 floats representing a RGBA color. Defaults
            to ``[1, 1, 1, 1]``.
        figsize : tuple (optional, keyword only, default: ``(7, 7)``)
            The width and height of the figure in inches.
        kwargs :
            Any keyword arguments used by
            :meth:`~matplotlib.figure.Figure.savefig`.

        Raises
        ------
        ImportError :
            If Matplotlib is not installed then an :exc:`ImportError`
            is raised.

        Notes
        -----
        If the ``fname`` keyword is passed, then the figure is saved
        locally.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required to draw the graph.")

        fig = plt.figure(figsize=kwargs.get('figsize', (7, 7)))
        ax = fig.gca()

        mpl_kwargs = {
            'line_kwargs': line_kwargs,
            'scatter_kwargs': scatter_kwargs,
            'pos': kwargs.get('pos')
        }

        line_kwargs, scatter_kwargs = self.lines_scatter_args(**mpl_kwargs)

        edge_collection = LineCollection(**line_kwargs)
        ax.add_collection(edge_collection)
        ax.scatter(**scatter_kwargs)

        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(kwargs.get('bgcolor', [1, 1, 1, 1]))
        else:
            ax.set_axis_bgcolor(kwargs.get('bgcolor', [1, 1, 1, 1]))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if 'fname' in kwargs:
            # savefig needs a positional argument for some reason
            new_kwargs = {k: v for k, v in kwargs.items() if k in SAVEFIG_KWARGS}
            fig.savefig(kwargs['fname'], **new_kwargs)
        else:
            plt.ion()
            plt.show()

    def lines_scatter_args(self, line_kwargs=None, scatter_kwargs=None, pos=None):
        """Returns the arguments used when plotting.

        Takes any keyword arguments for
        :class:`~matplotlib.collections.LineCollection` and
        :meth:`~matplotlib.axes.Axes.scatter` and returns two
        dictionaries with all the defaults set.

        Parameters
        ----------
        line_kwargs : dict (optional, default: ``None``)
            Any keyword arguments accepted by
            :class:`~matplotlib.collections.LineCollection`.
        scatter_kwargs : dict (optional, default: ``None``)
            Any keyword arguments accepted by
            :meth:`~matplotlib.axes.Axes.scatter`.

        Returns
        -------
        tuple
            A 2-tuple of dicts. The first entry is the keyword
            arguments for
            :class:`~matplotlib.collections.LineCollection` and the
            second is the keyword args for
            :meth:`~matplotlib.axes.Axes.scatter`.

        Notes
        -----
        If a specific keyword argument is not passed then the defaults
        are used.
        """
        if pos is not None:
            self.set_pos(pos)
        elif self.pos is None:
            self.set_pos()

        edge_pos = [0 for e in self.edges()]
        for e in self.edges():
            ei = self.edge_index[e]
            edge_pos[ei] = (self.pos[e[0]], self.pos[e[1]])

        line_collecton_kwargs = {
            'segments': edge_pos,
            'colors': self.edge_color,
            'linewidths': (1,),
            'antialiaseds': (1,),
            'linestyle': 'solid',
            'transOffset': None,
            'cmap': plt.cm.ocean_r,
            'pickradius': 5,
            'zorder': 0,
            'facecolors': None,
            'norm': None,
            'offsets': None,
            'hatch': None,
        }
        scatter_kwargs_ = {
            'x': self.pos[:, 0],
            'y': self.pos[:, 1],
            's': 50,
            'c': self.vertex_fill_color,
            'alpha': None,
            'norm': None,
            'vmin': None,
            'vmax': None,
            'marker': 'o',
            'zorder': 2,
            'linewidths': 1,
            'edgecolors': self.vertex_color,
            'facecolors': None,
            'antialiaseds': None,
            'hatch': None,
        }

        line_kwargs = {} if line_kwargs is None else line_kwargs
        scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs

        for key, value in line_kwargs.items():
            if key in line_collecton_kwargs:
                line_collecton_kwargs[key] = value

        for key, value in scatter_kwargs.items():
            if key in scatter_kwargs_:
                scatter_kwargs_[key] = value

        return line_collecton_kwargs, scatter_kwargs_
