import collections
import numbers
import copy

import networkx as nx
import numpy as np
from numpy.random import uniform

from queueing_tool.graph import _prepare_graph
from queueing_tool.queues import (
    NullQueue,
    QueueServer,
    LossQueue
)
from queueing_tool.network.sorting import (
    oneBisectSort,
    bisectSort,
    oneSort,
    twoSort
)

class InitializationError(Exception):
    pass

EPS = np.float64(1e-7)

class QueueNetwork(object):
    """A class that simulates a network of queues.

    Takes a graph-tool :class:`~graph_tool.Graph` and places queues on each
    edge of the graph. The simulations are event based, and this class handles
    the scheduling of events.

    Each edge on the graph has a *type*, and this *type* is used to define the
    type of :class:`.QueueServer` that sits on that edge.

    Parameters
    ----------
    g : str or :class:`~graph_tool.Graph`
        The graph specifies the network on which the queues sit.
    q_classes : :class:`.dict` (optional)
        Used to specify the :class:`.QueueServer` class for each edge type.
        The keys are integers for the edge types, and the values are classes.
    q_args : :class:`.dict` (optional)
        Used to specify the class arguments for each type of
        :class:`.QueueServer`\. The keys are integers for the edge types and
        the values are the arguments that are passed when instantiating each
        ``QueueServer`` created with that edge type.
    seed : int (optional)
        An integer used to initialize numpy's and graph-tool's psuedorandom
        number generators.
    colors : :class:`.dict` (optional)
        A dictionary of RGBA colors used to color the graph. The keys are
        specified in the Notes section. If this parameter is supplied and a
        particular key is missing, then the default value for that key is used.
    max_agents : int (optional, the default is 1000)
        The maximum number of agents that can be in the network at any time.
    blocking : str ``{'BAS', 'RS'}`` (optional, the default is ``'BAS'``)
        Specifies the blocking behavior for the system. If ``blocking`` is not
        ``'RS'``, then it is assumed to be ``'BAS'``.

        ``'BAS'``
            Blocking After Service: when an agent attempts to enter a
            :class:`.LossQueue` that is at capacity the agent is forced to
            wait at his current queue until an agent departs from the queue.
        ``'RS'``
            Repetitive Service Blocking: when an agent attempts to enter a
            :class:`.LossQueue` that is at capacity, the agent is forced to
            receive another service from the queue it is departing from.
            After the agent receives the service, she then checks to see if
            the desired queue is still at capacity, and if it is this process
            is repeated, otherwise she enters the queue.

    Attributes
    ----------
    g : :class:`~graph_tool.Graph`
        The graph for the network.
    blocking : str
        Specifies whether the system's blocking behavior is either Blocking
        After Service (BAS) or Repetitive Service Blocking (RS).
    in_edges : :class:`.list`
        A mapping between vertex indices and the in-edges at that vertex.
        Specifically, ``in_edges[v]`` returns a list containing the edge index
        for all edges with the head of the edge at ``v``, where ``v`` is the
        the vertex's index number.
    out_edges : :class:`.list`
        A mapping between vertex indices and the out-edges at that vertex.
        Specifically, ``out_edges[v]`` returns a list containing the edge index
        for all edges with the tail of the edge at ``v``, where ``v`` is the
        the vertex's index number.
    edge2queue : :class:`.list`
        A list of queues where the ``edge2queue[k]`` returns the queue on the
        edge with edge index ``k``.
    nAgents : :class:`~numpy.ndarray`
        A one-dimensional array where the ``k``'th entry corresponds to the
        total number of agents in the :class:`.QueueServer` with edge index
        ``k``. This include agents that are scheduled to arrive at the queue
        at some future time but haven't yet.
    nEdges : int
        The number of edges in the graph.
    nEvents : int
        The number of events that have occurred thus far. Every arrival from
        outside the network counts as one event, but the departure of an agent
        from a queue and the arrival of that same agent to another queue counts
        as one event.
    nVertices : int
        The number of vertices in the graph.
    nNodes : int
        The number of vertices in the graph.
    current_time : float
        The time of the last event.
    time : float
        The time of the next event.
    max_agents : int
        The maximum number of agents that can be in the network at any time.
    colors : :class:`.dict`
        A dictionary of colors used when drawing a graph. See the notes for the
        defaults.

    Raises
    ------
    TypeError
        The parameter ``g`` must be either a :class:`~graph_tool.Graph`, a
        string of a file location to a graph, or ``None``.

    Notes
    -----
    * If only :class:`.Agent`\s enter the network, then the ``QueueNetwork``
      instance is a `Jackson network`_. The default transition probabilities
      at any vertex ``v`` is ``1 / v.out_degree()`` for each adjacent vertex.
    * This class must be initialized before any simulations can take place. To
      initialize, call the :meth:`~initialize` method.
    * When simulating the network, the departure of an agent from one queue
      coincides with their arrival to another. There is no time lag between
      these events.
    * When defining your ``q_classes`` you should not assign queues with edge
      type ``0`` to anything other than the :class:`.NullQueue` class.  Edges
      with edge type ``0`` are treated by ``QueueNetwork`` as terminal edges
      (edges that point to a terminal vertex).
    * If an edge type is used in your network but not given in ``q_classes``
      parameter then the defaults are used, where the defaults are:

      >>> default_classes = { # doctest: +SKIP
      ...     0: qt.NullQueue,
      ...     1: qt.QueueServer,
      ...     2: qt.LossQueue,
      ...     3: qt.LossQueue,
      ...     4: qt.LossQueue
      ... }

      For example, if your network has type ``0``\, ``1``\, and ``2`` edges but
      your ``q_classes`` parameter looks like:

      >>> my_classes = {1 : qt.ResourceQueue} # doctest: +SKIP

      then each type ``0`` or type ``2`` edge is a :class:`.NullQueue` or
      :class:`.LossQueue` respectively.
    * The following properties are assigned as a :class:`~graph_tool.PropertyMap`
      to the graph; their default values for each edge or vertex is shown:

        * ``vertex_pen_width``: ``1.1``,
        * ``vertex_size``: ``8``,
        * ``edge_control_points``: ``[]``
        * ``edge_marker_size``: ``8``
        * ``edge_pen_width``: ``1.25``

      There are also property maps created for graph visualization, they are
      ``vertex_color``\, ``vertex_fill_color``\, ``pos``\, and ``edge_color``\.
      The default colors, which are used by various methods, are:

      >>> default_colors = {'vertex_fill_color': [0.9, 0.9, 0.9, 1.0],
      ...                   'vertex_color'     : [0.0, 0.5, 1.0, 1.0],
      ...                   'vertex_highlight' : [0.5, 0.5, 0.5, 1.0],
      ...                   'edge_departure'   : [0, 0, 0, 1],
      ...                   'vertex_active'    : [0.1, 1.0, 0.5, 1.0],
      ...                   'vertex_inactive'  : [0.9, 0.9, 0.9, 0.8],
      ...                   'edge_active'      : [0.1, 0.1, 0.1, 1.0],
      ...                   'edge_inactive'    : [0.8, 0.8, 0.8, 0.3],
      ...                   'bg_color'         : [1, 1, 1, 1]}

    If the graph is not connected then there may be issues with ``Agents``
    that arrive at an edge that points to terminal vertex. If the graph was
    created using :func:`.adjacency2graph` then this is not an issue so
    long as ``q_classes`` key  ``0`` is a :class:`.NullQueue` (or not given).

    .. _Jackson network: http://en.wikipedia.org/wiki/Jackson_network

    Examples
    --------
    The following creates a queueing network with 100 vertices:

    >>> import queueing_tool as qt
    >>> import numpy as np
    >>>
    >>> g = qt.generate_pagerank_graph(100, seed=13)
    >>> q_cl = {2: qt.QueueServer}
    >>> arr = lambda t: t + np.random.gamma(4, 0.0025)
    >>> ser2 = lambda t: t + np.random.exponential(0.025)
    >>> ser3 = lambda t: t + np.random.exponential(4)
    >>> q_ar = {2: {'arrival_f': arr, 'service_f': ser2, 'nServers': 5},
    ...         3: {'service_f': ser3, 'nServers': 10}}
    >>> net = qt.QueueNetwork(g, q_classes=q_cl, q_args=q_ar, seed=13)

    To specify that arrivals enter from type 2 edges and simulate run:

    >>> net.initialize(eType=2)
    >>> net.simulate(n=1000)

    Now we'd like to see how many agents are in type 2 edges:

    >>> nA = [(q.nSystem, q.edge[2]) for q in net.edge2queue if q.edge[3] == 2]
    >>> nA.sort(reverse=True)
    >>> nA[:5]
    [(10, 498), (6, 0), (4, 609), (3, 753), (2, 463)]
    >>> q = net.edge2queue[213]
    >>> q.nSystem, q.nServers
    (0, 10)

    To view the state of the network do the following (note, your
    graph may be rotated):

    >>> pos = gt.sfdp_layout(g, max_iter=100000) # doctest: +SKIP
    >>> net.draw(output="my_network.png", pos=pos, output_size=(700,300)) # doctest: +SKIP
    <...>

    .. figure:: my_network.png
       :align: center
    """

    def __init__(self, g, q_classes=None, q_args=None, seed=None, colors=None,
                    max_agents=1000, blocking='BAS'):

        if not isinstance(blocking, str):
            raise TypeError("blocking must be a string")

        self.nEvents      = 0
        self._t           = 0
        self.max_agents   = max_agents

        self._to_animate  = False
        self._initialized = False
        self._prev_edge   = None
        self._queues      = []
        self._blocking    = True if blocking.lower() != 'rs' else False

        if colors is None :
            colors = {}

        default_colors = {
            'vertex_fill_color': [0.9, 0.9, 0.9, 1.0],
            'vertex_color'     : [0.0, 0.5, 1.0, 1.0],
            'vertex_highlight' : [0.5, 0.5, 0.5, 1.0],
            'edge_departure'   : [0, 0, 0, 1],
            'vertex_active'    : [0.1, 1.0, 0.5, 1.0],
            'vertex_inactive'  : [0.9, 0.9, 0.9, 0.8],
            'edge_active'      : [0.1, 0.1, 0.1, 1.0],
            'edge_inactive'    : [0.8, 0.8, 0.8, 0.3],
            'bg_color'         : [1, 1, 1, 1]
        }

        colors.update(default_colors)

        self.colors = colors

        default_classes = {
            0: NullQueue,
            1: QueueServer,
            2: LossQueue,
            3: LossQueue,
            4: LossQueue
        }

        if q_classes is None:
            q_classes = default_classes
        else:
            for k in set(default_classes.keys()) - set(q_classes.keys()):
                q_classes[k] = default_classes[k]

        if q_args is None:
            q_args  = {k: {} for k in range(5)}
        else:
            for k in set(q_classes.keys()) - set(q_args.keys()):
                q_args[k] = {}

        v_pens = [
            [0.5, 0.5, 0.5, 0.5],
            [0, 0.5, 1, 1],
            [0.133, 0.545, 0.133, 1],
            [0.282, 0.239, 0.545, 1],
            [1, 0.135, 0, 1]
        ]
        q_colors  = {k: {'edge_loop_color'   : [0, 0, 0, 0],
                          'edge_color'       : [0.7, 0.7, 0.7, 0.5],
                          'vertex_fill_color': [0.9, 0.9, 0.9, 1.0],
                          'vertex_color'     : v_pens[k]} for k in range(5)}

        for keys in q_args.keys():
            if keys not in q_colors:
                q_colors[keys] = q_colors[1]

        for key, args in q_args.items():
            if 'colors' not in args:
                args['colors'] = q_colors[key]

        if isinstance(seed, numbers.Integral):
            np.random.seed(seed)

        for k in range(5):
            if k not in q_args:
                q_args[k] = {}

        if g is not None:
            g, qs = _prepare_graph(g, self.colors, q_classes, q_args)

            self.nV = g.num_vertices()
            self.nE = g.num_edges()

            self.edge2queue   = qs
            self.nAgents      = np.zeros(g.num_edges(), int)
            self.out_edges    = [0 for v in range(self.nV)]
            self.in_edges     = [0 for v in range(self.nV)]
            self._route_probs = [0 for v in range(self.nV)]

            def edge_index(e):
                return g.edge_index[e]

            for v in g.vertices():
                vod = g.out_degree(v)
                self.out_edges[v] = [i for i in map(edge_index, g.out_edges(v))]
                self.in_edges[v]  = [i for i in map(edge_index, g.in_edges(v))]
                self._route_probs[v] = [np.float64(1 / vod) for i in range(vod)]

            g.freeze()
            self.g = g

    def __repr__(self):
        the_string = 'QueueNetwork. # nodes: {0}, edges: {1}, agents: {2}'
        return  the_string.format(self.nV, self.nE, np.sum(self.nAgents))

    @property
    def nVertices(self):
        return self.nV

    @property
    def nNodes(self):
        return self.nV

    @property
    def nEdges(self):
        return self.nE

    @property
    def current_time(self):
        return self._t

    @property
    def time(self):
        if len(self._queues) > 0 :
            t = self._queues[-1]._time
        else :
            t = np.infty
        return t

    @property
    def blocking(self):
        return 'BAS' if self._blocking else 'RS'
    @blocking.setter
    def blocking(self, tmp):
        if not isinstance(tmp, str):
            raise TypeError("blocking must be a string")
        self._blocking = True if tmp.lower() != 'rs' else False


    def initialize(self, nActive=1, queues=None, edge=None, eType=None):
        """Prepares the ``QueueNetwork`` for simulation.

        Each :class:`.QueueServer` in the network starts inactive, which
        means they do not accept arrivals from outside the network, and
        they have no agents in their system. Note that in order to
        simulate the :class:`.QueueNetwork`\, there must be at least one
        :class:`.Agent` in the network. This method sets queues to active,
        which then allows agents to arrive from outside the network.

        Parameters
        ----------
        nActive : int (optional, the default is ``1``)
            The number of queues to set as active. The queues are selected randomly.
        queues : int *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) to make active by.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues to make active. Must be either: a
            2-tuple of the edge's source and target vertex indices or an iterable
            of 2-tuples of the edge's source and target vertex indices.
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.

        Raises
        ------
        RuntimeError
            If ``queues``, ``egdes``, and ``eType`` are all ``None`` and
            ``nActive`` is not an integer or is less than 1 then a
            :exc:`~RuntimeError` is raised.
        """
        if queues is None and edge is None and eType is None:
            if nActive >= 1 and isinstance(nActive, numbers.Integral):
                queues = np.random.choice(self.nE, size=nActive, replace=False)
            else:
                msg = ("If queues is None, then nActive must be a strictly "
                       "positive int.")
                raise ValueError(msg)
        else:
            queues = _get_queues(self.g, queues, edge, eType)

        if len(queues) > self.max_agents - np.sum(self.nAgents):
            queues = queues[:self.max_agents]

        for ei in queues:
            self.edge2queue[ei].set_active()
            self.nAgents[ei] = self.edge2queue[ei]._nTotal

        self._queues = [q for q in self.edge2queue]
        self._queues.sort()
        while self._queues[-1]._time == np.infty:
            self._queues.pop()

        self._queues.sort(reverse=True)
        self._initialized  = True


    def transitions(self, return_matrix=True):
        """Returns the transition probabilities for each vertex in the graph.

        Parameters
        ----------
        return_matrix : bool (optional, the default is ``True``\)
            Specifies whether a :class:`~numpy.ndarray` is returned. If
            ``False``\, a :class:`.dict` is returned instead.

        Returns
        -------
        out : an :class:`~numpy.ndarray` or a :class:`.dict`
            The transition probabilities for each vertex in the graph. If
            ``out`` is an :class:`~numpy.ndarray`\, then ``out[v, u]`` returns
            the probability of a transition from vertex ``v`` to vertex ``u``\.
            If ``out`` is a :class:`.dict` then ``out_edge[v][k]`` is the
            probability of moving from vertex ``v`` to the vertex at the head
            of the ``k``\-th out-edge.

        Notes
        -----
        Use ``v.out_edges()`` to get a generator of all out edges from ``v``
        where ``v`` is a :class:`~graph_tool.Vertex`\.

        Examples
        --------
        The default transition matrix is every out edge being equally likely.
        Lets change them randomly:

        >>> import queueing_tool as qt
        >>> g = qt.generate_random_graph(5, seed=96)
        >>> mat = qt.generate_transition_matrix(g, seed=96)
        >>> net = qt.QueueNetwork(g)
        >>> net.set_transitions(mat)
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        {0: [0.194..., 0.805...], 1: [1.0], 2: [0.473..., 0.526...], 3: [0.763..., 0.129..., 0.107...], 4: [0.495..., 0.504...]}
        >>> {k: list(val.keys()) for k, val in qt.graph2dict(g).items()}
        {0: [1, 3], 1: [0], 2: [3, 4], 3: [0, 2, 4], 4: [2, 3]}

        What this shows is the following: when an :class:`.Agent` is at vertex
        ``0`` they will transition to vertex ``1`` with probability ``0.805``
        and route to vertex ``3`` probability ``0.195``\, when at vertex
        ``3`` they will transition to vertex ``4`` with probability ``0.474``
        and route back to vertex ``0`` probability ``0.526``,... etc.
        """
        if return_matrix :
            mat = np.zeros( (self.nV, self.nV) )
            for v in self.g.vertices():
                ind = [e[1] for e in self.g.out_edges(v)]
                mat[v, ind] = self._route_probs[v]
        else :
            mat = {k: value for k, value in enumerate(self._route_probs)}

        return mat


    def set_transitions(self, mat):
        """Change the routing transitions probabilities for the network.

        Parameters
        ----------
        mat : :class:`.dict` or :class:`~numpy.ndarray`
            A transition routing matrix or transition dictionary. If passed a
            dictionary, the keys should be vertex indices and the values are
            the probabilities for each adjacent vertex, or all vertices
            adjacent or otherwise.

        Raises
        ------
        RuntimeError
            A :exc:`.RuntimeError` is raised if: the keys in the :class:`.dict`
            don't match with a vertex index in the graph; or if the
            :class:`~numpy.ndarray` is passed with the wrong shape, must be
            (``nVertices``, ``nVertices``); or the values passed are not
            probabilities (for each vertex they are positive and sum to 1);
        TypeError
            If mat is not a :class:`.dict` or :class:`~numpy.ndarray` a
            TypeError is raised.

        Examples
        --------
        The default transition matrix is every out edge being equally likely:

        >>> import queueing_tool as qt
        >>> g = qt.generate_random_graph(5, seed=10)
        >>> net = qt.QueueNetwork(g)
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        {0: [1.0], 1: [0.5, 0.5], 2: [0.333..., 0.333..., 0.333...], 3: [1.0], 4: [1.0]}

        If you want to change only one vertex's transition probabilities, you
        can do so with the following:

        >>> net.set_transitions({1 : [0.75, 0.25]})
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        {0: [1.0], 1: [0.75, 0.25], 2: [0.333..., 0.333..., 0.333...], 3: [1.0], 4: [1.0]}

        One can generate a transition matrix using
        :func:`.generate_transition_matrix`\. You can change all transition
        probabilities with an :class:`~numpy.ndarray`\:

        >>> mat = qt.generate_transition_matrix(g, seed=10)
        >>> net.set_transitions(mat)
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        {0: [1.0], 1: [0.962..., 0.037...], 2: [0.338..., 0.396..., 0.264...], 3: [1.0], 4: [1.0]}
        """
        if isinstance(mat, dict):
            for key, value in mat.items():
                if key >= self.nV or key < 0 :
                    raise RuntimeError("One of the keys don't correspond to a vertex.")
                elif len(self.out_edges[key]) > 0 and not np.isclose(np.sum(value), 1):
                    raise RuntimeError("Sum of transition probabilities at a vertex was not 1.")
                elif (np.array(value) < 0).any():
                    raise RuntimeError("Some transition probabilities were negative.")

                if len(value) == self.nV:
                    self._route_probs[key] = []
                    for e in self.g.out_edges(key):
                        p = value[e[1]]
                        self._route_probs[key].append(np.float64(p))
                elif len(value) == len(self._route_probs[key]):
                    self._route_probs[key] = []
                    for p in value :
                        self._route_probs[key].append(np.float64(p))

        elif isinstance(mat, np.ndarray):
            non_terminal = np.array([self.g.out_degree(v) > 0 for v in self.g.vertices()])
            if mat.shape != (self.nV, self.nV):
                msg = ("Matrix is the wrong shape, should "
                       "be {0} x {1}.").format(self.nV, self.nV)
                raise RuntimeError(msg)
            elif not np.allclose(np.sum(mat[non_terminal,:], axis=1), 1):
                msg = "Sum of transition probabilities at a vertex was not 1."
                raise RuntimeError(msg)
            elif (mat < 0).any():
                raise RuntimeError("Some transition probabilities were negative.")

            for k in range(self.nV):
                self._route_probs[k] = []
                for e in self.g.out_edges(k):
                    p = mat[k, e[1]]
                    self._route_probs[k].append(np.float64(p))
        else:
            raise TypeError("mat must be a numpy array or a dict.")


    def start_collecting_data(self, queues=None, edge=None, eType=None):
        """Tells the queues to collect data on agents' arrival, service start,
        and departure times.

        If none of the parameters are given then every :class:`.QueueServer`
        will start collecting data.

        Parameters
        ----------
        queues : int, *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) that will start collecting data.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues will collect data. Must be either:
            a 2-tuple of the edge's source and target vertex indices, or an
            iterable of 2-tuples of the edge's source and target vertex
            indices.
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.
        """
        queues = _get_queues(self.g, queues, edge, eType)

        for k in queues:
            self.edge2queue[k].collect_data = True


    def stop_collecting_data(self, queues=None, edge=None, eType=None):
        """Tells the queues to stop collecting data on agents.

        If none of the parameters are given then every :class:`.QueueServer`
        will stop collecting data.

        Parameters
        ----------
        queues : int, *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) that will stop collecting data.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues will stop collecting data. Must be
            either: a 2-tuple of the edge's source and target vertex indices,
            or an iterable of 2-tuples of the edge's source and target
            vertex indices.
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will stop collecting data.
        """
        queues = _get_queues(self.g, queues, edge, eType)

        for k in queues:
            self.edge2queue[k].collect_data = False


    def get_queue_data(self, queues=None, edge=None, eType=None):
        """Fetches data from queues.

        If none of the parameters are given then data from every
        :class:`.QueueServer` is retrieved.

        Parameters
        ----------
        queues : int or an *array_like* of int, (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) whose data will be retrieved.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues to retrieve data from. Must be
            either: a 2-tuple of the edge's source and target vertex indices,
            or an iterable of 2-tuples of the edge's source and target vertex
            indices.
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            to retrieve data from.

        Returns
        -------
        :class:`~numpy.ndarray`
            A six column ``np.array`` of the data. The first, second, and third
            columns represent, respectively, the arrival, service start, and
            departure times of each :class:`.Agent` that has visited the queue.
            The fourth column identifies how many other agents were in the
            queue upon arrival,  the fifth column identifies the number of
            agents in the system, and the sixth column specifies which queue
            this occurred at (by identifying it's edge index).

        Examples
        --------
        Data is not collected by default. Before simulating, by sure to turn it
        on (as well as initialize the network). The following returns data from
        queues with ``eType`` 1 or 3:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.start_collecting_data()
        >>> net.initialize(10)
        >>> net.simulate(2000)
        >>> data = net.get_queue_data(eType=(1,3))

        To get data from an edge connecting two vertices do the following:

        >>> data = net.get_queue_data(edge=(1,50))

        To get data from several edges do the following:

        >>> data = net.get_queue_data(edge=[(1,50), (10,91), (99,99)])

        You can specify the edge indices as well:

        >>> data = net.get_queue_data(queues=(20, 14, 0, 4))
        """
        queues = _get_queues(self.g, queues, edge, eType)

        data = np.zeros((0, 6))
        for q in queues:
            dat = self.edge2queue[q].fetch_data()

            if len(dat) > 0:
                data = np.vstack( (data, dat) )

        return data


    def get_agent_data(self, queues=None, edge=None, eType=None):
        """Fetches data from queues, and organizes it by agent.

        If none of the parameters are given then data from every
        :class:`.QueueServer` is retrieved.

        Parameters
        ----------
        queues : int or *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) whose data will be retrieved.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues to retrieve agent data from. Must
            be either: a 2-tuple of the edge's source and target vertex
            indices, or an iterable of 2-tuples of the edge's source and target
            vertex indices.
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            to retrieve agent data from.

        Returns
        -------
        :class:`.dict`
            Returns a ``dict`` where the keys are the :class:`.Agent`\'s
            ``issn`` and the values are :class:`~numpy.ndarray`\s for that
            :class:`.Agent`\'s data. The first, second, and third columns
            represent, respectively, the arrival, service start, and departure
            times of that :class:`.Agent` at a queue; The fourth column
            identifies how many other agents were waiting to be serviced upon
            arrival, the fifth column identifies the number of agents in the
            system, and the sixth column specifies this queue by its edge index.
        """
        queues = _get_queues(self.g, queues, edge, eType)

        data = {}
        for qid in queues:
            for issn, dat in self.edge2queue[qid].data.items():
                datum = np.zeros( (len(dat), 6) )
                datum[:,:5] = np.array(dat)
                datum[:, 5] = qid
                if issn in data:
                    data[issn] = np.vstack( (data[issn], datum) )
                else:
                    data[issn] = datum

        dType = [
            ('a', float),
            ('s', float),
            ('d', float),
            ('q', float),
            ('n', float),
            ('id', float)
        ]
        for issn, dat in data.items():
            datum = np.array([tuple(d) for d in dat.tolist()], dtype=dType)
            datum = np.sort(datum, order='a')
            data[issn] = np.array([tuple(d) for d in datum])

        return data


    def draw(self, update_colors=True, **kwargs):
        """Draws the network. The coloring of the network corresponds to the
        number of agents at each queue.

        Parameters
        ----------
        update_colors : ``bool`` (optional, the default is ``True``).
            Specifies whether all the colors are updated.
        **kwargs
            Any parameters to pass to :func:`~graph_tool.draw.graph_draw`.
        output_size : :class:`.tuple` (optional, the default is ``(700, 700)``).
            This is :func:`~graph_tool.draw.graph_draw` parameter for
            specifying the size of canvas.
        output : str (optional, the default is ``None``)
            Specifies the directory where the drawing is saved. If output is
            ``None``, then the results are drawn using GraphViz.

        Notes
        -----
        There are several parameters passed to :func:`~graph_tool.draw.graph_draw`
        by default. The following parameters are :class:`~graph_tool.PropertyMap`\s
        that are automatically set to the graph when a ``QueueNetwork``
        instance is created. These property maps include:

          * ``vertex_color``, ``vertex_fill_color``, ``vertex_size``,
            ``vertex_pen_width``, ``pos``.
          * ``edge_color``, ``edge_control_points``, ``edge_marker_size``,
            ``edge_pen_width``.

        Each of these properties are used by ``draw`` to style the canvas.
        There is also a parameter that sets the background color of the canvas,
        which is the ``bg_color`` parameter. This color is defined in the class
        property ``QueueNetwork.colors`` (which is a :class:`.dict`).

        If any of these parameters are supplied as arguments to ``draw`` then
        the passed arguments are used over the defaults.

        Drawing output may differ between Python 2 and Python 3. Use Python 3
        for with better output.

        Examples
        --------
        To draw the current state of the network, call:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.draw() # doctest: +SKIP

        If you specify a file name and location, the drawing will be saved to
        disk. For example, to save the drawing to the current working directory
        do the following:

        >>> net.draw(output="current_state.png", output_size=(400,400)) # doctest: +SKIP

        .. figure:: current_state.png
            :align: center

        The shade of each edge depicts how many agents are located at the
        corresponding queue. The shade of each vertex is determined by the
        total number of inbound agents. Although loops are not visible by
        default, the vertex that corresponds to a loop shows how many agents
        are in that loop.

        There are several additional parameters that can be passed -- all
        :func:`~graph_tool.draw.graph_draw` parameters are valid. For example,
        to show the vertex number in the graph, one could do the following:

        >>> net.draw(vertex_text=net.g.vertex_index) # doctest: +SKIP
        """
        if update_colors :
            self._update_all_colors()

        kwargs = self._update_kwargs(kwargs, out='output' in kwargs, update_props=True)
        if 'bg_color' not in kwargs:
            kwargs['bg_color'] = self.colors['bg_color']

        ans = self.g.draw_graph(**kwargs)


    def show_active(self, **kwargs):
        """Draws the network, highlighting active queues (queues that accept
        arrivals from outside the network).

        The colored vertices represent vertices that have at least one queue
        on an in-edge that is active. Dark edges represent queues that are
        active, light edges represent queues that are inactive.

        Parameters
        ----------
        **kwargs
            Any additional parameters to pass to :meth:`.draw`, and
            :func:`~graph_tool.draw.graph_draw`.

        Notes
        -----
        The colors are defined by the class attribute ``colors``. The relevant
        keys are ``vertex_active``, ``vertex_inactive``, ``edge_active``, and
        ``edge_inactive``.
        """
        g  = self.g
        for v in g.vertices():
            self.g.set_vp(v, 'vertex_color', [0, 0, 0, 0.9])
            is_active = False
            my_iter   = g.in_edges(v) if g.is_directed() else g.out_edges(v)
            for e in my_iter:
                ei = g.edge_index[e]
                if self.edge2queue[ei]._active:
                    is_active = True
                    break
            if is_active:
                self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_active'])
            else:
                self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_inactive'])

        for e in g.edges():
            ei = g.edge_index[e]
            if self.edge2queue[ei]._active:
                self.g.set_ep(e, 'edge_color', self.colors['edge_active'])
            else :
                self.g.set_ep(e, 'edge_color', self.colors['edge_inactive'])

        self.draw(update_colors=False, **kwargs)
        self._update_all_colors()


    def show_type(self, eType, **kwargs):
        """Draws the network, highlighting queues of a certain type.

        The colored vertices represent self loops of type ``eType``. Dark edges
        represent queues of type ``eType``.

        Parameters
        ----------
        eType : int
            The type of vertices and edges to be shown.
        **kwargs
            Any additional parameters to pass to :meth:`.draw`, and
            :func:`~graph_tool.draw.graph_draw`.

        Notes
        -----
        The colors are defined by the class attribute ``colors``\. The
        relevant colors are ``vertex_active``\, ``vertex_inactive``\,
        ``vertex_highlight``\, ``edge_active``\, and ``edge_inactive``\.

        Examples
        --------
        The following code highlights all edges with edge type ``2``. If the
        edge is a loop then the vertex is highlighted as well. In this case
        all edges with edge type ``2`` happen to be loops.

        >>> import queueing_tool as qt
        >>> import matplotlib.pyplot as plt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> fname = 'edge_type_2.png'
        >>> net.show_type(2, output_size=(400,400), output=fname) # doctest: +SKIP

        .. figure:: edge_type_2.png
           :align: center
        """
        for v in self.g.vertices():
            e = (v, v)
            if self.g.is_edge(e) and self.g.ep(e, 'eType') == eType:
                ei = self.g.edge_index[e]
                self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_highlight'])
                self.g.set_vp(v, 'vertex_color', self.edge2queue[ei].colors['vertex_color'])
            else:
                self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_inactive'])
                self.g.set_vp(v, 'vertex_color', [0, 0, 0, 0.9])

        for e in self.g.edges():
            if self.g.ep(e, 'eType') == eType:
                self.g.set_ep(e, 'edge_color', self.colors['edge_active'])
            else :
                self.g.set_ep(e, 'edge_color', self.colors['edge_inactive'])

        self.draw(update_colors=False, **kwargs)
        self._update_all_colors()


    def _update_all_colors(self):
        do  = [True for v in range(self.nV)]
        for q in self.edge2queue:
            e = q.edge[:2]
            v = q.edge[1]
            if q.edge[0] == q.edge[1]:
                self.g.set_ep(e, 'edge_color', q._current_color(1))
                self.g.set_vp(v, 'vertex_color', q._current_color(2))
                if q.edge[3] != 0:
                    self.g.set_vp(v, 'vertex_fill_color', q._current_color())
                do[v] = False
            else :
                self.g.set_ep(e, 'edge_color', q._current_color())
                if do[v]:
                    self._update_vertex_color(v)
                    do[v] = False
                if do[q.edge[0]]:
                    self._update_vertex_color(q.edge[0])
                    do[q.edge[0]] = False

    def _update_vertex_color(self, v):
        ee  = (v, v)
        ee_is_edge = self.g.is_edge(ee)
        eei = self.g.edge_index[ee] if ee_is_edge else 0

        if not ee_is_edge or (ee_is_edge and self.edge2queue[eei].edge[3] == 0):
            nSy = 0
            cap = 0
            for ei in self.in_edges[v] :
                nSy += self.edge2queue[ei].nSystem
                cap += self.edge2queue[ei].nServers

            div = 5 if cap <= 1 else (2 * cap)
            tmp = 0.9 - min(nSy / div, 0.9)

            color = [i * tmp / 0.9 for i in self.colors['vertex_fill_color']]
            color[3] = 1.0 - tmp
            self.g.set_vp(v, 'vertex_fill_color', color)
            if not ee_is_edge:
                self.g.set_vp(v, 'vertex_color', self.colors['vertex_color'])

    def _update_graph_colors(self, qedge):
        e  = qedge[:2]
        v  = qedge[1]
        if self._prev_edge is not None:
            pe = self._prev_edge[:2]
            pv = self._prev_edge[1]
            q  = self.edge2queue[self._prev_edge[2]]

            if pe[0] == pe[1]:
                self.g.set_ep(pe, 'edge_color', q._current_color(1))
                self.g.set_vp(pv, 'vertex_color', q._current_color(2))
                if q.edge[3] != 0:
                    self.g.set_vp(v, 'vertex_fill_color', q._current_color())

            else:
                self.g.set_ep(pe, 'edge_color', q._current_color())
                self._update_vertex_color(pv)

        q = self.edge2queue[qedge[2]]
        if qedge[0] == qedge[1]:
            self.g.set_ep(e, 'edge_color', q._current_color(1))
            self.g.set_vp(v, 'vertex_color', q._current_color(2))
            if q.edge[3] != 0:
                self.g.set_vp(v, 'vertex_fill_color', q._current_color())

        else:
            self.g.set_ep(e, 'edge_color', q._current_color())
            self._update_vertex_color(v)


    def _add_arrival(self, ei, agent, t=None):
        """Used to force an arrival to a queue.

        Parameters
        ----------
        ei : int
            Identifies which queue the agent is going to be added; is the
            queue's edge index.
        agent : ``Agent``
            The agent that is to be added to the queue.
        t : float (optional)
            The arrival time of the agent. If not given then it is set to that
            queue's time-of-next-event (its ``time``\) plus ``1``\.
        """
        q   = self.edge2queue[ei]
        qt  = q._time
        if t is None :
            t = q._time + 1 if q._time < np.infty else self._queues[-1]._time + 1

        agent._time = t
        q._add_arrival(agent)

        if qt == np.infty and q._time < np.infty :
            self._queues.append(q)

        self._queues.sort(reverse=True)


    def next_event_description(self):
        """Returns whether the next event is either an arrival or a departure
        and the edge index corresponding to that queue.

        Returns
        -------
        des : str
            Indicates whether the next event is an arrival, a departure, or
            nothing; returns ``'Arrival'``, ``'Departure'``, or ``'Nothing'``.
        edge : int or ``None``
            The edge index of the edge that this event will occur at. If there
            are no events then ``None`` is returned.
        """
        if len(self._queues) == 0 :
            ans = ('Nothing', None)
        else :
            ad1 = 'Arrival' if self._queues[-1].next_event_description() == 1 else 'Departure'
            ad2 = self._queues[-1].edge[2]
            ans = (ad1, ad2)
        return ans


    def _simulate_next_event(self, slow=True):
        n = len(self._queues)
        if n == 0 :
            self._t = np.infty
            return

        q1  = self._queues.pop()
        q1t = q1._time
        e1  = q1.edge[2]

        event   = q1.next_event_description()
        self._t = q1t
        self.nEvents += 1

        if event == 2 : # This is a departure
            e2  = q1._departures[0].desired_destination(self, q1.edge)
            q2  = self.edge2queue[e2]
            q2t = q2._time

            if q2.at_capacity() and e2 != e1 :
                q2.nBlocked += 1
                q1._departures[0].blocked += 1
                if self._blocking :
                    t = q2._departures[0]._time + EPS * uniform(0.33, 0.66)
                    q1.delay_service(t)
                else :
                    q1.delay_service()
            else :
                agent = q1.next_event()
                agent._time = q1t

                q2._add_arrival(agent)
                self.nAgents[e1] = q1._nTotal
                self.nAgents[e2] = q2._nTotal

                if slow :
                    self._update_graph_colors(qedge=q1.edge)
                    self._prev_edge = q1.edge

                if q2._active and self.max_agents < np.infty and np.sum(self.nAgents) > self.max_agents - 1 :
                    q2._active = False

                q2.next_event()
                self.nAgents[e2] = q2._nTotal

                if slow :
                    self._update_graph_colors(qedge=q2.edge)
                    self._prev_edge = q2.edge

            if q1._time < np.infty :
                if q2._time < q2t < np.infty and e2 != e1 :
                    if n > 2 :
                        oneBisectSort(self._queues, q1, q2t, n-1)
                    else :
                        if q1._time < q2._time :
                            self._queues.append(q1)
                        else :
                            self._queues.insert(0, q1)
                elif q2._time < q2t and e2 != e1 :
                    if n == 1 :
                        if q1._time < q2._time :
                            self._queues.append(q2)
                            self._queues.append(q1)
                        else :
                            self._queues.append(q1)
                            self._queues.append(q2)
                    else :
                        twoSort(self._queues, q1, q2, n-1)
                else :
                    if n == 1 :
                        self._queues.append(q1)
                    else :
                        bisectSort(self._queues, q1, n-1)
            else :
                if q2._time < q2t < np.infty :
                    if n > 2 :
                        oneSort(self._queues, q2t, n-1)
                elif q2._time < q2t :
                    if n == 1 :
                        self._queues.append(q2)
                    else :
                        bisectSort(self._queues, q2, n-1)

        elif event == 1 : # This is an arrival
            if q1._active and self.max_agents < np.infty and np.sum(self.nAgents) > self.max_agents - 1 :
                q1._active = False

            q1.next_event()
            self.nAgents[e1] = q1._nTotal

            if slow :
                self._update_graph_colors(qedge=q1.edge)
                self._prev_edge  = q1.edge

            if q1._time < np.infty :
                if n == 1 :
                    self._queues.append(q1)
                else :
                    bisectSort(self._queues, q1, n - 1)

        if self._to_animate :
            if self._to_disk :
                self._kwargs['output'] = self._outdir + '%s.' % (self._count) + self._fmt
                self.draw(update_colors=False, **self._kwargs)
                self._count += 1
            else :
                self._window.graph.regenerate_surface(lazy=False)
                self._window.graph.queue_draw()

            return True


    def animate(self, out=None, n=10, t=None, **kwargs):
        """Animates the network as it's simulating.

        The animations can be saved to disk or view in interactive mode.
        Closing the window ends the animation if viewed in interactive mode.

        Parameters
        ----------
        out : str (optional)
            The location where the frames for the images will be saved. If this
            parameter is not given, then the animation is shown in interactive
            mode.
        n : int (optional, the default is 10)
            Indicates the number of frames to save to disk. This parameter is
            only used if ``out`` is passed.
        t : float (optional)
            The amount of simulation time to simulate forward. If given, and
            ``out`` is given, ``t`` is used instead of ``n``.
        **kwargs :
            This method calls :class:`~graph_tool.draw.GraphWindow`, ``kwargs``
            allows you to specify any extra parameters to pass to it.

        Notes
        -----
        There are several parameters passed to :func:`~graph_tool.draw.GraphWindow`
        by default. The following parameters are property maps that are
        automatically set to the graph when a ``QueueNetwork`` instance is
        created. These property maps include:

            * ``vertex_color``, ``vertex_fill_color``, ``vertex_size``,
              ``vertex_pen_width``, ``pos``.
            * ``edge_color``, ``edge_control_points``, ``edge_marker_size``,
              ``edge_pen_width``.

        Each of these properties are used by ``animate`` to style the canvas.
        Also, the ``bg_color`` parameter is defined in the :class:`.dict`
        ``QueueNetwork.colors``\. The ``output_size`` defaults to
        ``(700, 700)``\. If any of these parameters are supplied as arguments
        then they are used over the defaults.

        See the documentation of :func:`~graph_tool.draw.graph_draw` for a
        more on the documentation of :class:`~graph_tool.draw.GraphWindow`

        Drawing output may differ between Python 2 and Python 3. Use Python 3
        for with better output.

        Raises
        ------
        InitializationError
            Will raise a ``InitializationError`` if the ``QueueNetwork`` has
            not been initialized. Call :meth:`.initialize` before running.

        Examples
        --------
        This function works similarly to ``QueueNetwork``\'s :meth:`.draw`
        method. To animate the network in interactive mode do the following:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.initialize()
        >>> net.animate(output_size=(400,400)) # doctest: +SKIP

        To stop the animation just close the window. If you want to write the frames
        to disk run something like the following:

        >>> kwargs = {
        ...     'count': 25,
        ...     'output_size': (400,400),
        ...     'vertex_size': 15
        ... }
        >>> net.animate(out="./test", **kwargs) # doctest: +SKIP

        The above code outputs the frames in the current working directory and
        outputs 25 ``png`` images whose names start with ``test`` e.g.
        ``test0.png``\, ``test1.png``\, ... etc. Also, the vertex size for each
        vertex was changed from the default (of 8) to 15.
        """
        if not self._initialized:
            msg = ("Network has not been initialized. "
                   "Call '.initialize()' first.")
            raise InitializationError(msg)

        self._to_animate = True
        self._update_all_colors()

        if out is None:
            kwargs = self._update_kwargs(kwargs, out=False, update_props=True)

            if 'bg_color' not in kwargs:
                kwargs['bg_color'] = self.colors['bg_color']

            window, gtk_main_quit, gtk_main, GObject = self.g.get_window(**kwargs)

            self._to_disk = False
            self._window  = window

            cid = GObject.idle_add(self._simulate_next_event)
            self._window.connect("delete_event", gtk_main_quit)
            self._window.show_all()
            gtk_main()
        else:
            kwargs = self._update_kwargs(kwargs, out=True, update_props=False)
            self._fmt     = kwargs['fmt'] if 'fmt' in kwargs else 'png'
            self._count   = 0
            self._to_disk = True
            self._outdir  = out
            self._kwargs  = kwargs

            if t is None:
                for k in range(n):
                    self._simulate_next_event(slow=True)
            else:
                now = self._t
                while self._t < now + t:
                    self._simulate_next_event(slow=True)

        self._to_animate = False
        self._to_disk    = False


    def simulate(self, n=1, t=None):
        """This method simulates the network forward for a specific number of
        events ``n`` or for a specified amount of simulation time ``t``\.

        Parameters
        ----------
        n : int (optional, the default is 1)
            The number of events to simulate. If ``t`` is not given then this
            parameter is used.
        t : float (optional)
            The amount of simulation time to simulate forward. If given, ``t``
            is used instead of ``n``.

        Raises
        ------
        InitializationError
            Will raise a ``InitializationError`` if the ``QueueNetwork`` has
            not been initialized. Call :meth:`.initialize` before running.

        Examples
        --------
        Let ``net`` denote your instance of a ``QueueNetwork``. Before you
        simulate, you need to initialize the network, which allows arrivals
        from outside the network. To initialize with 2 (random chosen) edges
        accepting arrivals run:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=50)
        >>> net = qt.QueueNetwork(g, seed=50)
        >>> net.initialize(2)

        To simulate the network 50000 events run:

        >>> nE = net.nEvents
        >>> net.simulate(50000)
        >>> net.nEvents - nE
        50000

        To simulate the network for at least 25 simulation time units run:

        >>> nE = net.nEvents
        >>> t0 = net.current_time
        >>> net.simulate(t=75)
        >>> t1 = net.current_time
        >>> round(float(t1 - t0), 3)
        75.003
        >>> net.nEvents - nE
        20926
        """
        if not self._initialized:
            msg = ("Network has not been initialized. "
                   "Call '.initialize()' first.")
            raise InitializationError(msg)
        if t is None :
            for k in range(n):
                self._simulate_next_event(slow=False)
        else :
            now = self._t
            while self._t < now + t :
                self._simulate_next_event(slow=False)


    def _update_kwargs(self, kwargs, out=False, update_props=True):

        output_size = (700, 700)

        arg1 = 'geometry' if out else 'output_size'
        arg2 = 'output_size' if out else 'geometry'

        if arg1 in kwargs:
            if arg2 not in kwargs:
                kwargs[arg2] = kwargs[arg1]

            del kwargs[arg1]

        if arg2 not in kwargs :
            kwargs[arg2] = output_size

        if update_props:
            vertex_props = set(['vertex_color', 'vertex_fill_color', 'vertex_size',
                                'vertex_pen_width', 'pos'])

            edge_props = set(['edge_color', 'edge_control_points',
                              'edge_marker_size', 'edge_pen_width'])

            for param in vertex_props:
                if param not in kwargs:
                    kwargs[param] = {}

            for param in edge_props:
                if param not in kwargs:
                    kwargs[param] = {}

        return kwargs


    def reset_colors(self):
        """Resets all edge and vertex colors to their default values."""
        for k, e in enumerate(self.g.edges()):
            self.g.set_ep(e, 'edge_color', self.edge2queue[k].colors['edge_color'])
        for v in self.g.vertices():
            self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_fill_color'])


    def clear(self):
        """Resets the queue to its initial state.

        The attributes ``t``, ``nEvents``, ``nAgents`` are set to zero
        :meth:`.reset_colors` is called; and the :class:`.QueueServer` method
        :meth:`.clear` is called for each queue in the network.

        Notes
        -----
        ``QueueNetwork`` must be re-initialized before any simulations can run.
        """
        self._t           = 0
        self.nEvents      = 0
        self.nAgents      = np.zeros(self.nE, int)
        self._queues      = []
        self._to_animate  = False
        self._prev_edge   = None
        self._initialized = False
        self.reset_colors()
        for q in self.edge2queue :
            q.clear()


    def clear_data(self, queues=None, edge=None, eType=None):
        """Clears data from queues.

        If none of the parameters are given then every queue's data is cleared.

        Parameters
        ----------
        queues : int or an iterable of int (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) whose data will be cleared.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues' data to clear. Must be either: a
            2-tuple of the edge's source and target vertex indices or an iterable
            of 2-tuples of the edge's source and target vertex indices.
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will have their data cleared.
        """
        queues = _get_queues(self.g, queues, edge, eType)

        for k in queues :
            self.edge2queue[k].data = {}


    def copy(self):
        """Returns a deep copy of itself."""
        net              = QueueNetwork(None)
        net.g            = self.g.copy()
        net.max_agents   = copy.copy(self.max_agents)
        net.nV           = copy.copy(self.nV)
        net.nE           = copy.copy(self.nE)
        net.nAgents      = copy.copy(self.nAgents)
        net.nEvents      = copy.copy(self.nEvents)
        net._t           = copy.copy(self._t)
        net._initialized = copy.copy(self._initialized)
        net._prev_edge   = copy.copy(self._prev_edge)
        net._to_animate  = copy.copy(self._to_animate)
        net._blocking    = copy.copy(self._blocking)
        net.colors       = copy.deepcopy(self.colors)
        net.out_edges    = copy.deepcopy(self.out_edges)
        net.in_edges     = copy.deepcopy(self.in_edges)
        net.edge2queue   = copy.deepcopy(self.edge2queue)
        net._route_probs = copy.deepcopy(self._route_probs)

        if net._initialized :
            net._queues = [q for q in net.edge2queue]
            net._queues.sort()
            while net._queues[-1]._time == np.infty :
                net._queues.pop()

            net._queues.sort(reverse=True)

        return net



def _get_queues(g, queues, edge, eType):
    """Used to specify edge indices from different types of arguments."""
    INT = numbers.Integral
    if isinstance(queues, INT):
        queues = [queues]

    elif queues is None:
        if edge is not None:
            if isinstance(edge, tuple):
                if isinstance(edge[0], INT) and isinstance(edge[1], INT):
                    queues = [g.edge_index[edge]]
            elif isinstance(edge[0], collections.Iterable):
                if np.array([len(e) == 2 for e in edge]).all():
                    queues = [g.edge_index[e] for e in edge]
            else:
                queues = [g.edge_index[edge]]
        elif eType is not None:
            tmp = np.array([g.ep(e, 'eType') for e in g.edges()])
            queues = np.where(np.in1d(tmp, eType) )[0]

        if queues is None:
            queues = range(g.num_edges())

    return queues
