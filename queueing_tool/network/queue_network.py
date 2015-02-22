import graph_tool.all as gt
import numpy   as np
import numbers
import copy
import sys

from .. generation import _prepare_graph
from .. queues     import NullQueue, QueueServer, LossQueue

from .sorting      import oneBisectSort, bisectSort, oneSort, twoSort

from numpy         import infty
from gi.repository import Gtk, GObject

EPS = np.float64(1e-10)

class QueueNetwork :
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
    in_edges : :class:`.dict`
        A mapping between vertex indices and the in-edges at that vertex.
        Specifically, ``in_edges[v]`` returns a list containing the edge index
        for all edges with the head of the edge at ``v``, where ``v`` is the
        the vertex's index number.
    out_edges : :class:`.dict`
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
    time : float
        The time of the last event.
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
    If only :class:`.Agent`\s enter the network, then ``QueueNetwork`` instance
    is a `Jackson network`_. The default transition probabilities at a vertex 
    ``v`` is ``1 / v.out_degree()`` for each adjacent vertex.

    * This class must be initialized before any simulations can take place. To
      initialize, call the :meth:`~initialize` method. If any of the queues are
      altered, make sure to re-run the ``initialize`` method again.
    * When simulating the network, the departure of an agent from one queue
      coincides with an arrival to another queue. There is no time lag between
      these events.

    .. _Jackson network: http://en.wikipedia.org/wiki/Jackson_network    


    The following properties are assigned as a :class:`~graph_tool.PropertyMap`
    to the graph; their default values for each edge or vertex is shown:
        
        * ``vertex_pen_width``: ``1.1``,
        * ``vertex_size``: ``8``,
        * ``edge_control_points``: ``[]``
        * ``edge_marker_size``: ``8``
        * ``edge_pen_width``: ``1.25``

    There are also property maps created for graph visualization, they are
    ``vertex_color``\, ``vertex_fill_color``\, ``pos``\, and ``edge_color``\.
    The default colors, which are used by various methods, are:

    >>> default_colors = { 'vertex_fill_color' : [0.9, 0.9, 0.9, 1.0],
    ...                    'vertex_color'      : [0.0, 0.5, 1.0, 1.0],
    ...                    'vertex_highlight'  : [0.5, 0.5, 0.5, 1.0],
    ...                    'edge_departure'    : [0, 0, 0, 1], 
    ...                    'vertex_active'     : [0.1, 1.0, 0.5, 1.0],
    ...                    'vertex_inactive'   : [0.9, 0.9, 0.9, 0.8],
    ...                    'edge_active'       : [0.1, 0.1, 0.1, 1.0],
    ...                    'edge_inactive'     : [0.8, 0.8, 0.8, 0.3],
    ...                    'bg_color'          : [1, 1, 1, 1]}

    If the graph is not connected then there may be issues with ``Agents``
    that arrive at an edge that points to terminal vertex. If the graph was 
    created using :func:`.adjacency2graph` then this is not an issue, so
    long as ``q_classes`` key  ``0`` is a :class:`.NullQueue` (note that ``0``
    does not need to be a key of the ``q_classes`` parameter, but if it is it
    should be set to :class:`.NullQueue`).

    Examples
    --------
    The following creates a queueing network with 100 vertices. 

    >>> g   = qt.generate_pagerank_graph(100, seed=13)
    >>> net = qt.QueueNetwork(g, seed=13)
    """

    def __init__(self, g, q_classes=None, q_args=None, seed=None, colors=None, max_agents=1000, blocking='BAS') :

        if not isinstance(blocking, str) :
            raise TypeError("blocking must be a string")

        self.nEvents      = 0
        self.t            = 0
        self.max_agents   = max_agents

        self._to_animate  = False
        self._initialized = False
        self._prev_edge   = None
        self._queues      = []
        self._blocking    = True if blocking.lower() != 'rs' else False

        if colors is None :
            colors = {}

        default_colors    = { 'vertex_fill_color' : [0.9, 0.9, 0.9, 1.0],
                              'vertex_color'      : [0.0, 0.5, 1.0, 1.0],
                              'vertex_highlight'  : [0.5, 0.5, 0.5, 1.0],
                              'edge_departure'    : [0, 0, 0, 1],
                              'vertex_active'     : [0.1, 1.0, 0.5, 1.0],
                              'vertex_inactive'   : [0.9, 0.9, 0.9, 0.8],
                              'edge_active'       : [0.1, 0.1, 0.1, 1.0],
                              'edge_inactive'     : [0.8, 0.8, 0.8, 0.3],
                              'bg_color'          : [1, 1, 1, 1]}

        for key, value in default_colors.items() :
            if key not in colors :
                colors[key] = value

        self.colors = colors

        default_classes = {0 : NullQueue, 1 : QueueServer, 2 : LossQueue,
                           3 : LossQueue, 4 : LossQueue}

        if q_classes is None :
            q_classes = default_classes
        else :
            for k in set(default_classes.keys()) - set(q_classes.keys()) :
                q_classes[k] = default_classes[k]

        if q_args is None :
            q_args  = {k : {} for k in range(5)}
        else :
            for k in set(q_classes.keys()) - set(q_args.keys()) :
                q_args[k] = {}

        v_pens    = [[0.5, 0.5, 0.5, 0.5], [0, 0.5, 1, 1], [0.133, 0.545, 0.133, 1],
                     [0.282, 0.239, 0.545, 1], [1, 0.135, 0, 1]]
        q_colors  = {k : {'edge_loop_color'   : [0, 0, 0, 0],
                          'edge_color'        : [0.7, 0.7, 0.7, 0.5],
                          'vertex_fill_color' : [0.9, 0.9, 0.9, 1.0],
                          'vertex_color'      : v_pens[k]} for k in range(5)}

        for key, args in q_args.items() :
            if 'colors' not in args :
                args['colors'] = q_colors[key]

        if isinstance(seed, numbers.Integral) :
            np.random.seed(seed)
            gt.seed_rng(seed)

        for k in range(5) :
            if k not in q_args :
                q_args[k] = {}

        if g is not None :
            if isinstance(g, str) or isinstance(g, gt.Graph) :
                g, qs = _prepare_graph(g, self.colors, q_classes, q_args)
            else :
                raise TypeError("The Parameter `g` needs to be either a graph-tool Graph, a string.")

            self.edge2queue   = qs
            self.nAgents      = np.zeros(g.num_edges(), int)
            self.out_edges    = {}
            self.in_edges     = {}
            self._route_probs = {}

            def edge_index(e) :
                return g.edge_index[e]

            for v in g.vertices() :
                vi  = int(v)
                vod = v.out_degree()
                self.out_edges[vi]    = [i for i in map(edge_index, list(v.out_edges()))]
                self.in_edges[vi]     = [i for i in map(edge_index, list(v.in_edges()))]
                self._route_probs[vi] = [np.float64(1 / vod) for i in range(vod)]

            self.g  = g
            self.nV = g.num_vertices()
            self.nE = g.num_edges()

    def __repr__(self) :
        return 'QueueNetwork. # nodes: %s, edges: %s, agents: %s' % (self.nV, self.nE, np.sum(self.nAgents))

    @property
    def nVertices(self):
        return self.nV
    @nVertices.deleter
    def nVertices(self): 
        pass
    @nVertices.setter
    def nVertices(self, tmp): 
        pass

    @property
    def nEdges(self):
        return self.nE
    @nEdges.deleter
    def nEdges(self): 
        pass
    @nEdges.setter
    def nEdges(self, tmp): 
        pass

    @property
    def time(self):
        return self.t
    @time.deleter
    def time(self): 
        pass
    @time.setter
    def time(self, tmp): 
        pass

    @property
    def blocking(self):
        return 'BAS' if self._blocking else 'RS'
    @blocking.deleter
    def blocking(self): 
        pass
    @blocking.setter
    def blocking(self, tmp):
        if not isinstance(tmp, str) :
            raise TypeError("blocking_type must be a string")
        self._blocking = True if tmp.lower() != 'rs' else False


    def initialize(self, nActive=1, queues=None, edge=None, eType=None) :
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
            2-tuple of the edge's source and target vertex indices, an iterable
            of 2-tuples of the edge's source and target vertex indices, or an 
            iterable of :class:`~graph_tool.Edge`\(s).
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
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edge is not None :
                if not isinstance(edge[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edge]
                elif isinstance(edge[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edge]
                else :
                    queues = [self.g.edge_index[self.g.edge(edge[0], edge[1])]]
            elif eType is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), eType) )[0]
            elif nActive >= 1 and isinstance(nActive, numbers.Integral) :
                queues = np.arange(self.nE)  
                np.random.shuffle(queues)
                queues = queues[:nActive]
            else :
                raise RuntimeError("If queues is None, then nActive must be a strictly positive int.")

        if len(queues) > self.max_agents - np.sum(self.nAgents) :
            queues = queues[:self.max_agents]

        for ei in queues :
            self.edge2queue[ei].set_active()
            self.nAgents[ei] = self.edge2queue[ei]._nTotal

        self._queues = [q for q in self.edge2queue]
        self._queues.sort()
        while self._queues[-1]._time == infty :
            self._queues.pop()

        self._queues.sort(reverse=True)
        self._initialized  = True


    def transitions(self, return_matrix=True) :
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

        >>> g = qt.generate_random_graph(5, seed=96)
        >>> mat = qt.generate_transition_matrix(g, seed=96)
        >>> net = qt.QueueNetwork(g)
        >>> net.set_transitions(mat)
        >>> net.transitions(False)
        {0: [0.195, 0.805], 1: [1.0], 2: [1.0], 3: [0.474, 0.526], 4: [0.855, 0.145]}
        >>> qt.graph2dict(g)[0]
        {0: [3, 1], 1: [0], 2: [4], 3: [4, 0], 4: [2, 3]}

        What this shows is the following: when an :class:`.Agent` is at vertex
        ``0`` they will transition to vertex ``1`` with probability ``0.805``
        and route to vertex ``3`` probability ``0.195``\, when at vertex
        ``3`` they will transition to vertex ``4`` with probability ``0.474``
        and route back to vertex ``0`` probability ``0.526``,... etc.
        """
        if return_matrix :
            mat = np.zeros( (self.nV, self.nV) )
            for v in self.g.vertices() :
                vi  = int(v)
                ind = [int(e.target()) for e in v.out_edges()]
                mat[vi, ind] = self._route_probs[vi]
        else :
            mat = copy.deepcopy(self._route_probs)

        return mat


    def set_transitions(self, mat) :
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
            don't match with a vertex index in the graph; or the sum of the
            transition probabilities out of a vertex is not 1 (for non-terminal
            edges); or if the a :class:`~numpy.ndarray` is passed with the
            wrong shape, must be (``nVertices``, ``nVertices``).

        Examples
        --------
        The default transition matrix is every out edge being equally likely:

        >>> g = qt.generate_random_graph(5, seed=10)
        >>> net = qt.QueueNetwork(g)
        >>> net.transitions(False)
        {0: [1.0], 1: [0.5, 0.5], 2: [0.333, 0.333, 0.333], 3: [1.0], 4: [1.0]}

        If you want to change only one vertex's transition probabilities, you
        can do so with the following:

        >>> net.set_transitions({1 : [0.75, 0.25]})
        >>> net.transitions(False)
        {0: [1.0], 1: [0.75, 0.25], 2: [0.333, 0.333, 0.333], 3: [1.0], 4: [1.0]}

        One can generate a transition matrix using 
        :func:`.generate_transition_matrix`\. You can change all transition
        probabilities with an :class:`~numpy.ndarray`\:

        >>> mat = qt.generate_transition_matrix(g, seed=10)
        >>> net.set_transitions(mat)
        >>> net.transitions(False)
        {0: [1.0], 1: [0.963, 0.037], 2: [0.338, 0.396, 0.265], 3: [1.0], 4: [1.0]}
        """
        if isinstance(mat, dict) :
            for key, value in mat.items() :
                if key not in self._route_probs :
                    raise RuntimeError("One of the keys don't correspond to a vertex.")
                elif len(self.out_edges[key]) > 0 and not np.isclose(np.sum(value), 1) :
                    raise RuntimeError("Sum of transition probabilities at a vertex was not 1.")

                if len(value) == self.nV :
                    self._route_probs[key] = []
                    for e in self.g.vertex(key).out_edges() :
                        p = value[ int(e.target()) ]
                        self._route_probs[key].append( np.float64(p) )
                elif len(value) == len(self._route_probs[key]) :
                    self._route_probs[key] = []
                    for p in value :
                        self._route_probs[key].append( np.float64(p) )

        elif isinstance(mat, np.ndarray) :
            non_terminal = np.array([v.out_degree() > 0 for v in self.g.vertices()])
            if mat.shape != (self.nV, self.nV) :
                raise RuntimeError("Matrix is the wrong shape, should be %s x %s." % (self.nV, self.nV))
            elif not np.allclose(np.sum(mat[non_terminal,:], axis=1), 1) :
                raise RuntimeError("Sum of transition probabilities at a vertex was not 1.")

            for k in range(self.nV) :
                self._route_probs[k] = []
                for e in self.g.vertex(k).out_edges() :
                    p = mat[k, int(e.target())]
                    self._route_probs[k].append( np.float64(p) )


    def collect_data(self, queues=None, edge=None, eType=None) :
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
            a 2-tuple of the edge's source and target vertex indices, an
            iterable of 2-tuples of the edge's source and target vertex
            indices, an iterable of :class:`~graph_tool.Edge`\(s).
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edge is not None :
                if not isinstance(edge[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edge]
                elif isinstance(edge[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edge]
                else :
                    queues = [self.g.edge_index[self.g.edge(edge[0], edge[1])]]
            elif eType is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), eType) )[0]
            else :
                queues = range(self.nE)

        for k in queues :
            self.edge2queue[k].collect_data = True


    def stop_collecting_data(self, queues=None, edge=None, eType=None) :
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
            an iterable of 2-tuples of the edge's source and target vertex
            indices, an iterable of :class:`~graph_tool.Edge`\(s).
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will stop collecting data.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edge is not None :
                if not isinstance(edge[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edge]
                elif isinstance(edge[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edge]
                else :
                    queues = [self.g.edge_index[self.g.edge(edge[0], edge[1])]]
            elif eType is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), eType) )[0]
            else :
                queues = range(self.nE)

        for k in queues :
            self.edge2queue[k].collect_data = False


    def data_queues(self, queues=None, edge=None, eType=None) :
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
            an iterable of 2-tuples of the edge's source and target vertex
            indices, an iterable of :class:`~graph_tool.Edge`\(s).
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

        >>> g   = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.collect_data()
        >>> net.initialize(10)
        >>> net.simulate(2000)
        >>> data = net.data_queues(eType=(1,3))

        To get data from an edge connecting two vertices do the following:

        >>> data = net.data_queues(edge=(1,50))

        To get data from several edges do the following:

        >>> data = net.data_queues(edge=[(1,3), (10,91), (90,90)])

        You can specify the edge indices as well:

        >>> data = net.data_queues(queues=(20, 14, 0, 4))
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edge is not None :
                if not isinstance(edge[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edge]
                elif isinstance(edge[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edge]
                else :
                    queues = [self.g.edge_index[self.g.edge(edge[0], edge[1])]]
            elif eType is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), eType) )[0]
            else :
                queues = range(self.nE)

        data = np.zeros( (0,6) )
        for q in queues :
            dat = self.edge2queue[q].fetch_data()

            if len(dat) > 0 :
                data = np.vstack( (data, dat) )

        return data


    def data_agents(self, queues=None, edge=None, eType=None) :
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
            indices, an iterable of 2-tuples of the edge's source and target
            vertex indices, an iterable of :class:`~graph_tool.Edge`\(s).
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
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edge is not None :
                if not isinstance(edge[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edge]
                elif isinstance(edge[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edge]
                else :
                    queues = [self.g.edge_index[self.g.edge(edge[0], edge[1])]]
            elif eType is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), eType) )[0]
            else :
                queues = range(self.nE)

        data = {}
        for q in queues :
            for issn, dat in self.edge2queue[q].data.items() :
                datum = np.zeros( (len(dat), 6) )
                datum[:,:5] = np.array(dat)
                datum[:, 5] = q
                if issn in data :
                    data[issn] = np.vstack( (data[issn], datum) )
                else :
                    data[issn] = datum

        dType = [('a', float), ('s', float), ('d', float), ('q', float), ('n', float), ('id', float)]
        for issn, dat in data.items() :
            datum = np.array([tuple(d) for d in dat.tolist()], dtype=dType)
            datum = np.sort(datum, order='a')
            data[issn] = np.array([tuple(d) for d in datum])

        return data


    def draw(self, update_colors=True, **kwargs) :
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
        Also, the ``bg_color`` parameter is defined in the :class:`.dict`
        ``QueueNetwork.colors``. The ``output_size`` parameter defaults to
        ``(700, 700)``.

        If any of these parameters are supplied as arguments to ``draw`` then
        the passed arguments are used over the defaults.

        Examples
        --------
        To draw the current state of the network, call:

        >>> g   = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.draw()

        If you specify a file name and location, the drawing will be saved to
        disk. For example, to save the drawing to the current working directory
        do the following:

        >>> net.draw(output="current_state.png", output_size=(400,400))

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

        >>> net.draw(vertex_text=net.g.vertex_index)
        """
        if update_colors :
            self._update_all_colors()

        output_size = (700, 700)

        if 'output' not in kwargs :
            if 'geometry' not in kwargs :
                if 'output_size' in kwargs :
                    kwargs['geometry'] = kwargs['output_size']
                else :
                    kwargs['geometry'] = output_size
        else :
            if 'output_size' not in kwargs :
                if 'geometry' in kwargs :
                    kwargs['output_size'] = kwargs['geometry']
                else :
                    kwargs['output_size'] = output_size
            
        vertex_params = set(['vertex_color', 'vertex_fill_color', 'pos',
                             'vertex_size', 'vertex_pen_width'])

        edge_params   = set(['edge_color', 'edge_control_points',
                             'edge_marker_size', 'edge_pen_width'])

        for param in vertex_params :
            if param not in kwargs :
                kwargs[param] = self.g.vp[param]

        for param in edge_params :
            if param not in kwargs :
                kwargs[param] = self.g.ep[param]

        ans = gt.graph_draw(g=self.g, bg_color=self.colors['bg_color'], **kwargs)


    def show_active(self, **kwargs) :
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
        for v in self.g.vertices() :
            self.g.vp['vertex_color'][v] = [0, 0, 0, 0.9]
            is_active = False
            my_iter   = v.in_edges() if self.g.is_directed() else v.out_edges()
            for e in my_iter :
                ei = self.g.edge_index[e]
                if self.edge2queue[ei]._active :
                    is_active = True
                    break
            if is_active :
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_active']
            else :
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_inactive']

        for e in self.g.edges() :
            ei = self.g.edge_index[e]
            if self.edge2queue[ei]._active :
                self.g.ep['edge_color'][e] = self.colors['edge_active']
            else :
                self.g.ep['edge_color'][e] = self.colors['edge_inactive']

        self.draw(update_colors=False, **kwargs)
        self._update_all_colors()


    def show_type(self, n, **kwargs) :
        """Draws the network, highlighting queues of a certain type.

        The colored vertices represent self loops of type ``n``. Dark edges
        represent queues of type ``n``.

        Parameters
        ----------
        n : int
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

        >>> g   = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.show_type(2, output_size=(400,400), output='edge_type_2.png')

        .. figure:: edge_type_2.png
           :align: center
        """
        for v in self.g.vertices() :
            e   = self.g.edge(v, v)
            if isinstance(e, gt.Edge) and self.g.ep['eType'][e] == n :
                ei  = self.g.edge_index[e]
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_highlight']
                self.g.vp['vertex_color'][v]      = self.edge2queue[ei].colors['vertex_color']
            else :
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_inactive']
                self.g.vp['vertex_color'][v]      = [0, 0, 0, 0.9]

        for e in self.g.edges() :
            ei = self.g.edge_index[e]
            if self.g.ep['eType'][e] == n :
                self.g.ep['edge_color'][e] = self.colors['edge_active']
            else :
                self.g.ep['edge_color'][e] = self.colors['edge_inactive']

        self.draw(update_colors=False, **kwargs)
        self._update_all_colors()


    def _update_all_colors(self) :
        ep  = self.g.ep
        vp  = self.g.vp
        do  = [True for v in range(self.nV)]
        for q in self.edge2queue :
            e = self.g.edge(q.edge[0], q.edge[1])
            v = self.g.vertex(q.edge[1])
            if q.edge[0] == q.edge[1] :
                ep['edge_color'][e]        = q._current_color(1)
                vp['vertex_color'][v]      = q._current_color(2)
                if q.edge[3] == 0 :
                    nSy = 0
                    cap = 0
                    for vi in self.in_edges[q.edge[1]] :
                        nSy += self.edge2queue[vi].nSystem
                        cap += self.edge2queue[vi].nServers

                    tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

                    color    = [ i * tmp / 0.9 for i in q.colors['vertex_fill_color'] ]
                    color[3] = 1.0 - tmp
                    vp['vertex_fill_color'][v] = color
                else :
                    vp['vertex_fill_color'][v] = q._current_color()
            else :
                ep['edge_color'][e] = q._current_color()
                if do[q.edge[1]] :
                    nSy = 0
                    cap = 0
                    for vi in self.in_edges[q.edge[1]] :
                        nSy += self.edge2queue[vi].nSystem
                        cap += self.edge2queue[vi].nServers

                    tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

                    color    = [ i * tmp / 0.9 for i in q.colors['vertex_fill_color'] ]
                    color[3] = 1.0 - tmp
                    vp['vertex_fill_color'][v]  = color
                    vp['vertex_color'][v]       = self.colors['vertex_color']
                    do[q.edge[1]] = False


    def _update_graph_colors(self, ad, qedge) :
        e   = self.g.edge(qedge[0], qedge[1])
        v   = e.target()
        ep  = self.g.ep
        vp  = self.g.vp

        if self._prev_edge is not None :
            pe  = self.g.edge(self._prev_edge[0], self._prev_edge[1])
            pv  = self.g.vertex(self._prev_edge[1])
            q   = self.edge2queue[self._prev_edge[2]]

            if pe.target() == pe.source() :
                ep['edge_color'][pe]        = q._current_color(1)
                vp['vertex_color'][pv]      = q._current_color(2)
                vp['vertex_fill_color'][pv] = q._current_color()
            else :
                ep['edge_color'][pe] = q._current_color()
                nSy = 0
                cap = 0
                for vi in self.in_edges[self._prev_edge[1]] :
                    nSy += self.edge2queue[vi].nSystem
                    cap += self.edge2queue[vi].nServers

                tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

                color    = [ i * tmp / 0.9 for i in self.colors['vertex_fill_color'] ]
                color[3] = 1.0 - tmp
                vp['vertex_fill_color'][pv] = color
                vp['vertex_color'][pv]      = self.colors['vertex_color']

        q   = self.edge2queue[qedge[2]]
        if qedge[0] == qedge[1] :
            ep['edge_color'][e]         = q._current_color(1)
            vp['vertex_fill_color'][v]  = q._current_color()
            vp['vertex_color'][v]       = q._current_color(2)

        else :
            ep['edge_color'][e] = q._current_color()
            nSy = 0
            cap = 0
            for vi in self.in_edges[qedge[1]] :
                nSy += self.edge2queue[vi].nSystem
                cap += self.edge2queue[vi].nServers

            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            color    = [ i * tmp / 0.9 for i in self.colors['vertex_fill_color'] ]
            color[3] = 1.0 - tmp
            vp['vertex_fill_color'][v] = color
            vp['vertex_color'][v]  = self.colors['vertex_color']


    def _add_departure(self, ei, agent, t) :
        q   = self.edge2queue[ei]
        qt  = q._time
        q._add_departure(agent, t)

        if qt == infty and q._time < infty :
            self._queues.append(q)

        self._queues.sort(reverse=True)


    def _add_arrival(self, ei, agent, t=None) :
        q   = self.edge2queue[ei]
        qt  = q._time
        if t is None :
            t = q._time + 1 if q._time < infty else self._queues[-1]._time + 1

        agent._time = t
        q._add_arrival(agent)

        if qt == infty and q._time < infty :
            self._queues.append(q)

        self._queues.sort(reverse=True)


    def next_event_description(self) :
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


    def _simulate_next_event(self, slow=True) :
        n   = len(self._queues)
        if n == 0 :
            self.t  = infty
            return

        q1  = self._queues.pop()
        q1t = q1._time
        e1  = q1.edge[2]

        event  = q1.next_event_description()
        self.t = q1t
        self.nEvents += 1

        if event == 2 : # This is a departure
            e2  = q1._departures[0].desired_destination(self, q1.edge)
            q2  = self.edge2queue[e2]
            q2t = q2._time

            if q2.at_capacity() :
                q2.nBlocked += 1
                q1._departures[0].blocked += 1
                if self._blocking :
                    t = q2._departures[0]._time + EPS
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
                    self._update_graph_colors(ad='departure', qedge=q1.edge)
                    self._prev_edge = q1.edge

                if q2._active and np.sum(self.nAgents) > self.max_agents - 1 :
                    q2._active = False

                q2.next_event()
                self.nAgents[e2] = q2._nTotal

                if slow :
                    self._update_graph_colors(ad='arrival', qedge=q2.edge)
                    self._prev_edge = q2.edge

            if q1._time < infty :
                if q2._time < q2t < infty and e2 != e1 :
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
                if q2._time < q2t < infty :
                    if n > 2 :
                        oneSort(self._queues, q2t, n-1)
                elif q2._time < q2t :
                    if n == 1 :
                        self._queues.append(q2)
                    else :
                        bisectSort(self._queues, q2, n-1)

        elif event == 1 : # This is an arrival
            if q1._active and np.sum(self.nAgents) > self.max_agents - 1 :
                q1._active = False

            q1.next_event()
            self.nAgents[e1] = q1._nTotal

            if slow :
                self._update_graph_colors(ad='arrival', qedge=q1.edge)
                self._prev_edge  = q1.edge

            if q1._time < infty :
                if n == 1 :
                    self._queues.append(q1)
                else :
                    bisectSort(self._queues, q1, n-1 )

        if self._to_animate :
            self._window.graph.regenerate_surface(lazy=False)
            self._window.graph.queue_draw()

            if self._to_disk :
                pixbuf = self._window.get_pixbuf()
                pixbuf.savev(self._outdir+'%s.' % (self._count) + self._fmt, self._fmt, [], [])
                if self._count >= self._max_count :
                    Gtk.main_quit()
                self._count += 1

            return True


    def animate(self, out=None, count=10, **kwargs) :
        """Animates the network as it's simulating.

        The animations can be saved to disk or view in interactive mode.
        Closing the window ends the animation if viewed in interactive mode.

        Parameters
        ----------
        out : str (optional)
            The location where the frames for the images will be saved. If this
            parameter is not given, then the animation is shown in interactive
            mode.
        count : int (optional, the default is 10)
            Indicates the number of frames to save to disk. This parameter is
            only used if ``out_dir`` is passed.
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

        Raises
        ------
        RuntimeError
            Will raise a :exc:`~RuntimeError` if the ``QueueNetwork`` has not
            been initialized. Call :meth:`.initialize` before running.

        Examples
        --------
        This function works similarly to ``QueueNetwork``\'s :meth:`.draw`
        method. To animate the network in interactive mode do the following:

        >>> g   = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.animate(output_size=(400,400))

        To stop the animation just close the window. If you want to write the frames
        to disk run something like the following:

        >>> net.animate(out="./test", count=25, output_size=(400,400), vertex_size=15)

        The above code outputs the frames in the current working directory and
        outputs 25 ``png`` images whose names start with ``test`` e.g.
        ``test0.png``\, ``test1.png``\, ... etc. Also, the vertex size for each
        vertex was changed from the default (of 8) to 15.
        """
        if not self._initialized :
            raise RuntimeError("Network has not been initialized. Call 'initialize()' first.")

        output_size = (700, 700)

        if out is None :
            if 'geometry' not in kwargs :
                if 'output_size' in kwargs :
                    kwargs['geometry'] = kwargs['output_size']
                    del kwargs['output_size']
                else :
                    kwargs['geometry'] = output_size
        else :
            if 'geometry' in kwargs :
                output_size = kwargs['geometry']
                del kwargs['geometry']
            if 'output_size' in kwargs :
                output_size = kwargs['output_size']
                del kwargs['output_size']
            
        vertex_params = set(['vertex_color', 'vertex_fill_color', 'vertex_size',
                             'vertex_pen_width', 'pos'])

        edge_params   = set(['edge_color', 'edge_control_points',
                             'edge_marker_size', 'edge_pen_width'])

        for param in vertex_params :
            if param not in kwargs :
                kwargs[param] = self.g.vp[param]

        for param in edge_params :
            if param not in kwargs :
                kwargs[param] = self.g.ep[param]

        self._to_animate = True
        self._update_all_colors()

        if out is None :
            self._to_disk = False
            self._window  = gt.GraphWindow(g=self.g, bg_color=self.colors['bg_color'], **kwargs)
        else :
            self._fmt       = kwargs['fmt'] if 'fmt' in kwargs else 'png'
            self._count     = 0
            self._max_count = count
            self._to_disk   = True
            self._outdir    = out
            self._window    = Gtk.OffscreenWindow()
            self._window.set_default_size(output_size[0], output_size[1])
            self._window.graph = gt.GraphWidget(self.g, bg_color=self.colors['bg_color'], **kwargs)
            self._window.add(self._window.graph)

        cid = GObject.idle_add(self._simulate_next_event)
        self._window.connect("delete_event", Gtk.main_quit)
        self._window.show_all()
        Gtk.main()

        self._to_animate = False
        self._to_disk    = False


    def simulate(self, n=1, t=None) :
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
        RuntimeError
            Will raise a :exc:`~RuntimeError` if the ``QueueNetwork`` has not
            been initialized. Call :meth:`.initialize` before running.

        Examples
        --------
        Let ``net`` denote your instance of a ``QueueNetwork``. Before you
        simulate, you need to initialize the network, which allows arrivals
        from outside the network. To initialize with 2 (random chosen) edges
        accepting arrivals run:

        >>> g   = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.initialize(2)

        To simulate the network 50000 events run:

        >>> nE0 = net.nEvents
        >>> net.simulate(50000)
        >>> net.nEvents - nE0
        50000

        To simulate the network for at least 25 simulation time units run:

        >>> nE0 = net.nEvents
        >>> t0  = net.time
        >>> net.simulate(t=75)
        >>> t1  = net.time
        >>> round(t1 - t0, 3)
        75.005
        >>> net.nEvents - nE0
        21595
        """
        if not self._initialized :
            raise RuntimeError("Network has not been initialized. Call '.initialize()' first.")
        if t is None :
            for k in range(n) :
                self._simulate_next_event(slow=False)
        else :
            now = self.t
            while self.t < now + t :
                self._simulate_next_event(slow=False)



    def reset_colors(self) :
        """Resets all edge and vertex colors to their default values."""
        for k, e in enumerate(self.g.edges()) :
            self.g.ep['edge_color'][e]    = self.edge2queue[k].colors['edge_color']
        for k, v in enumerate(self.g.vertices()) :
            self.g.vp['vertex_fill_color'][v] = self.colors['vertex_fill_color']


    def clear(self) :
        """Resets the queue to its initial state.

        The attributes ``t``, ``nEvents``, ``nAgents`` are set to zero
        :meth:`.reset_colors` is called; and the :class:`.QueueServer` method
        :meth:`.clear` is called for each queue in the network.

        Notes
        -----
        ``QueueNetwork`` must be re-initialized before any simulations can run.
        """
        self.t            = 0
        self.nEvents      = 0
        self.nAgents      = np.zeros(self.nE, int)
        self._queues      = []
        self._to_animate  = False
        self._prev_edge   = None
        self._initialized = False
        self.reset_colors()
        for q in self.edge2queue :
            q.clear()


    def clear_data(self, queues=None, edge=None, eType=None) :
        """Clears data from queues.

        If none of the parameters are given then every queue's data is cleared.

        Parameters
        ----------
        queues : int or an iterable of int (optional)
            The edge index (or an iterable of edge indices) identifying the
            :class:`.QueueServer`\(s) whose data will be cleared.
        edge : 2-:class:`.tuple` of int or *array_like* (optional)
            Explicitly specify which queues' data to clear. Must be either: a
            2-tuple of the edge's source and target vertex indices, an iterable
            of 2-tuples of the edge's source and target vertex indices, an 
            iterable of :class:`~graph_tool.Edge`\(s).
        eType : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which edge types
            will have their data cleared.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edge is not None :
                if not isinstance(edge[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edge]
                elif isinstance(edge[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edge]
                else :
                    queues = [self.g.edge_index[self.g.edge(edge[0], edge[1])]]
            elif eType is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), eType) )[0]
            else :
                queues = range(self.nE)

        for k in queues :
            self.edge2queue[k].data = {}


    def copy(self) :
        """Returns a deep copy of self."""
        net                 = QueueNetwork(None)
        net.g               = self.g.copy()
        net.t               = copy.copy(self.t)
        net.max_agents      = copy.copy(self.max_agents)
        net.nV              = copy.copy(self.nV)
        net.nE              = copy.copy(self.nE)
        net.nAgents         = copy.copy(self.nAgents)
        net.nEvents         = copy.copy(self.nEvents)
        net.shortest_path   = copy.copy(self.shortest_path)
        net._initialized    = copy.copy(self._initialized)
        net._prev_edge      = copy.copy(self._prev_edge)
        net._to_animate     = copy.copy(self._to_animate)
        net._blocking       = copy.copy(self._blocking)
        net.colors          = copy.deepcopy(self.colors)
        net.out_edges       = copy.deepcopy(self.out_edges)
        net.in_edges        = copy.deepcopy(self.in_edges)
        net.edge2queue      = copy.deepcopy(self.edge2queue)

        if net._initialized :
            net.queues = [q for q in net.edge2queue]
            net.queues.sort()
            while net.queues[-1]._time == infty :
                net.queues.pop()

            net.queues.sort(reverse=True)
        return net
