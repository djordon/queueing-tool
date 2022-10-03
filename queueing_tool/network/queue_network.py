import collections
import numbers
import copy
import array

import numpy as np
from numpy.random import uniform

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    plt.style.use('ggplot')
    HAS_MATPLOTLIB = True

except ImportError:
    HAS_MATPLOTLIB = False

from queueing_tool.graph import _prepare_graph
from queueing_tool.queues import (
    NullQueue,
    QueueServer,
    LossQueue
)
from queueing_tool.network.priority_queue import PriorityQueue


class QueueingToolError(Exception):
    """Base class for exceptions in Queueing-tool."""
    pass


EPS = np.float64(1e-7)

v_pens = [
    [0.5, 0.5, 0.5, 0.5],
    [0, 0.5, 1, 1],
    [0.133, 0.545, 0.133, 1],
    [0.282, 0.239, 0.545, 1],
    [1, 0.135, 0, 1]
]


class QueueNetwork(object):
    """A class that simulates a network of queues.

    Takes a networkx :any:`DiGraph<networkx.DiGraph>` and places queues
    on each edge of the graph. The simulations are event based, and
    this class handles the scheduling of events.

    Each edge on the graph has a *type*, and this *type* is used to
    specify the subclass of :class:`.QueueServer` and arguments used
    when creating the queue that sits on each edge.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :class:`numpy.ndarray`, dict, \
        ``None``,  etc.
        Any object that networkx can turn into a
        :any:`DiGraph<networkx.DiGraph>`. The graph specifies the
        network, and the queues sit on top of the edges.
    q_classes : dict (optional)
        Used to specify the :class:`.QueueServer` class for each edge
        type. The keys are integers for the edge types, and the values
        are classes.
    q_args : dict (optional)
        Used to specify the class arguments for each type of
        :class:`.QueueServer`. The keys are integers for the edge types
        and the values are dictionarys holding the arguments that are
        passed when instantiating each :class:`.QueueServer` created
        with that edge type.
    seed : int (optional)
        An integer used to initialize numpy's psuedo-random number
        generator.
    colors : dict (optional)
        A dictionary of RGBA colors used to color the graph. The keys
        are specified in the Notes section. If this parameter is
        supplied but a particular key is missing, then the default
        value for that key is used.
    max_agents : int (optional, default: 1000)
        The maximum number of agents that can be in the network at any
        time.
    blocking : ``{'BAS', 'RS'}`` (optional, default: ``'BAS'``)
        Specifies the blocking behavior for the system. If `blocking
        is not ``'RS'``, then it is assumed to be ``'BAS'``.

        ``'BAS'``
            *Blocking After Service*: when an agent attempts to enter a
            :class:`.LossQueue` that is at capacity the agent is forced
            to wait at his current queue until an agent departs from
            the target queue.
        ``'RS'``
            *Repetitive Service Blocking*: when an agent attempts to
            enter a :class:`.LossQueue` that is at capacity, the agent
            is forced to receive another service from the queue it is
            departing from. After the agent receives the service, she
            then checks to see if the desired queue is still at
            capacity, and if it is this process is repeated, otherwise
            she enters the queue.
    adjust_graph : bool (optional, default: ``True``)
        Specifies whether the graph will be adjusted to make sure
        terminal nodes do not cause any issues when simulating. For
        most cases, this should be set to ``True``.

    Attributes
    ----------
    blocking : str
        Specifies whether the system's blocking behavior is either
        Blocking After Service (BAS) or Repetitive Service Blocking (RS).
    colors : dict
        A dictionary of colors used when drawing a graph. See the notes
        for the defaults.
    current_time : float
        The time of the last event.
    edge2queue : list
        A list of queues where the ``edge2queue[k]`` returns the queue
        on the edge with edge index ``k``.
    g : :class:`.QueueNetworkDiGraph`
        The graph for the network.
    default_classes : dict
        Specifies the default queue classes for each edge type.
    default_colors : dict
        Specifies various default colors.
    default_q_colors : dict
        Specifies the default colors used by the queues.
    in_edges : list
        A mapping between vertex indices and the in-edges at that
        vertex. Specifically, ``in_edges[v]`` returns a list containing
        the edge index for all edges with the head of the edge at
        ``v``, where ``v`` is the vertex's index number.
    max_agents : int
        The maximum number of agents that can be in the network at any
        time.
    num_agents : :class:`~numpy.ndarray`
        A one-dimensional array where the ``k``-th entry corresponds to
        the total number of agents in the :class:`.QueueServer` with
        edge index ``k``. This number includes agents that are
        scheduled to arrive at the queue at some future time but
        haven't yet.
    num_edges : int
        The number of edges in the graph.
    num_events : int
        The number of events that have occurred thus far. Every arrival
        from outside the network counts as one event, but the departure
        of an agent from a queue and the arrival of that same agent to
        another queue counts as one event.
    num_vertices : int
        The number of vertices in the graph.
    num_nodes : int
        The number of vertices in the graph.
    out_edges : list
        A mapping between vertex indices and the out-edges at that
        vertex. Specifically, ``out_edges[v]`` returns a list
        containing the edge index for all edges with the tail of the
        edge at ``v``, where ``v`` is the vertex's index number.
    time : float
        The time of the next event.


    Raises
    ------
    TypeError
        Raised when the parameter ``g`` is not of a type that can be
        made into a :any:`networkx.DiGraph`, or when ``g`` is not
        ``None``.

    Notes
    -----
    * If only :class:`Agents<.Agent>` enter the network, then the
      ``QueueNetwork`` instance is a `Jackson network`_. The default
      transition probabilities at any vertex ``v`` is
      ``1 / g.out_degree(v)`` for each adjacent vertex.
    * This class must be initialized before any simulations can take
      place. To initialize, call the :meth:`~initialize` method.
    * When simulating the network, the departure of an agent from one
      queue coincides with their arrival to another. There is no time
      lag between these events.
    * When defining your ``q_classes`` you should not assign queues
      with edge type ``0`` to anything other than the
      :class:`.NullQueue` class.  Edges with edge type ``0`` are
      treated by ``QueueNetwork`` as terminal edges (edges that point
      to a terminal vertex).
    * If an edge type is used in your network but not given in
      ``q_classes`` parameter then the defaults are used, which are:

      >>> default_classes = { # doctest: +SKIP
      ...     0: qt.NullQueue,
      ...     1: qt.QueueServer,
      ...     2: qt.LossQueue,
      ...     3: qt.LossQueue,
      ...     4: qt.LossQueue
      ... }

      For example, if your network has type ``0``, ``1``, and ``2``
      edges but your ``q_classes`` parameter looks like:

      >>> my_classes = {1 : qt.ResourceQueue} # doctest: +SKIP

      then each type ``0`` and type ``2`` edge is a :class:`.NullQueue`
      and :class:`.LossQueue` respectively.
    * The following properties are assigned as a node or edge attribute
      to the graph; their default values for each edge or node is shown:

        * ``vertex_pen_width``: ``1.1``,
        * ``vertex_size``: ``8``,
        * ``edge_control_points``: ``[]``
        * ``edge_marker_size``: ``8``
        * ``edge_pen_width``: ``1.25``

      There are also property maps created for graph visualization,
      they are ``vertex_color``, ``vertex_fill_color``, ``pos``, and
      ``edge_color``. The default colors, which are used by various
      methods, are:

      >>> default_colors = {
      ...     'vertex_fill_color': [0.9, 0.9, 0.9, 1.0],
      ...     'vertex_color': [0.0, 0.5, 1.0, 1.0],
      ...     'vertex_highlight': [0.5, 0.5, 0.5, 1.0],
      ...     'edge_departure': [0, 0, 0, 1],
      ...     'vertex_active': [0.1, 1.0, 0.5, 1.0],
      ...     'vertex_inactive': [0.9, 0.9, 0.9, 0.8],
      ...     'edge_active': [0.1, 0.1, 0.1, 1.0],
      ...     'edge_inactive': [0.8, 0.8, 0.8, 0.3],
      ...     'bgcolor': [1, 1, 1, 1]
      ... }

    .. _Jackson network: http://en.wikipedia.org/wiki/Jackson_network

    Examples
    --------
    The following creates a queueing network with the Moebius-Kantor
    graph. Each queue has 5 servers, and the same arrival and service
    distributions:

    >>> import queueing_tool as qt
    >>> import networkx as nx
    >>> import numpy as np
    >>>
    >>> g = nx.moebius_kantor_graph()
    >>> q_cl = {1: qt.QueueServer}
    >>> def arr(t):
    ...     return t + np.random.gamma(4, 0.0025)
    >>> def ser(t):
    ...     return t + np.random.exponential(0.025)
    >>> q_ar = {
    ...     1: {
    ...         'arrival_f': arr,
    ...         'service_f': ser,
    ...         'num_servers': 5
    ...     }
    ... }
    >>> net = qt.QueueNetwork(g, q_classes=q_cl, q_args=q_ar, seed=13)

    To specify that arrivals enter from type 1 edges and simulate run:

    >>> net.initialize(edge_type=1)
    >>> net.simulate(n=100)

    Now we'd like to see how many agents are in type 1 edges:

    >>> nA = [(q.num_system, q.edge[2]) for q in net.edge2queue if q.edge[3] == 1]
    >>> nA.sort(reverse=True)
    >>> nA[:5]            # doctest: +SKIP
    [(4, 37), (4, 34), (3, 43), (3, 32), (3, 30)]

    To view the state of the network do the following (note, you need
    to have pygraphviz installed and your graph may be rotated):

    >>> net.simulate(n=500)
    >>> pos = nx.nx_agraph.graphviz_layout(g.to_undirected(), prog='neato') # doctest: +SKIP
    >>> net.draw(pos=pos) # doctest: +SKIP
    <...>

    .. figure:: my_network1.png
       :align: center
    """

    default_colors = {
        'vertex_fill_color': [0.95, 0.95, 0.95, 1.0],
        'vertex_color': [0.0, 0.5, 1.0, 1.0],
        'vertex_highlight': [0.5, 0.5, 0.5, 1.0],
        'edge_departure': [0, 0, 0, 1],
        'vertex_active': [0.1, 1.0, 0.5, 1.0],
        'vertex_inactive': [0.95, 0.95, 0.95, 1.0],
        'edge_active': [0.1, 0.1, 0.1, 1.0],
        'edge_inactive': [0.8, 0.8, 0.8, 0.3],
        'bgcolor': [1, 1, 1, 1]
    }

    default_classes = {
        0: NullQueue,
        1: QueueServer,
        2: LossQueue,
        3: LossQueue,
        4: LossQueue
    }

    default_q_colors = {
        k: {'edge_loop_color': [0, 0, 0, 0],
            'edge_color': [0.8, 0.8, 0.8, 1.0],
            'vertex_fill_color': [0.95, 0.95, 0.95, 1.0],
            'vertex_color': v_pens[k]}
        for k in range(5)
    }

    def __init__(self, g, q_classes=None, q_args=None, seed=None, colors=None,
                 max_agents=1000, blocking='BAS', adjust_graph=True):

        if not isinstance(blocking, str):
            raise TypeError("blocking must be a string")

        self._t = 0
        self.num_events = 0
        self.max_agents = max_agents

        self._initialized = False
        self._prev_edge = None
        self._fancy_heap = PriorityQueue()
        self._blocking = True if blocking.lower() != 'rs' else False

        if colors is None:
            colors = {}

        for key, value in self.default_colors.items():
            if key not in colors:
                colors[key] = value

        self.colors = colors

        if q_classes is None:
            q_classes = self.default_classes
        else:
            for k in set(self.default_classes.keys()) - set(q_classes.keys()):
                q_classes[k] = self.default_classes[k]

        if q_args is None:
            q_args = {k: {} for k in range(5)}
        else:
            for k in set(q_classes.keys()) - set(q_args.keys()):
                q_args[k] = {}

        for key, args in q_args.items():
            if 'colors' not in args:
                args['colors'] = self.default_q_colors.get(key, self.default_q_colors[1])

        if isinstance(seed, numbers.Integral):
            np.random.seed(seed)

        if g is not None:
            g, qs = _prepare_graph(g, self.colors, q_classes, q_args, adjust_graph)

            self.nV = g.number_of_nodes()
            self.nE = g.number_of_edges()

            self.edge2queue = qs
            self.num_agents = np.zeros(g.number_of_edges(), int)
            self.out_edges = [0 for v in range(self.nV)]
            self.in_edges = [0 for v in range(self.nV)]
            self._route_probs = [0 for v in range(self.nV)]

            for v in g.nodes():
                vod = g.out_degree(v)
                probs = array.array('d', [1. / vod for i in range(vod)])
                self.out_edges[v] = [g.edge_index[e] for e in sorted(g.out_edges(v))]
                self.in_edges[v] = [g.edge_index[e] for e in sorted(g.in_edges(v))]
                self._route_probs[v] = probs

            g.freeze()
            self.g = g

    def __repr__(self):
        the_string = 'QueueNetwork. # nodes: {0}, edges: {1}, agents: {2}'
        return the_string.format(self.nV, self.nE, np.sum(self.num_agents))

    @property
    def blocking(self):
        return 'BAS' if self._blocking else 'RS'

    @blocking.setter
    def blocking(self, tmp):
        if not isinstance(tmp, str):
            raise TypeError("blocking must be a string")
        self._blocking = True if tmp.lower() != 'rs' else False

    @property
    def num_vertices(self):
        return self.nV

    @property
    def num_nodes(self):
        return self.nV

    @property
    def num_edges(self):
        return self.nE

    @property
    def current_time(self):
        return self._t

    @property
    def time(self):
        if self._fancy_heap.size > 0:
            e = self._fancy_heap.array_edges[0]
            t = self.edge2queue[e]._time
        else:
            t = np.infty
        return t

    def animate(self, out=None, t=None, line_kwargs=None,
                scatter_kwargs=None, **kwargs):
        """Animates the network as it's simulating.

        The animations can be saved to disk or viewed in interactive
        mode. Closing the window ends the animation if viewed in
        interactive mode. This method calls
        :meth:`~matplotlib.axes.scatter`, and
        :class:`~matplotlib.collections.LineCollection`, and any
        keyword arguments they accept can be passed to them.

        Parameters
        ----------
        out : str (optional)
            The location where the frames for the images will be saved.
            If this parameter is not given, then the animation is shown
            in interactive mode.
        t : float (optional)
            The amount of simulation time to simulate forward. If
            given, and ``out`` is given, ``t`` is used instead of
            ``n``.
        line_kwargs : dict (optional, default: None)
            Any keyword arguments accepted by
            :class:`~matplotlib.collections.LineCollection`.
        scatter_kwargs : dict (optional, default: None)
            Any keyword arguments accepted by
            :meth:`~matplotlib.axes.Axes.scatter`.
        bgcolor : list (optional, keyword only)
            A list with 4 floats representing a RGBA color. The
            default is defined in ``self.colors['bgcolor']``.
        figsize : tuple (optional, keyword only, default: ``(7, 7)``)
            The width and height of the figure in inches.
        **kwargs :
            This method calls
            :class:`~matplotlib.animation.FuncAnimation` and
            optionally :meth:`.matplotlib.animation.FuncAnimation.save`.
            Any keyword that can be passed to these functions are
            passed via ``kwargs``.

        Notes
        -----
        There are several parameters automatically set and passed to
        matplotlib's :meth:`~matplotlib.axes.Axes.scatter`,
        :class:`~matplotlib.collections.LineCollection`, and
        :class:`~matplotlib.animation.FuncAnimation` by default.
        These include:

            * :class:`~matplotlib.animation.FuncAnimation`: Uses the
              defaults for that function. Saving the animation is done
              by passing the 'filename' keyword argument to this method.
              This method also accepts any keyword arguments accepted
              by :meth:`~matplotlib.animation.FuncAnimation.save`.
            * :class:`~matplotlib.collections.LineCollection`: The default
              arguments are taken from
              :meth:`.QueueNetworkDiGraph.lines_scatter_args`.
            * :meth:`~matplotlib.axes.Axes.scatter`: The default
              arguments are taken from
              :meth:`.QueueNetworkDiGraph.lines_scatter_args`.

        Raises
        ------
        QueueingToolError
            Will raise a :exc:`.QueueingToolError` if the
            ``QueueNetwork`` has not been initialized. Call
            :meth:`.initialize` before running.

        Examples
        --------
        This function works similarly to ``QueueNetwork's``
        :meth:`.draw` method.

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.initialize()
        >>> net.animate(figsize=(4, 4)) # doctest: +SKIP

        To stop the animation just close the window. If you want to
        write the animation to disk run something like the following:

        >>> kwargs = {
        ...     'filename': 'test.mp4',
        ...     'frames': 300,
        ...     'fps': 30,
        ...     'writer': 'mencoder',
        ...     'figsize': (4, 4),
        ...     'vertex_size': 15
        ... }
        >>> net.animate(**kwargs) # doctest: +SKIP
        """

        if not self._initialized:
            msg = ("Network has not been initialized. "
                   "Call '.initialize()' first.")
            raise QueueingToolError(msg)

        if not HAS_MATPLOTLIB:
            msg = "Matplotlib is necessary to animate a simulation."
            raise ImportError(msg)

        self._update_all_colors()
        kwargs.setdefault('bgcolor', self.colors['bgcolor'])

        fig = plt.figure(figsize=kwargs.get('figsize', (7, 7)))
        ax = fig.gca()

        mpl_kwargs = {
            'line_kwargs': line_kwargs,
            'scatter_kwargs': scatter_kwargs,
            'pos': kwargs.get('pos')
        }

        line_args, scat_args = self.g.lines_scatter_args(**mpl_kwargs)

        lines = LineCollection(**line_args)
        lines = ax.add_collection(lines)
        scatt = ax.scatter(**scat_args)

        t = np.infty if t is None else t
        now = self._t

        def update(frame_number):
            if t is not None:
                if self._t > now + t:
                    return False
            self._simulate_next_event(slow=True)
            lines.set_color(line_args['colors'])
            scatt.set_edgecolors(scat_args['edgecolors'])
            scatt.set_facecolor(scat_args['c'])

        if hasattr(ax, 'set_facecolor'):
            ax.set_facecolor(kwargs['bgcolor'])
        else:
            ax.set_axis_bgcolor(kwargs['bgcolor'])

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        animation_args = {
            'fargs': None,
            'event_source': None,
            'init_func': None,
            'frames': None,
            'blit': False,
            'interval': 10,
            'repeat': None,
            'func': update,
            'repeat_delay': None,
            'fig': fig,
            'save_count': None,
        }
        for key, value in kwargs.items():
            if key in animation_args:
                animation_args[key] = value

        animation = FuncAnimation(**animation_args)
        if 'filename' not in kwargs:
            plt.ioff()
            plt.show()
        else:
            save_args = {
                'filename': None,
                'writer': None,
                'fps': None,
                'dpi': None,
                'codec': None,
                'bitrate': None,
                'extra_args': None,
                'metadata': None,
                'extra_anim': None,
                'savefig_kwargs': None
            }
            for key, value in kwargs.items():
                if key in save_args:
                    save_args[key] = value

            animation.save(**save_args)

    def clear(self):
        """Resets the queue to its initial state.

        The attributes ``t``, ``num_events``, ``num_agents`` are set to
        zero, :meth:`.reset_colors` is called, and the
        :meth:`.QueueServer.clear` method is called for each queue in
        the network.

        Notes
        -----
        ``QueueNetwork`` must be re-initialized before any simulations
        can run.
        """
        self._t = 0
        self.num_events = 0
        self.num_agents = np.zeros(self.nE, int)
        self._fancy_heap = PriorityQueue()
        self._prev_edge = None
        self._initialized = False
        self.reset_colors()
        for q in self.edge2queue:
            q.clear()

    def clear_data(self, queues=None, edge=None, edge_type=None):
        """Clears data from all queues.

        If none of the parameters are given then every queue's data is
        cleared.

        Parameters
        ----------
        queues : int or an iterable of int (optional)
            The edge index (or an iterable of edge indices) identifying
            the :class:`QueueServer(s)<.QueueServer>` whose data will
            be cleared.
        edge : 2-tuple of int or *array_like* (optional)
            Explicitly specify which queues' data to clear. Must be
            either:

            * A 2-tuple of the edge's source and target vertex
              indices, or
            * An iterable of 2-tuples of the edge's source and
              target vertex indices.

        edge_type : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which
            edge types will have their data cleared.
        """
        queues = _get_queues(self.g, queues, edge, edge_type)

        for k in queues:
            self.edge2queue[k].data = {}

    def copy(self):
        """Returns a deep copy of itself."""
        net = QueueNetwork(None)
        net.g = self.g.copy()
        net.max_agents = copy.deepcopy(self.max_agents)
        net.nV = copy.deepcopy(self.nV)
        net.nE = copy.deepcopy(self.nE)
        net.num_agents = copy.deepcopy(self.num_agents)
        net.num_events = copy.deepcopy(self.num_events)
        net._t = copy.deepcopy(self._t)
        net._initialized = copy.deepcopy(self._initialized)
        net._prev_edge = copy.deepcopy(self._prev_edge)
        net._blocking = copy.deepcopy(self._blocking)
        net.colors = copy.deepcopy(self.colors)
        net.out_edges = copy.deepcopy(self.out_edges)
        net.in_edges = copy.deepcopy(self.in_edges)
        net.edge2queue = copy.deepcopy(self.edge2queue)
        net._route_probs = copy.deepcopy(self._route_probs)

        if net._initialized:
            keys = [q._key() for q in net.edge2queue if q._time < np.infty]
            net._fancy_heap = PriorityQueue(keys, net.nE)

        return net

    def draw(self, update_colors=True, line_kwargs=None,
             scatter_kwargs=None, **kwargs):
        """Draws the network. The coloring of the network corresponds
        to the number of agents at each queue.

        Parameters
        ----------
        update_colors : ``bool`` (optional, default: ``True``).
            Specifies whether all the colors are updated.
        line_kwargs : dict (optional, default: None)
            Any keyword arguments accepted by
            :class:`~matplotlib.collections.LineCollection`
        scatter_kwargs : dict (optional, default: None)
            Any keyword arguments accepted by
            :meth:`~matplotlib.axes.Axes.scatter`.
        bgcolor : list (optional, keyword only)
            A list with 4 floats representing a RGBA color. The
            default is defined in ``self.colors['bgcolor']``.
        figsize : tuple (optional, keyword only, default: ``(7, 7)``)
            The width and height of the canvas in inches.
        **kwargs
            Any parameters to pass to
            :meth:`.QueueNetworkDiGraph.draw_graph`.

        Notes
        -----
        This method relies heavily on
        :meth:`.QueueNetworkDiGraph.draw_graph`. Also, there is a
        parameter that sets the background color of the canvas, which
        is the ``bgcolor`` parameter.

        Examples
        --------
        To draw the current state of the network, call:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.initialize(100)
        >>> net.simulate(1200)
        >>> net.draw() # doctest: +SKIP

        If you specify a file name and location, the drawing will be
        saved to disk. For example, to save the drawing to the current
        working directory do the following:

        >>> net.draw(fname="state.png", scatter_kwargs={'s': 40}) # doctest: +SKIP

        .. figure:: current_state1.png
            :align: center

        The shade of each edge depicts how many agents are located at
        the corresponding queue. The shade of each vertex is determined
        by the total number of inbound agents. Although loops are not
        visible by default, the vertex that corresponds to a loop shows
        how many agents are in that loop.

        There are several additional parameters that can be passed --
        all :meth:`.QueueNetworkDiGraph.draw_graph` parameters are
        valid. For example, to show the edges as dashed lines do the
        following.

        >>> net.draw(line_kwargs={'linestyle': 'dashed'}) # doctest: +SKIP
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is necessary to draw the network.")

        if update_colors:
            self._update_all_colors()

        if 'bgcolor' not in kwargs:
            kwargs['bgcolor'] = self.colors['bgcolor']

        self.g.draw_graph(line_kwargs=line_kwargs,
                          scatter_kwargs=scatter_kwargs, **kwargs)

    def get_agent_data(self, queues=None, edge=None, edge_type=None, return_header=False):
        """Gets data from queues and organizes it by agent.

        If none of the parameters are given then data from every
        :class:`.QueueServer` is retrieved.

        Parameters
        ----------
        queues : int or *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying
            the :class:`QueueServer(s)<.QueueServer>` whose data will
            be retrieved.
        edge : 2-tuple of int or *array_like* (optional)
            Explicitly specify which queues to retrieve agent data
            from. Must be either:

            * A 2-tuple of the edge's source and target vertex
              indices, or
            * An iterable of 2-tuples of the edge's source and
              target vertex indices.

        edge_type : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which
            edge types to retrieve agent data from.
        return_header : bool (optonal, default: False)
            Determines whether the column headers are returned.

        Returns
        -------
        dict
            Returns a ``dict`` where the keys are the
            :class:`Agent's<.Agent>` ``agent_id`` and the values are
            :class:`ndarrays<~numpy.ndarray>` for that
            :class:`Agent's<.Agent>` data. The columns of this array
            are as follows:

            * First: The arrival time of an agent.
            * Second: The service start time of an agent.
            * Third: The departure time of an agent.
            * Fourth: The length of the queue upon the agents arrival.
            * Fifth: The total number of :class:`Agents<.Agent>` in the
              :class:`.QueueServer`.
            * Sixth: the :class:`QueueServer's<.QueueServer>` id
              (its edge index).

        headers : str (optional)
            A comma seperated string of the column headers. Returns
            ``'arrival,service,departure,num_queued,num_total,q_id'``
        """
        queues = _get_queues(self.g, queues, edge, edge_type)

        data = {}
        for qid in queues:
            for agent_id, dat in self.edge2queue[qid].data.items():
                datum = np.zeros((len(dat), 6))
                datum[:, :5] = np.array(dat)
                datum[:, 5] = qid
                if agent_id in data:
                    data[agent_id] = np.vstack((data[agent_id], datum))
                else:
                    data[agent_id] = datum

        dType = [
            ('a', float),
            ('s', float),
            ('d', float),
            ('q', float),
            ('n', float),
            ('id', float)
        ]
        for agent_id, dat in data.items():
            datum = np.array([tuple(d) for d in dat.tolist()], dtype=dType)
            datum = np.sort(datum, order='a')
            data[agent_id] = np.array([tuple(d) for d in datum])

        if return_header:
            return data, 'arrival,service,departure,num_queued,num_total,q_id'

        return data

    def get_queue_data(self, queues=None, edge=None, edge_type=None, return_header=False):
        """Gets data from all the queues.

        If none of the parameters are given then data from every
        :class:`.QueueServer` is retrieved.

        Parameters
        ----------
        queues : int or an *array_like* of int, (optional)
            The edge index (or an iterable of edge indices) identifying
            the :class:`QueueServer(s)<.QueueServer>` whose data will
            be retrieved.
        edge : 2-tuple of int or *array_like* (optional)
            Explicitly specify which queues to retrieve data from. Must
            be either:

            * A 2-tuple of the edge's source and target vertex
              indices, or
            * An iterable of 2-tuples of the edge's source and
              target vertex indices.

        edge_type : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which
            edge types to retrieve data from.
        return_header : bool (optonal, default: False)
            Determines whether the column headers are returned.

        Returns
        -------
        out : :class:`~numpy.ndarray`
            * 1st: The arrival time of an agent.
            * 2nd: The service start time of an agent.
            * 3rd: The departure time of an agent.
            * 4th: The length of the queue upon the agents arrival.
            * 5th: The total number of :class:`Agents<.Agent>` in the
              :class:`.QueueServer`.
            * 6th: The :class:`QueueServer's<.QueueServer>` edge index.

        out : str (optional)
            A comma seperated string of the column headers. Returns
            ``'arrival,service,departure,num_queued,num_total,q_id'```

        Examples
        --------
        Data is not collected by default. Before simulating, by sure to
        turn it on (as well as initialize the network). The following
        returns data from queues with ``edge_type`` 1 or 3:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> net.start_collecting_data()
        >>> net.initialize(10)
        >>> net.simulate(2000)
        >>> data = net.get_queue_data(edge_type=(1, 3))

        To get data from an edge connecting two vertices do the
        following:

        >>> data = net.get_queue_data(edge=(1, 50))

        To get data from several edges do the following:

        >>> data = net.get_queue_data(edge=[(1, 50), (10, 91), (99, 99)])

        You can specify the edge indices as well:

        >>> data = net.get_queue_data(queues=(20, 14, 0, 4))
        """
        queues = _get_queues(self.g, queues, edge, edge_type)

        data = np.zeros((0, 6))
        for q in queues:
            dat = self.edge2queue[q].fetch_data()

            if len(dat) > 0:
                data = np.vstack((data, dat))

        if return_header:
            return data, 'arrival,service,departure,num_queued,num_total,q_id'

        return data

    def initialize(self, nActive=1, queues=None, edges=None, edge_type=None):
        """Prepares the ``QueueNetwork`` for simulation.

        Each :class:`.QueueServer` in the network starts inactive,
        which means they do not accept arrivals from outside the
        network, and they have no agents in their system. This method
        sets queues to active, which then allows agents to arrive from
        outside the network.

        Parameters
        ----------
        nActive : int (optional, default: ``1``)
            The number of queues to set as active. The queues are
            selected randomly.
        queues : int *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying
            the :class:`QueueServer(s)<.QueueServer>` to make active by.
        edges : 2-tuple of int or *array_like* (optional)
            Explicitly specify which queues to make active. Must be
            either:

            * A 2-tuple of the edge's source and target vertex
              indices, or
            * An iterable of 2-tuples of the edge's source and
              target vertex indices.

        edge_type : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which
            edge types will be set active.

        Raises
        ------
        ValueError
            If ``queues``, ``egdes``, and ``edge_type`` are all ``None``
            and ``nActive`` is an integer less than 1
            :exc:`~ValueError` is raised.
        TypeError
            If ``queues``, ``egdes``, and ``edge_type`` are all ``None``
            and ``nActive`` is not an integer then a :exc:`~TypeError`
            is raised.
        QueueingToolError
            Raised if all the queues specified are
            :class:`NullQueues<.NullQueue>`.

        Notes
        -----
        :class:`NullQueues<.NullQueue>` cannot be activated, and are
        sifted out if they are specified. More specifically, every edge
        with edge type 0 is sifted out.
        """
        if queues is None and edges is None and edge_type is None:
            if nActive >= 1 and isinstance(nActive, numbers.Integral):
                qs = [q.edge[2] for q in self.edge2queue if q.edge[3] != 0]
                n = min(nActive, len(qs))
                queues = np.random.choice(qs, size=n, replace=False)
            elif not isinstance(nActive, numbers.Integral):
                msg = "If queues is None, then nActive must be an integer."
                raise TypeError(msg)
            else:
                msg = ("If queues is None, then nActive must be a "
                       "positive int.")
                raise ValueError(msg)
        else:
            queues = _get_queues(self.g, queues, edges, edge_type)

        queues = [e for e in queues if self.edge2queue[e].edge[3] != 0]

        if len(queues) == 0:
            raise QueueingToolError("There were no queues to initialize.")

        if len(queues) > self.max_agents:
            queues = queues[:self.max_agents]

        for ei in queues:
            self.edge2queue[ei].set_active()
            self.num_agents[ei] = self.edge2queue[ei]._num_total

        keys = [q._key() for q in self.edge2queue if q._time < np.infty]
        self._fancy_heap = PriorityQueue(keys, self.nE)
        self._initialized = True

    def next_event_description(self):
        """Returns whether the next event is an arrival or a departure
        and the queue the event is accuring at.

        Returns
        -------
        des : str
            Indicates whether the next event is an arrival, a
            departure, or nothing; returns ``'Arrival'``,
            ``'Departure'``, or ``'Nothing'``.
        edge : int or ``None``
            The edge index of the edge that this event will occur at.
            If there are no events then ``None`` is returned.
        """
        if self._fancy_heap.size == 0:
            event_type = 'Nothing'
            edge_index = None
        else:
            s = [q._key() for q in self.edge2queue]
            s.sort()
            e = s[0][1]
            q = self.edge2queue[e]

            event_type = 'Arrival' if q.next_event_description() == 1 else 'Departure'
            edge_index = q.edge[2]
        return event_type, edge_index

    def reset_colors(self):
        """Resets all edge and vertex colors to their default values."""
        for k, e in enumerate(self.g.edges()):
            self.g.set_ep(e, 'edge_color', self.edge2queue[k].colors['edge_color'])
        for v in self.g.nodes():
            self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_fill_color'])

    def set_transitions(self, mat):
        """Change the routing transitions probabilities for the
        network.

        Parameters
        ----------
        mat : dict or :class:`~numpy.ndarray`
            A transition routing matrix or transition dictionary. If
            passed a dictionary, the keys are source vertex indices and
            the values are dictionaries with target vertex indicies
            as the keys and the probabilities of routing from the
            source to the target as the values.

        Raises
        ------
        ValueError
            A :exc:`.ValueError` is raised if: the keys in the dict
            don't match with a vertex index in the graph; or if the
            :class:`~numpy.ndarray` is passed with the wrong shape,
            must be (``num_vertices``, ``num_vertices``); or the values
            passed are not probabilities (for each vertex they are
            positive and sum to 1);
        TypeError
            A :exc:`.TypeError` is raised if mat is not a dict or
            :class:`~numpy.ndarray`.

        Examples
        --------
        The default transition matrix is every out edge being equally
        likely:

        >>> import queueing_tool as qt
        >>> adjacency = {
        ...     0: [2],
        ...     1: [2, 3],
        ...     2: [0, 1, 2, 4],
        ...     3: [1],
        ...     4: [2],
        ... }
        >>> g = qt.adjacency2graph(adjacency)
        >>> net = qt.QueueNetwork(g)
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        ...                         # doctest: +NORMALIZE_WHITESPACE
        {0: {2: 1.0},
         1: {2: 0.5, 3: 0.5},
         2: {0: 0.25, 1: 0.25, 2: 0.25, 4: 0.25},
         3: {1: 1.0},
         4: {2: 1.0}}

        If you want to change only one vertex's transition
        probabilities, you can do so with the following:

        >>> net.set_transitions({1 : {2: 0.75, 3: 0.25}})
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        ...                         # doctest: +NORMALIZE_WHITESPACE
        {0: {2: 1.0},
         1: {2: 0.75, 3: 0.25},
         2: {0: 0.25, 1: 0.25, 2: 0.25, 4: 0.25},
         3: {1: 1.0},
         4: {2: 1.0}}

        One can generate a transition matrix using
        :func:`.generate_transition_matrix`. You can change all
        transition probabilities with an :class:`~numpy.ndarray`:

        >>> mat = qt.generate_transition_matrix(g, seed=10)
        >>> net.set_transitions(mat)
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        ...                         # doctest: +NORMALIZE_WHITESPACE
        {0: {2: 1.0},
         1: {2: 0.962..., 3: 0.037...},
         2: {0: 0.301..., 1: 0.353..., 2: 0.235..., 4: 0.108...},
         3: {1: 1.0},
         4: {2: 1.0}}

        See Also
        --------
        :meth:`.transitions` : Return the current routing
            probabilities.
        :func:`.generate_transition_matrix` : Generate a random routing
            matrix.
        """
        if isinstance(mat, dict):
            for key, value in mat.items():
                probs = list(value.values())

                if key not in self.g.nodes:
                    msg = "One of the keys don't correspond to a vertex."
                    raise ValueError(msg)
                elif len(self.out_edges[key]) > 0 and not np.isclose(sum(probs), 1):
                    msg = "Sum of transition probabilities at a vertex was not 1."
                    raise ValueError(msg)
                elif (np.array(probs) < 0).any():
                    msg = "Some transition probabilities were negative."
                    raise ValueError(msg)

                for k, e in enumerate(sorted(self.g.out_edges(key))):
                    self._route_probs[key][k] = value.get(e[1], 0)

        elif isinstance(mat, np.ndarray):
            non_terminal = np.array([self.g.out_degree(v) > 0 for v in self.g.nodes()])
            if mat.shape != (self.nV, self.nV):
                msg = ("Matrix is the wrong shape, should "
                       "be {0} x {1}.").format(self.nV, self.nV)
                raise ValueError(msg)
            elif not np.allclose(np.sum(mat[non_terminal, :], axis=1), 1):
                msg = "Sum of transition probabilities at a vertex was not 1."
                raise ValueError(msg)
            elif (mat < 0).any():
                raise ValueError("Some transition probabilities were negative.")

            for k in range(self.nV):
                for j, e in enumerate(sorted(self.g.out_edges(k))):
                    self._route_probs[k][j] = mat[k, e[1]]
        else:
            raise TypeError("mat must be a numpy array or a dict.")

    def show_active(self, **kwargs):
        """Draws the network, highlighting active queues.

        The colored vertices represent vertices that have at least one
        queue on an in-edge that is active. Dark edges represent
        queues that are active, light edges represent queues that are
        inactive.

        Parameters
        ----------
        **kwargs
            Any additional parameters to pass to :meth:`.draw`, and
            :meth:`.QueueNetworkDiGraph.draw_graph`.

        Notes
        -----
        Active queues are :class:`QueueServers<.QueueServer>` that
        accept arrivals from outside the network. The colors are
        defined by the class attribute ``colors``. The relevant keys
        are ``vertex_active``, ``vertex_inactive``, ``edge_active``,
        and ``edge_inactive``.
        """
        g = self.g
        for v in g.nodes():
            self.g.set_vp(v, 'vertex_color', [0, 0, 0, 0.9])
            is_active = False
            my_iter = g.in_edges(v) if g.is_directed() else g.out_edges(v)
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
            else:
                self.g.set_ep(e, 'edge_color', self.colors['edge_inactive'])

        self.draw(update_colors=False, **kwargs)
        self._update_all_colors()

    def show_type(self, edge_type, **kwargs):
        """Draws the network, highlighting queues of a certain type.

        The colored vertices represent self loops of type ``edge_type``.
        Dark edges represent queues of type ``edge_type``.

        Parameters
        ----------
        edge_type : int
            The type of vertices and edges to be shown.
        **kwargs
            Any additional parameters to pass to :meth:`.draw`, and
            :meth:`.QueueNetworkDiGraph.draw_graph`

        Notes
        -----
        The colors are defined by the class attribute ``colors``. The
        relevant colors are ``vertex_active``, ``vertex_inactive``,
        ``vertex_highlight``, ``edge_active``, and ``edge_inactive``.

        Examples
        --------
        The following code highlights all edges with edge type ``2``.
        If the edge is a loop then the vertex is highlighted as well.
        In this case all edges with edge type ``2`` happen to be loops.

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=13)
        >>> net = qt.QueueNetwork(g, seed=13)
        >>> fname = 'edge_type_2.png'
        >>> net.show_type(2, fname=fname) # doctest: +SKIP

        .. figure:: edge_type_2-1.png
           :align: center
        """
        for v in self.g.nodes():
            e = (v, v)
            if self.g.is_edge(e) and self.g.ep(e, 'edge_type') == edge_type:
                ei = self.g.edge_index[e]
                self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_highlight'])
                self.g.set_vp(v, 'vertex_color', self.edge2queue[ei].colors['vertex_color'])
            else:
                self.g.set_vp(v, 'vertex_fill_color', self.colors['vertex_inactive'])
                self.g.set_vp(v, 'vertex_color', [0, 0, 0, 0.9])

        for e in self.g.edges():
            if self.g.ep(e, 'edge_type') == edge_type:
                self.g.set_ep(e, 'edge_color', self.colors['edge_active'])
            else:
                self.g.set_ep(e, 'edge_color', self.colors['edge_inactive'])

        self.draw(update_colors=False, **kwargs)
        self._update_all_colors()

    def simulate(self, n=1, t=None):
        """Simulates the network forward.

        Simulates either a specific number of events or for a specified
        amount of simulation time.

        Parameters
        ----------
        n : int (optional, default: 1)
            The number of events to simulate. If ``t`` is not given
            then this parameter is used.
        t : float (optional)
            The amount of simulation time to simulate forward. If
            given, ``t`` is used instead of ``n``.

        Raises
        ------
        QueueingToolError
            Will raise a :exc:`.QueueingToolError` if the
            ``QueueNetwork`` has not been initialized. Call
            :meth:`.initialize` before calling this method.

        Examples
        --------
        Let ``net`` denote your instance of a ``QueueNetwork``. Before
        you simulate, you need to initialize the network, which allows
        arrivals from outside the network. To initialize with 2 (random
        chosen) edges accepting arrivals run:

        >>> import queueing_tool as qt
        >>> g = qt.generate_pagerank_graph(100, seed=50)
        >>> net = qt.QueueNetwork(g, seed=50)
        >>> net.initialize(2)

        To simulate the network 50000 events run:

        >>> net.num_events
        0
        >>> net.simulate(50000)
        >>> net.num_events
        50000

        To simulate the network for at least 75 simulation time units
        run:

        >>> t0 = net.current_time
        >>> net.simulate(t=75)
        >>> t1 = net.current_time
        >>> t1 - t0 # doctest: +ELLIPSIS
        75...
        """
        if not self._initialized:
            msg = ("Network has not been initialized. "
                   "Call '.initialize()' first.")
            raise QueueingToolError(msg)
        if t is None:
            for dummy in range(n):
                self._simulate_next_event(slow=False)
        else:
            now = self._t
            while self._t < now + t:
                self._simulate_next_event(slow=False)

    def _simulate_next_event(self, slow=True):
        if self._fancy_heap.size == 0:
            self._t = np.infty
            return

        q1k = self._fancy_heap.pop()
        q1t = q1k[0]
        q1 = self.edge2queue[q1k[1]]
        e1 = q1.edge[2]

        event = q1.next_event_description()
        self._t = q1t
        self._qkey = q1k
        self.num_events += 1

        if event == 2:  # This is a departure
            e2 = q1._departures[0].desired_destination(self, q1.edge)
            q2 = self.edge2queue[e2]
            q2k = q2._key()

            if q2.at_capacity() and e2 != e1:
                q2.num_blocked += 1
                q1._departures[0].blocked += 1
                if self._blocking:
                    t = q2._departures[0]._time + EPS * uniform(0.33, 0.66)
                    q1.delay_service(t)
                else:
                    q1.delay_service()
            else:
                agent = q1.next_event()
                agent._time = q1t

                q2._add_arrival(agent)
                self.num_agents[e1] = q1._num_total
                self.num_agents[e2] = q2._num_total

                if slow:
                    self._update_graph_colors(qedge=q1.edge)
                    self._prev_edge = q1.edge

                if q2._active and self.max_agents < np.infty and \
                        np.sum(self.num_agents) > self.max_agents - 1:
                    q2._active = False

                q2.next_event()
                self.num_agents[e2] = q2._num_total

                if slow:
                    self._update_graph_colors(qedge=q2.edge)
                    self._prev_edge = q2.edge

            new_q1k = q1._key()
            new_q2k = q2._key()

            if new_q2k[0] != q2k[0]:
                self._fancy_heap.push(*new_q2k)

                if new_q1k[0] < np.infty and new_q1k != new_q2k:
                    self._fancy_heap.push(*new_q1k)
            else:
                if new_q1k[0] < np.infty:
                    self._fancy_heap.push(*new_q1k)

        elif event == 1:  # This is an arrival
            if q1._active and self.max_agents < np.infty and \
                    np.sum(self.num_agents) > self.max_agents - 1:
                q1._active = False

            q1.next_event()
            self.num_agents[e1] = q1._num_total

            if slow:
                self._update_graph_colors(qedge=q1.edge)
                self._prev_edge = q1.edge

            new_q1k = q1._key()
            if new_q1k[0] < np.infty:
                self._fancy_heap.push(*new_q1k)

    def start_collecting_data(self, queues=None, edge=None, edge_type=None):
        """Tells the queues to collect data on agents' arrival, service
        start, and departure times.

        If none of the parameters are given then every
        :class:`.QueueServer` will start collecting data.

        Parameters
        ----------
        queues : :any:`int`, *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying
            the :class:`QueueServer(s)<.QueueServer>` that will start
            collecting data.
        edge : 2-tuple of int or *array_like* (optional)
            Explicitly specify which queues will collect data. Must be
            either:

            * A 2-tuple of the edge's source and target vertex
              indices, or
            * An iterable of 2-tuples of the edge's source and
              target vertex indices.

        edge_type : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which
            edge types will be set active.
        """
        queues = _get_queues(self.g, queues, edge, edge_type)

        for k in queues:
            self.edge2queue[k].collect_data = True

    def stop_collecting_data(self, queues=None, edge=None, edge_type=None):
        """Tells the queues to stop collecting data on agents.

        If none of the parameters are given then every
        :class:`.QueueServer` will stop collecting data.

        Parameters
        ----------
        queues : int, *array_like* (optional)
            The edge index (or an iterable of edge indices) identifying
            the :class:`QueueServer(s)<.QueueServer>` that will stop
            collecting data.
        edge : 2-tuple of int or *array_like* (optional)
            Explicitly specify which queues will stop collecting data.
            Must be either:

            * A 2-tuple of the edge's source and target vertex
              indices, or
            * An iterable of 2-tuples of the edge's source and
              target vertex indices.

        edge_type : int or an iterable of int (optional)
            A integer, or a collection of integers identifying which
            edge types will stop collecting data.
        """
        queues = _get_queues(self.g, queues, edge, edge_type)

        for k in queues:
            self.edge2queue[k].collect_data = False

    def transitions(self, return_matrix=True):
        """Returns the routing probabilities for each vertex in the
        graph.

        Parameters
        ----------
        return_matrix : bool (optional, the default is ``True``)
            Specifies whether an :class:`~numpy.ndarray` is returned.
            If ``False``, a dict is returned instead.

        Returns
        -------
        out : a dict or :class:`~numpy.ndarray`
            The transition probabilities for each vertex in the graph.
            If ``out`` is an :class:`~numpy.ndarray`, then
            ``out[v, u]`` returns the probability of a transition from
            vertex ``v`` to vertex ``u``. If ``out`` is a dict
            then ``out_edge[v][u]`` is the probability of moving from
            vertex ``v`` to the vertex ``u``.

        Examples
        --------
        Lets change the routing probabilities:

        >>> import queueing_tool as qt
        >>> import networkx as nx
        >>> g = nx.sedgewick_maze_graph()
        >>> net = qt.QueueNetwork(g)

        Below is an adjacency list for the graph ``g``.

        >>> ans = qt.graph2dict(g, False)
        >>> {k: sorted(v) for k, v in ans.items()}
        ...                         # doctest: +NORMALIZE_WHITESPACE
        {0: [2, 5, 7],
         1: [7],
         2: [0, 6],
         3: [4, 5],
         4: [3, 5, 6, 7],
         5: [0, 3, 4],
         6: [2, 4],
         7: [0, 1, 4]}

        The default transition matrix is every out edge being equally
        likely:

        >>> net.transitions(False)  # doctest: +ELLIPSIS
        ...                         # doctest: +NORMALIZE_WHITESPACE
        {0: {2: 0.333..., 5: 0.333..., 7: 0.333...},
         1: {7: 1.0},
         2: {0: 0.5, 6: 0.5},
         3: {4: 0.5, 5: 0.5},
         4: {3: 0.25, 5: 0.25, 6: 0.25, 7: 0.25},
         5: {0: 0.333..., 3: 0.333..., 4: 0.333...},
         6: {2: 0.5, 4: 0.5},
         7: {0: 0.333..., 1: 0.333..., 4: 0.333...}}

        Now we will generate a random routing matrix:

        >>> mat = qt.generate_transition_matrix(g, seed=96)
        >>> net.set_transitions(mat)
        >>> net.transitions(False)  # doctest: +ELLIPSIS
        ...                         # doctest: +NORMALIZE_WHITESPACE
        {0: {2: 0.112..., 5: 0.466..., 7: 0.420...},
         1: {7: 1.0},
         2: {0: 0.561..., 6: 0.438...},
         3: {4: 0.545..., 5: 0.454...},
         4: {3: 0.374..., 5: 0.381..., 6: 0.026..., 7: 0.217...},
         5: {0: 0.265..., 3: 0.460..., 4: 0.274...},
         6: {2: 0.673..., 4: 0.326...},
         7: {0: 0.033..., 1: 0.336..., 4: 0.630...}}

        What this shows is the following: when an :class:`.Agent` is at
        vertex ``2`` they will transition to vertex ``0`` with
        probability ``0.561`` and route to vertex ``6`` probability
        ``0.438``, when at vertex ``6`` they will transition back to
        vertex ``2`` with probability ``0.673`` and route vertex ``4``
        probability ``0.326``, etc.
        """
        if return_matrix:
            mat = np.zeros((self.nV, self.nV))
            for v in self.g.nodes():
                ind = [e[1] for e in sorted(self.g.out_edges(v))]
                mat[v, ind] = self._route_probs[v]
        else:
            mat = {
                k: {e[1]: p for e, p in zip(sorted(self.g.out_edges(k)), value)}
                for k, value in enumerate(self._route_probs)
            }

        return mat

    def _update_all_colors(self):
        do = [True for v in range(self.nV)]
        for q in self.edge2queue:
            e = q.edge[:2]
            v = q.edge[1]
            if q.edge[0] == q.edge[1]:
                self.g.set_ep(e, 'edge_color', q._current_color(1))
                self.g.set_vp(v, 'vertex_color', q._current_color(2))
                if q.edge[3] != 0:
                    self.g.set_vp(v, 'vertex_fill_color', q._current_color())
                do[v] = False
            else:
                self.g.set_ep(e, 'edge_color', q._current_color())
                if do[v]:
                    self._update_vertex_color(v)
                    do[v] = False
                if do[q.edge[0]]:
                    self._update_vertex_color(q.edge[0])
                    do[q.edge[0]] = False

    def _update_vertex_color(self, v):
        ee = (v, v)
        ee_is_edge = self.g.is_edge(ee)
        eei = self.g.edge_index[ee] if ee_is_edge else 0

        if not ee_is_edge or (ee_is_edge and self.edge2queue[eei].edge[3] == 0):
            nSy = 0
            cap = 0
            for ei in self.in_edges[v]:
                nSy += self.edge2queue[ei].num_system
                cap += self.edge2queue[ei].num_servers

            div = (2 * cap) + 2.
            tmp = 1. - min(nSy / div, 1.)

            color = [i * tmp for i in self.colors['vertex_fill_color']]
            color[3] = 1.0
            self.g.set_vp(v, 'vertex_fill_color', color)
            if not ee_is_edge:
                self.g.set_vp(v, 'vertex_color', self.colors['vertex_color'])

    def _update_graph_colors(self, qedge):
        e = qedge[:2]
        v = qedge[1]
        if self._prev_edge is not None:
            pe = self._prev_edge[:2]
            pv = self._prev_edge[1]
            q = self.edge2queue[self._prev_edge[2]]

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


def _get_queues(g, queues, edge, edge_type):
    """Used to specify edge indices from different types of arguments."""
    INT = numbers.Integral
    if isinstance(queues, INT):
        queues = [queues]

    elif queues is None:
        if edge is not None:
            if isinstance(edge, tuple):
                if isinstance(edge[0], INT) and isinstance(edge[1], INT):
                    queues = [g.edge_index[edge]]
            elif isinstance(edge[0], collections.abc.Iterable):
                if np.array([len(e) == 2 for e in edge]).all():
                    queues = [g.edge_index[e] for e in edge]
            else:
                queues = [g.edge_index[edge]]
        elif edge_type is not None:
            if isinstance(edge_type, collections.abc.Iterable):
                edge_type = set(edge_type)
            else:
                edge_type = set([edge_type])
            tmp = []
            for e in g.edges():
                if g.ep(e, 'edge_type') in edge_type:
                    tmp.append(g.edge_index[e])

            queues = np.array(tmp, int)

        if queues is None:
            queues = range(g.number_of_edges())

    return queues
