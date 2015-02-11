import graph_tool.all   as gt
import numpy            as np
import numbers
import copy

from .. generation.graph_preparation   import _prepare_graph

from .. queues          import NullQueue, QueueServer, LossQueue
from .sorting           import oneBisectSort, bisectSort, oneSort, twoSort

from numpy              import infty
from gi.repository      import Gtk, GObject


class QueueNetwork :
    """A class that simulates a network of queues.

    Takes a graph-tool :class:`~graph_tool.Graph` and places queues on each
    edge of the graph. The simulations are event based, and this class handles
    the scheduling of events.

    Each edge on the graph has a *type*, and this *type* is used to define the
    type of ``QueueServer`` that sits on that edge.

    Parameters
    ----------
    g : str or :class:`~graph_tool.Graph`
        The graph specifies the network on which the queues sit.
    q_classes : dict (optional)
        Used to Specify the :class:`~queueing_tool.queues.QueueServer` class
        for each edge type.
    q_args : dict (optional)
        Used to specify the class arguments for each type of
        :class:`~queueing_tool.queues.QueueServer`.
    seed : int (optional)
        An integer used to initialize ``numpy``'s and ``graph-tool``'s
        psuedorandom number generators.

    Attributes
    ----------
    g : :class:`~graph_tool.Graph`
        The graph for the network.
    in_edges : list
        A list of all in-edges for each vertex. Specifically, ``in_edges[v]``
        returns a list containing the edge index for all edges with the
        head of the edge at ``v``, where ``v`` is a vertex's index.
    out_edges : list
        A list of all out-edges for each vertex. Specifically, ``out_edges[v]``
        returns a list containing the edge index for all edges with the
        tail of the edge at ``v``, where ``v`` is a vertex's index.
    edge2queue : list
        A list of queues where the ``edge2queue[k]`` returns the queue on the
        edge with edge index ``k``.
    nAgents : :class:`~numpy.ndarray`
        A one-dimensional array where the ``k``'th entry corresponds to the
        total number of agents in the ``QueueServer`` with edge index ``k``.
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
    agent_cap : int (the default is 1000)
        The maximum number of agents that can be in the network at any time.
    colors : dict
        A dictionary of colors used when drawing a graph. See the notes for the
        defaults.


    Raises
    ------
    TypeError
        The parameter ``g`` must be either a :class:`~graph_tool.Graph`, a string or
        file location to a graph, or ``None``. Raises a :exc:`~TypeError` otherwise.

    Methods
    -------

    Notes
    -----
    * This class must be initialized before any simulations can take place. To
      initialize, call the :meth:`~initialize` method. If any of the queues are
      altered, make sure to run the ``initialize`` method again.
    * When simulating the network, the departure of an agent coincides with an 
      arrival to another queue. There is no time lag between these events.
    

    The default colors are:

    >>> self.colors       = { 'vertex_fill_color' : [0.9, 0.9, 0.9, 1.0],
    ...                       'vertex_color'      : [0.0, 0.5, 1.0, 1.0],
    ...                       'vertex_type'       : [0.5, 0.5, 0.5, 1.0],
    ...                       'edge_departure'    : [0, 0, 0, 1], 
    ...                       'vertex_halo_color' : [0, 0, 0, 0],
    ...                       'halo_arrival'      : [0.1, 0.8, 0.8, 0.25],
    ...                       'halo_departure'    : [0.9, 0.9, 0.9, 0.25],
    ...                       'vertex_active'     : [0.1, 1.0, 0.5, 1.0],
    ...                       'vertex_inactive'   : [0.9, 0.9, 0.9, 0.8],
    ...                       'edge_active'       : [0.1, 0.1, 0.1, 1.0],
    ...                       'edge_inactive'     : [0.8, 0.8, 0.8, 0.3],
    ...                       'bg_color'          : [1, 1, 1, 1]}
    """

    def __init__(self, g=None, q_classes=None, q_args=None, seed=None) :
        self.nEvents      = 0
        self.t            = 0
        self.agent_cap    = 1000
        self.colors       = { 'vertex_fill_color' : [0.9, 0.9, 0.9, 1.0],
                              'vertex_color'      : [0.0, 0.5, 1.0, 1.0],
                              'vertex_type'       : [0.5, 0.5, 0.5, 1.0],
                              'edge_departure'    : [0, 0, 0, 1],
                              'vertex_halo_color' : [0, 0, 0, 0],
                              'halo_arrival'      : [0.1, 0.8, 0.8, 0.25],
                              'halo_departure'    : [0.9, 0.9, 0.9, 0.25],
                              'vertex_active'     : [0.1, 1.0, 0.5, 1.0],
                              'vertex_inactive'   : [0.9, 0.9, 0.9, 0.8],
                              'edge_active'       : [0.1, 0.1, 0.1, 1.0],
                              'edge_inactive'     : [0.8, 0.8, 0.8, 0.3],
                              'bg_color'          : [1, 1, 1, 1]}

        self._to_animate  = False
        self._initialized = False
        self._prev_edge   = None
        self._queues      = []

        if q_classes is None :
            q_classes = {0 : NullQueue, 1 : QueueServer, 2 : LossQueue, 3 : LossQueue, 4 : LossQueue}

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
                raise TypeError("The Parameter `g` needs to be either a graph-tool Graph, a string, or None.")

            def edge_index(e) :
                return g.edge_index[e]

            self.out_edges  = [ [i for i in map(edge_index, list(v.out_edges()))] for v in g.vertices()]
            self.in_edges   = [ [i for i in map(edge_index, list(v.in_edges()))] for v in g.vertices() ]
            self.edge2queue = qs
            self.nAgents    = np.zeros( g.num_edges() )

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

    def initialize(self, nActive=1, queues=None, edges=None, types=None) :
        """Prepares the ``QueueNetwork`` for simulation.

        Each :class:`~queueing_tool.queues.QueueServer` in the network starts inactive,
        which means they do not accept arrivals from outside the network, and they have
        no :class:`~queueing_tool.queues.Agent`\s in their system. Note that in order to
        simulate the ``QueueNetwork``, there must be at least one ``Agent`` in the
        network. This method sets queues to active, which then allows agents to arrive
        from outside the network.

        Parameters
        ----------
        nActive : int (optional, the default is ``1``)
            The number of queues to set as active. The queues are selected randomly.
        queues : `array_like` (optional)
            Used to explicitly specify which queues to make active by passing
            their edge index. Must be an iterable of integers representing edges
            in the graph.
        edges : `array_like` (optional)
            Explicitly specify which queues to make active by passing either the 
            edge's source and target vertex indices, or the 
            :class:`~graph_tool.Edge` . Must have 2 slots with integers in
            both, or be an iterable of 2-`array_like` objects.
        types : int or `array_like` (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.

        Raises
        ------
        RuntimeError
            If ``queues``, ``egdes``, and ``types`` are all ``None`` and 
            ``nActive`` is not an integer or is less than 1 then a 
            :exc:`~RuntimeError` is raised.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edges is not None :
                if not isinstance(edges[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edges]
                elif isinstance(edges[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edges]
                else :
                    queues = [self.g.edge_index[self.g.edge(edges[0], edges[1])]]
            elif types is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), types) )[0]
            elif nActive >= 1 and isinstance(nActive, numbers.Integral) :
                queues = np.arange(self.nE)  
                np.random.shuffle(queues)
                queues = queues[:nActive]
            else :
                raise RuntimeError("If queues is None, then nActive must be a strictly positive int.")

        for ei in queues :
            self.edge2queue[ei].set_active()

        self._queues = [q for q in self.edge2queue]
        self._queues.sort()
        while self._queues[-1]._time == infty :
            self._queues.pop()

        self._queues.sort(reverse=True)
        self._initialized  = True


    def collect_data(self, queues=None, edges=None, types=None) :
        """Tells the queues to collect data on agents.

        Parameters
        ----------
        queues : int, `array_like` (optional)
            An integer (or any iterable of integers) identifying the
            :class:`~queueing_tool.queues.QueueServer`\(s) that will start
            collecting data. If ``queues`` is not specified then every
            ``QueueServer``\'s data will start collecting data.
        edges : `array_like` (optional)
            Explicitly specify which queues to make active by passing either the 
            edge's source and target vertex indices, or the 
            :class:`~graph_tool.Edge` . Must have 2 slots with integers in
            both, or be an iterable of 2-`array_like` objects.
        types : int or `array_like` (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edges is not None :
                if not isinstance(edges[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edges]
                elif isinstance(edges[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edges]
                else :
                    queues = [self.g.edge_index[self.g.edge(edges[0], edges[1])]]
            elif types is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), types) )[0]
            else :
                queues = range(self.nE)

        for k in queues :
            self.edge2queue[k].collect_data = True


    def stop_collecting_data(self, queues=None, edges=None, types=None) :
        """Tells the queues to stop collecting data on agents.

        Parameters
        ----------
        queues : int, `array_like` (optional)
            An integer (or any iterable of integers) identifying the
            :class:`~queueing_tool.queues.QueueServer`\(s) that will stop 
            collecting data. If ``queues`` is not specified then every
            ``QueueServer``\'s data will stop collecting data.
        edges : `array_like` (optional)
            Explicitly specify which queues to make active by passing either the 
            edge's source and target vertex indices, or the 
            :class:`~graph_tool.Edge` . Must have 2 slots with integers in
            both, or be an iterable of 2-`array_like` objects.
        types : int or `array_like` (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edges is not None :
                if not isinstance(edges[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edges]
                elif isinstance(edges[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edges]
                else :
                    queues = [self.g.edge_index[self.g.edge(edges[0], edges[1])]]
            elif types is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), types) )[0]
            else :
                queues = range(self.nE)

        for k in queues :
            self.edge2queue[k].collect_data = False


    def data_queues(self, queues=None, edges=None, types=None) :
        """Fetches data from queues.

        Parameters
        ----------
        queues : int, `array_like` (optional)
            An integer (or any iterable of integers) identifying the
            :class:`~queueing_tool.queues.QueueServer`\(s) whose data will be
            retrieved. If ``queues`` is not specified then every
            ``QueueServer``\'s data will be retrieved.
        edges : `array_like` (optional)
            Explicitly specify which queues to make active by passing either the 
            edge's source and target vertex indices, or the 
            :class:`~graph_tool.Edge` . Must have 2 slots with integers in
            both, or be an iterable of 2-`array_like` objects.
        types : int or `array_like` (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.

        Returns
        -------
        out : :class:`~numpy.ndarray`
            A five column numpy array of all the data. The first, second, and
            third columns represent, respectively, the arrival, service start,
            and departure times of each :class:`~queueing_tool.queues.Agent`
            that has visited the queue. The fourth column identifies how many 
            other agents were in the queue upon arrival, and the fifth column
            identifies which queue this occurred at.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edges is not None :
                if not isinstance(edges[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edges]
                elif isinstance(edges[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edges]
                else :
                    queues = [self.g.edge_index[self.g.edge(edges[0], edges[1])]]
            elif types is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), types) )[0]
            else :
                queues = range(self.nE)

        data = np.zeros( (1,5) )
        for q in queues :
            qdata = []
            for d in self.edge2queue[q].data.values() :
                qdata.extend(d)

            if len(qdata) > 0 :
                dat       = np.zeros( (len(qdata), 5) )
                dat[:,:4] = np.array(qdata)
                dat[:, 4] = q
                data      = np.vstack( (data, dat) )

        return data[1:,:]


    def data_agents(self, queues=None, edges=None, types=None) :
        """Fetches data from queues, and organizes it by agent.

        Parameters
        ----------
        queues : int or `array_like` (optional)
            An integer (or any iterable of integers) identifying the
            :class:`~queueing_tool.queues.QueueServer`\(s) whose data will be
            retrieved. If ``queues`` is not specified then every
            ``QueueServer``\'s data will be retrieved.
        edges : `array_like` (optional)
            Explicitly specify which queues to make active by passing either the 
            edge's source and target vertex indices, or the 
            :class:`~graph_tool.Edge` . Must have 2 slots with integers in
            both, or be an iterable of 2-`array_like` objects.
        types : int or `array_like` (optional)
            A integer, or a collection of integers identifying which edge types
            will be set active.

        Returns
        -------
        out : dict
            Returns a ``dict`` where the keys are the 
            :class:`~queueing_tool.queues.Agent`\'s ``issn`` and the values are
            :class:`~numpy.ndarray`\s for that ``Agent``\'s data. The first,
            second, and third columns represent, respectively, the arrival,
            service start, and departure times of that ``Agent`` at a queue.
            The fourth column identifies how many other agents were in the
            queue upon arrival, and the fifth column identifies which queue
            this occurred at.
        """
        if isinstance(queues, numbers.Integral) :
            queues = [queues]
        elif queues is None :
            if edges is not None :
                if not isinstance(edges[0], numbers.Integral) :
                    queues = [self.g.edge_index[self.g.edge(u,v)] for u,v in edges]
                elif isinstance(edges[0], gt.Edge) :
                    queues = [self.g.edge_index[e] for e in edges]
                else :
                    queues = [self.g.edge_index[self.g.edge(edges[0], edges[1])]]
            elif types is not None :
                queues = np.where(np.in1d(np.array(self.g.ep['eType'].a), types) )[0]
            else :
                queues = range(self.nE)
        data = {}
        for q in queues :
            for issn, dat in self.edge2queue[q].data.items() :
                datum = np.zeros( (len(dat), 5) )
                datum[:,:4] = np.array(dat)
                datum[:, 4] = q
                if issn in data :
                    data[issn] = np.vstack( (data[issn], datum) )
                else :
                    data[issn] = datum

        dType = [('a', float), ('s', float), ('d', float), ('q', float), ('id', float)]
        for issn, dat in data.items() :
            datum = np.array([tuple(d) for d in dat.tolist()], dtype=dType)
            datum = np.sort(datum, order='a')
            data[issn] = np.array([tuple(d) for d in datum])

        return data


    def draw(self, out_size=(700, 700), output=None, update_colors=True, **kwargs) :
        """Draws the network. The coloring of the network corresponds to the 
        number of agents at each queue.

        Parameters
        ----------
        out_size : tuple (optional, the default is ``(700, 700)``).
            Specifies the size of canvas. See
            `graph-tool`_ \'s documentation.
        output : str (optional, the default is ``None``)
            Specifies the directory where the drawing is saved. If output is
            ``None``, then the results are drawn using GraphViz.
        update_colors : bool (optional, the default is ``True``).
            Specifies whether all the colors are updated.
        **kwargs : 
            Any extra parameters to pass to :func:`~graph_tool.draw.graph_draw`.

            .. _graph-tool: http://graph-tool.skewed.de/static/doc/index.html
        """
        if update_colors :
            self._update_all_colors()

        more_kwargs = {'geometry' : out_size} if output is None else {'output_size': out_size , 'output' : output}
        kwargs.update(more_kwargs)

        ans = gt.graph_draw(g=self.g, pos=self.g.vp['pos'],
                bg_color=self.colors['bg_color'],
                edge_color=self.g.ep['edge_color'],
                edge_control_points=self.g.ep['edge_control_points'],
                edge_marker_size=self.g.ep['edge_marker_size'],
                edge_pen_width=self.g.ep['edge_pen_width'],
                vertex_color=self.g.vp['vertex_color'],
                vertex_fill_color=self.g.vp['vertex_fill_color'],
                vertex_halo=self.g.vp['vertex_halo'],
                vertex_halo_color=self.g.vp['vertex_halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                vertex_size=self.g.vp['vertex_size'], **kwargs)


    def show_active(self) :
        """Draws the network, highlighting active queues.

        The colored vertices represent vertices that have at least one queue
        on an in-edge that is active. Dark edges represent queues that are
        active, light edges represent queues that are inactive.

        Notes
        -----
        The colors are defined by the class attribute ``colors`` as the keys
        ``vertex_active``, ``vertex_inactive``, ``edge_active``, ``edge_inactive``.
        """
        for v in self.g.vertices() :
            self.g.vp['vertex_color'][v] = [0, 0, 0, 0.9]
            is_active = False
            my_iter   = v.in_edges() if self.g.is_directed() else v.out_edges()
            for e in my_iter :
                ei = self.g.edge_index[e]
                if self.edge2queue[ei].active :
                    is_active = True
                    break
            if is_active :
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_active']
            else :
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_inactive']

        for e in self.g.edges() :
            ei = self.g.edge_index[e]
            if self.edge2queue[ei].active :
                self.g.ep['edge_color'][e] = self.colors['edge_active']
            else :
                self.g.ep['edge_color'][e] = self.colors['edge_inactive']

        self.draw(update_colors=False)
        self._update_all_colors()


    def show_type(self, n) :
        """Draws the network, highlighting queues of a certain type.

        The colored vertices represent self loops of type ``n``. Dark edges represent
        queues of type ``n``.

        Parameters
        ----------
        n : int
            The type of vertices and edges to be shown.

        Notes
        -----
        The colors are defined by the class attribute ``colors``.
        """
        for v in self.g.vertices() :
            is_active = False
            edge_iter = v.in_edges() if self.g.is_directed() else v.out_edges()
            for e in edge_iter :
                ei = self.g.edge_index[e]
                if self.edge2queue[ei].active :
                    is_active = True
                    break

            e   = self.g.edge(v, v)
            if isinstance(e, gt.Edge) and self.g.ep['eType'][e] == n :
                ei  = self.g.edge_index[e]
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_type']
                self.g.vp['vertex_color'][v]      = self.edge2queue[ei].colors['vertex_color']
            else :
                self.g.vp['vertex_fill_color'][v] = self.colors['vertex_inactive']
                self.g.vp['vertex_color'][v]      = [0, 0, 0, 0.9]

        for e in self.g.edges() :
            ei = self.g.edge_index[e]
            if self.edge2queue[ei].eType == n :
                self.g.ep['edge_color'][e] = self.colors['edge_active']
            else :
                self.g.ep['edge_color'][e] = self.colors['edge_inactive']

        self.draw(update_colors=False)
        self._update_all_colors()


    def _update_all_colors(self) :
        ep  = self.g.ep
        vp  = self.g.vp
        do  = [True for v in range(self.nV)]
        for q in self.edge2queue :
            e = self.g.edge(q.edge[0], q.edge[1])
            v = self.g.vertex(q.edge[1])
            if q.edge[0] == q.edge[1] :
                ep['edge_color'][e]         = q.current_color(1)
                vp['vertex_color'][v]       = q.current_color(2)
                vp['vertex_fill_color'][v]  = q.current_color()
            else :
                ep['edge_color'][e]   = q.current_color()
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
                ep['edge_color'][pe]        = q.current_color(1)
                vp['vertex_color'][pv]      = q.current_color(2)
                vp['vertex_fill_color'][pv] = q.current_color()
                vp['vertex_halo_color'][pv] = self.colors['vertex_halo_color']
                vp['vertex_halo'][pv]       = False
            else :
                ep['edge_color'][pe] = q.current_color()
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
            ep['edge_color'][e]         = q.current_color(1)
            vp['vertex_fill_color'][v]  = q.current_color()
            vp['vertex_color'][v]       = q.current_color(2)
            vp['vertex_halo'][v]        = True

            if ad == 'arrival' :
                vp['vertex_halo_color'][v] = self.colors['halo_arrival']
            elif ad == 'departure' :
                vp['vertex_halo_color'][v] = self.colors['halo_departure']
        else :
            ep['edge_color'][e] = q.current_color()
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
        q._append_departure(agent, t)

        if qt == infty and q._time < infty :
            self._queues.append(q)

        self._queues.sort(reverse=True)


    def _add_arrival(self, ei, agent, t=None) :
        q   = self.edge2queue[ei]
        qt  = q._time
        if t is None :
            t = q._time + 1 if q._time < infty else self._queues[-1]._time + 1

        agent.set_arrival(t)
        q._add_arrival(agent)

        if qt == infty and q._time < infty :
            self._queues.append(q)

        self._queues.sort(reverse=True)


    def next_event_description(self) :
        """Returns whether the next event is either an arrival or a departure
        and the edge index corresponding to that edge.

        Returns
        -------
        out : tuple
            Returns a 2-tuple where the first entry indicates whether the next
            event is an arrival or a departure, and the second entry is which
            queue/edge this event corresponds to.

        Raises
        ------
        IndexError
            This error is raised if there are no events scheduled.
        """
        if len(self._queues) == 0 :
            ans = "Nothing"
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

        if event == 2 : # This is a departure
            agent             = q1.next_event()
            self.nAgents[e1]  = q1._nTotal
            self.nEvents     += 1

            e2  = agent.desired_destination(self, q1.edge) # expects QueueNetwork, and current location
            q2  = self.edge2queue[e2]
            q2t = q2._time
            agent.set_arrival(q1t)

            q2._add_arrival(agent)
            self.nAgents[e2] = q2._nTotal

            if slow :
                self._update_graph_colors(ad='departure', qedge=q1.edge)
                self._prev_edge = q1.edge

            if q2.active and np.sum(self.nAgents) > self.agent_cap - 1 :
                q2.active = False

            if q2._departures[0]._time < q2._arrivals[0]._time :
                print("WHOA! THIS NEEDS CHANGING! %s %s %s" % (q2._departures[0]._time, q2._arrivals[0]._time, q2) )

            q2.next_event()

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
            if q1.active and np.sum(self.nAgents) > self.agent_cap - 1 :
                q1.active = False

            q1.next_event()
            self.nAgents[e1]  = q1._nTotal
            self.nEvents     += 1

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
            return True


    def animate(self, out_size=(700, 700), **kwargs) :
        """Animates the network as it's simulating.

        Closing the window ends the animation.

        Parameters
        ----------
        out_size : tuple (optional, the default is ``(700, 700)``).
            The size of the canvas used for the animation.
        **kwargs :
            Any extra parameters to pass to :class:`~graph_tool.draw.GraphWindow`.

        Raises
        ------
        RuntimeError
            Will raise a :exc:`~RuntimeError` if the ``QueueNetwork`` has not
            been initialized. Call
            :meth:`~queueing_tool.network.QueueNetwork.initialize` before running.
        """
        if not self._initialized :
            raise RuntimeError("Network has not been initialized. Call 'initialize()' first.")

        self._to_animate = True
        self._update_all_colors()
        self._window = gt.GraphWindow(g=self.g, pos=self.g.vp['pos'],
                geometry=out_size,
                bg_color=self.colors['bg_color'],
                edge_color=self.g.ep['edge_color'],
                edge_control_points=self.g.ep['edge_control_points'],
                edge_marker_size=self.g.ep['edge_marker_size'],
                edge_pen_width=self.g.ep['edge_pen_width'],
                vertex_color=self.g.vp['vertex_color'],
                vertex_fill_color=self.g.vp['vertex_fill_color'],
                vertex_halo=self.g.vp['vertex_halo'],
                vertex_halo_color=self.g.vp['vertex_halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                vertex_size=self.g.vp['vertex_size'], **kwargs)

        cid = GObject.idle_add(self._simulate_next_event)
        self._window.connect("delete_event", Gtk.main_quit)
        self._window.show_all()
        Gtk.main()
        self._to_animate = False


    def simulate(self, n=None, t=25) :
        """Simulates the network forward.

        This method simulates the network forward for a specified amount of
        *system time* ``t``\, or for a specific number of events ``n``.

        Parameters
        ----------
        n : int (optional)
            The number of events to simulate. Supercedes the parameter ``t`` if
            supplied.
        t : float (optional, the default is ``25``)
            The amount of system time to simulate forward. If ``n`` is ``None``
            then this parameter is used.

        Raises
        ------
        RuntimeError
            Will raise a :exc:`~RuntimeError` if the ``QueueNetwork`` has not been
            initialized. Call
            :meth:`~queueing_tool.network.QueueNetwork.initialize` before running.
        """
        if not self._initialized :
            raise RuntimeError("Network has not been initialized. Call '.initialize()' first.")
        if n is None :
            now = self.t
            while self.t < now + t :
                self._simulate_next_event(slow=False)
        elif isinstance(n, numbers.Integral) :
            for k in range(n) :
                self._simulate_next_event(slow=False)


    def reset_colors(self) :
        """Sets all edge and vertex colors to their default values."""
        for k, e in enumerate(self.g.edges()) :
            self.g.ep['edge_color'][e]    = self.edge2queue[k].colors['edge_color']
        for k, v in enumerate(self.g.vertices()) :
            self.g.vp['vertex_fill_color'][v] = self.colors['vertex_fill_color']
            self.g.vp['vertex_halo_color'][v] = self.colors['vertex_halo_color']
            self.g.vp['vertex_halo'][v]       = False
            self.g.vp['vertex_text'][v]       = ''


    def clear(self) :
        """Resets the queue to its initial state.

        The attributes ``t``, ``nEvents``, ``nAgents`` are set to zero
        :meth:`~queueing_tool.network.QueueNetwork.reset_colors` is called;
        and the :class:`~queueing_tool.queues.QueueServer` method
        :meth:`~queueing_tool.queues.QueueServer.clear` is called for each
        queue in the network.

        Notes
        -----
        ``QueueNetwork`` must be re-initialized before any simulations can run.
        """
        self.t            = 0
        self.nEvents      = 0
        self.nAgents      = np.zeros(self.nE)
        self._to_animate  = False
        self._prev_edge   = None
        self._initialized = False
        self.reset_colors()
        for q in self.edge2queue :
            q.clear()


    def clear_data(self, queues=None) :
        """Clears data from queues.

        Parameters
        ----------
        queues : int, list, tuple (optional)
            An integer (or an iterable of integers) identifying the 
            :class:`~queueing_tool.queues.QueueServer`
            whose data will be cleared. If ``queues`` is not specified then all
            ``QueueServer``'s data will be cleared.
        """
        queues = [queues] if isinstance(queues, numbers.Integral) else queues

        if queues is None :
            for q in self.edge2queue :
                q.data = {}
        else :
            for k in queues :
                self.edge2queue[k].data = {}


    def copy(self) :
        """Returns a deep copy of self."""
        net               = QueueNetwork()
        net.g             = self.g.copy()
        net.t             = copy.copy(self.t)
        net.agent_cap     = copy.copy(self.agent_cap)
        net.nV            = copy.copy(self.nV)
        net.nE            = copy.copy(self.nE)
        net.nAgents       = copy.copy(self.nAgents)
        net.nEvents       = copy.copy(self.nEvents)
        net.shortest_path = copy.copy(self.shortest_path)
        net._initialized  = copy.copy(self._initialized)
        net._prev_edge    = copy.copy(self._prev_edge)
        net._to_animate   = copy.copy(self._to_animate)
        net.colors        = copy.deepcopy(self.colors)
        net.out_edges     = copy.deepcopy(self.out_edges)
        net.in_edges      = copy.deepcopy(self.in_edges)
        net.edge2queue    = copy.deepcopy(self.edge2queue)

        if net._initialized :
            net.queues = [q for q in net.edge2queue]
            net.queues.sort()
            while net.queues[-1]._time == infty :
                net.queues.pop()

            net.queues.sort(reverse=True)
        return net



class CongestionNetwork(QueueNetwork) :
    """A network of queues that handles congestion by holding back agents.

    This class is identical to the :class:`~QueueNetwork` class, with the only
    exception being how blocking is handled when an agent is blocked by a
    :class:`~queueing_tool.queues.LossQueue`. In this network, if an agent will
    be blocked at his desired destination then they are held back at
    his current queue and receives an extra service there.
    """
    def __init__(self, g=None, q_classes=None, q_args=None, seed=None) :
        QueueNetwork.__init__(self, g, seed, calcpath)


    def __repr__(self) :
        return 'CongestionNetwork. # nodes: %s, edges: %s, agents: %s' % (self.nV, self.nE, np.sum(self.nAgents))


    def _simulate_next_event(self, slow=True) :
        q1  = self.queues.pop()
        q1t = q1._time
        e1  = q1.edge[2]

        event  = q1.next_event_description()
        self.t = q1t
        self.nEvents += 1 if event else 0

        if event == 2 : # This is a departure
            e2  = q1._departures[0].desired_destination(self, q1.edge) # expects QueueNetwork, and current location
            q2  = self.edge2queue[e2]
            q2t = q2._time

            if q2.at_capacity() :
                q2.nBlocked += 1
                q1._departures[0].blocked += 1
                q1.delay_service()
            else :
                agent = q1.next_event()
                agent.set_arrival(q1t)

                q2._add_arrival(agent)

                self.nAgents[e1]  = q1._nTotal
                self.nAgents[e2]  = q2._nTotal

                if q2.active and np.sum(self.nAgents) > self.agent_cap - 1 :
                    q2.active = False

                if slow :
                    self._update_graph_colors(ad='departure', qedge=q1.edge)
                    self._prev_edge = q1.edge

                if q2._departures[0]._time <= q2._arrivals[0]._time :
                    print("WHOA! THIS NEEDS CHANGING! %s %s" % (q2._departures[0]._time, q2._arrivals[0]._time) )

                q2.next_event()

                if slow :
                    self._update_graph_colors(ad='arrival', qedge=q2.edge)
                    self._prev_edge = q2.edge

            if q1._time < infty :
                if q2._time < q2t < infty and e2 != e1 :
                    oneBisectSort(self.queues, q1, q2t, len(self.queues))
                elif q2._time < q2t and e2 != e1 :
                    twoSort(self.queues, q1, q2, len(self.queues))
                else :
                    bisectSort(self.queues, q1, len(self.queues))
            else :
                if q2._time < q2t < infty :
                    oneSort(self.queues, q2t, len(self.queues))
                elif q2._time < q2t :
                    bisectSort(self.queues, q2, len(self.queues))

        elif event == 1 : # This is an arrival
            if q1.active and np.sum(self.nAgents) > self.agent_cap - 1 :
                q1.active = False

            q1.next_event()
            self.nAgents[e1]  = q1._nTotal

            if slow :
                self._update_graph_colors(ad='arrival', qedge=q1.edge)
                self._prev_edge  = q1.edge

            if q1._time < infty :
                bisectSort(self.queues, q1, len(self.queues) )

        if self._to_animate :
            self._window.graph.regenerate_surface(lazy=False)
            self._window.graph.queue_draw()
            return True
