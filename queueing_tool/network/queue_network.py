"""The Queueing network module.
"""

import numpy            as np
import graph_tool.all   as gt
import copy

from .. generation.graph_preparation   import _prepare_graph

from .. queues          import NullQueue, QueueServer, LossQueue
from .sorting           import oneBisectSort, bisectSort, oneSort, twoSort

from numpy              import infty
from gi.repository      import Gtk, GObject




# Garages changed to FCQ (finite capacity queue)
# Each edge and vertex has an eType and vType respectively. 
#   The eType is used to specify they queue class to use, as well as
#   the default values to create the queue with.

class QueueNetwork :
    """The class that handles the graph and all the queues on the graph.

    Parameters
    ----------
    g : graph (a :class:`~graph_tool.Graph` instance or the location of a graph)
        The graph specifies the network on which the queues sit.
    q_classes : dict (optional)
        This allows the user to specify the type of :class:`~queueing_tool.queues.QueueServer`
        class for each edge.
    q_args : dict (optional)
        This allows the user to specify the class arguments for each type of
        :class:`~queueing_tool.queues.QueueServer`.
    seed : int (optional)
        An integer used to initialize ``numpy``'s and ``graph-tool``'s psuedorandom number generators.

    Attributes
    ----------
    g : :class:`~graph_tool.Graph`
        The graph for the network.
    in_edges : list
        A list of all in-edges for each vertex. Specifically, ``in_edges[v]`` returns a list 
        edge indeces corresponding to each of the edges with the **tail** of the edge at ``v``, 
        where ``v`` is a vertex's index.       
    out_edges : list
        A list of all out-edges for each vertex.
    edge2queue : list
        A list of queues where the ``edge2queue[k]`` returns the queue on the edge with edge index ``k``.
    nAgents : :class:`~numpy.ndarray`
        A one-dimensional array where the ``k``'th entry corresponds to the total number of agents at
        the edge with edge index ``k``.
    nEdges : int
        The number of edges in the graph.
    nEvents : int
        The number of that have occurred thus far. Every arrival from outside the network counts as
        one event, but the departure of an agent from a queue and the arrival of that same agent to 
        another queue counts as one event.
    nVertices : int
        The number of vertices in the graph.
    time : float
        The time of the last event.
    agent_cap : int (the default is 1000)
        The maximum number of agents that can be in the queue at any time.
    colors : dict
        A dictionary of colors used when drawing a graph. See the notes for the defaults


    Raises
    ------
    TypeError
        The parameter ``g`` must be either a :class:`~graph_tool.Graph`, a string or file location to a graph, 
        or ``None``. Raises a :exc:`~TypeError` otherwise.

    Methods
    -------

    Notes
    -----

    The default colors are:

    >>> self.colors       = { 'vertex_normal'   : [0.9, 0.9, 0.9, 1.0],
    ...                       'vertex_color'    : [0.0, 0.5, 1.0, 1.0],
    ...                       'edge_departure'  : [0, 0, 0, 1], 
    ...                       'halo_normal'     : [0, 0, 0, 0],
    ...                       'halo_arrival'    : [0.1, 0.8, 0.8, 0.25],
    ...                       'halo_departure'  : [0.9, 0.9, 0.9, 0.25],
    ...                       'vertex_active'   : [0.1, 1.0, 0.5, 1.0],
    ...                       'vertex_inactive' : [0.9, 0.9, 0.9, 0.8],
    ...                       'edge_active'     : [0.1, 0.1, 0.1, 1.0],
    ...                       'edge_inactive'   : [0.8, 0.8, 0.8, 0.3],
    ...                       'text_normal'     : [1, 1, 1, 0.5],
    ...                       'bg_color'        : [1, 1, 1, 1]}
    """

    def __init__(self, g=None, q_classes=None, q_args=None, seed=None) :
        self.nEvents      = 0
        self.t            = 0
        self.agent_cap    = 1000
        self.colors       = { 'vertex_normal'   : [0.9, 0.9, 0.9, 1.0],
                              'vertex_color'    : [0.0, 0.5, 1.0, 1.0],
                              'vertex_type'     : [0.5, 0.5, 0.5, 1.0],
                              'edge_departure'  : [0, 0, 0, 1],
                              'halo_normal'     : [0, 0, 0, 0],
                              'halo_arrival'    : [0.1, 0.8, 0.8, 0.25],
                              'halo_departure'  : [0.9, 0.9, 0.9, 0.25],
                              'vertex_active'   : [0.1, 1.0, 0.5, 1.0],
                              'vertex_inactive' : [0.9, 0.9, 0.9, 0.8],
                              'edge_active'     : [0.1, 0.1, 0.1, 1.0],
                              'edge_inactive'   : [0.8, 0.8, 0.8, 0.3],
                              'text_normal'     : [1, 1, 1, 0.5],
                              'bg_color'        : [1, 1, 1, 1]}

        self._to_animate  = False
        self._initialized = False
        self._prev_edge   = None

        if q_classes is None :
            q_classes = {0 : NullQueue, 1 : QueueServer, 2 : LossQueue, 3 : LossQueue, 4 : LossQueue}

        if q_args is None :
            q_args  = {k : {} for k in range(5)}
        else :
            for k in set(q_classes.keys()) - set(q_args.keys()) :
                q_args[k] = {}

        v_pens    = [[0, 0.5, 1, 1], [0, 0.5, 1, 1], [0.133, 0.545, 0.133, 1],
                     [0.282, 0.239, 0.545, 1], [1, 0.135, 0, 1]]
        q_colors  = {k : {'edge_loop'     : [0, 0, 0, 0],
                          'edge_normal'   : [0.7, 0.7, 0.7, 0.5],
                          'vertex_normal' : [0.9, 0.9, 0.9, 1.0],
                          'vertex_pen'    : v_pens[k]} for k in range(5)}

        for key, args in q_args.items() :
            if 'colors' not in args :
                args['colors'] = q_colors[key]

        if isinstance(seed, int) :
            np.random.seed(seed)
            gt.seed_rng(seed)

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

    def initialize(self, nActive=1, queues=None) :
        """Prepares the ``QueueNetwork`` for simulation.

        Each :class:`~queueing_tool.queues.QueueServer` in the network starts inactive, 
        which means they do not accept arrivals from outside the network, and they have
        no :class:`~queueing_tool.queues.Agent` in their systems. Note that in order to 
        simulate the ``QueueNetwork``, there must be at least one 
        :class:`~queueing_tool.queues.Agent` in the network. This method sets queues
        to active, which then allows agents to arrive from outside the network.

        Parameters
        ----------
        nActive : int (optional, the default is one)
            The number of queues to set as active. The queues are selected randomly.
        queues : list, tuple (optional, can be any iterable)
            Used to explicitly specify which queues to make active. Must be an iterable
            of integers representing edges in the graph.

        Raises
        ------
        RuntimeError
            If ``queues`` is ``None`` and ``nActive`` is not an integer or is less than 1
            then a :exc:`~RuntimeError` is raised.
        """
        if queues is None :
            if nActive < 1 or not isinstance(nActive, int) :
                raise RuntimeError("If queues is None, then nActive must be a strictly positive int.")
                return
            queues = np.arange(self.nE)  
            np.random.shuffle(queues)
            queues = queues[:nActive]

        for ei in queues :
            self.edge2queue[ei].set_active()

        self.queues = [q for q in self.edge2queue]
        self.queues.sort()
        while self.queues[-1]._time == infty :
            self.queues.pop()

        self.queues.sort(reverse=True)
        self._initialized  = True


    def start_bookkeeping(self, queues=None) :
        if queues is None :
            for q in self.edge2queue :
                q.keep_data = True
        else :
            for k in queues :
                self.edge2queue[k].keep_data = True


    def data(self, queues=None) :
        if queues is None :
            data = [[] for k in range(self.nE)]
            for k, q in enumerate(self.edge2queue) :
                for d in q.data.values() :
                    data[k].extend(d)
        else :
            data = [[] for k in range(len(queues))]
            for k, p in enumerate(queues) :
                for d in self.edge2queue[p].data.values() :
                    data[k].extend(d)

        return data


    def draw(self, out_size=(750, 750), output=None, update_colors=True, **kwargs) :
        """Draws the network. The coloring of the network corresponds to the 
        number of agents at each queue.

        Parameters
        ----------
        out_size : tuple (optional, the default is (750, 750).
            Specifies the size of canvas. See
            `graph-tool <http://graph-tool.skewed.de/static/doc/index.html>`_'s documentation.
        output : str (optional, the default is ``None``)
            Specifies the directory where the drawing is saved. The default is ``None``, 
            so the output is drawn using GraphViz.
        update_colors : bool (optional, the default is ``True``).
            Specifies whether all the colors are updated.
        **kwargs : 
            Any extra parameters to pass to :func:`~graph_tool.draw.graph_draw`.
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
                edge_text=self.g.ep['edge_text'],
                edge_font_size=self.g.ep['edge_font_size'],
                edge_text_distance=self.g.ep['edge_text_distance'],
                edge_text_parallel=self.g.ep['edge_text_parallel'],
                edge_text_color=self.g.ep['edge_text_color'],
                vertex_color=self.g.vp['vertex_color'],
                vertex_fill_color=self.g.vp['vertex_fill_color'],
                vertex_halo=self.g.vp['vertex_halo'],
                vertex_halo_color=self.g.vp['vertex_halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                vertex_text=self.g.vp['vertex_text'],
                vertex_text_position=self.g.vp['vertex_text_position'],
                vertex_font_size=self.g.vp['vertex_font_size'],
                vertex_size=self.g.vp['vertex_size'], **kwargs)


    def show_active(self) :
        """Draws the network, highlighting active queues.

        The colored vertices represent vertices that have at least one queue
        on an in-edge that is active. Dark edges represent queues that are active,
        light edges represent queues that are inactive.

        Notes
        -----
        The colors are defined by the class attribute ``colors``.
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
                self.g.vp['vertex_color'][v]      = self.edge2queue[ei].colors['vertex_pen']
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

                    color    = [ i * tmp / 0.9 for i in q.colors['vertex_normal'] ]
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
                vp['vertex_halo_color'][pv] = self.colors['halo_normal']
                vp['vertex_halo'][pv]       = False
            else :
                ep['edge_color'][pe] = q.current_color()
                nSy = 0
                cap = 0
                for vi in self.in_edges[self._prev_edge[1]] :
                    nSy += self.edge2queue[vi].nSystem
                    cap += self.edge2queue[vi].nServers

                tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

                color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
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

            color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
            color[3] = 1.0 - tmp
            vp['vertex_fill_color'][v] = color
            vp['vertex_color'][v]  = self.colors['vertex_color']


    def _add_departure(self, ei, agent, t) :
        q   = self.edge2queue[ei]
        qt  = q._time
        q._append_departure(agent, t)

        if qt == infty and q._time < infty :
            self.queues.append(q)

        self.queues.sort(reverse=True)


    def _add_arrival(self, ei, agent, t=None) :
        q   = self.edge2queue[ei]
        qt  = q._time
        if t is None :
            t = q._time + 1 if q._time < infty else self.queues[-1]._time + 1

        agent.set_arrival(t)
        q._add_arrival(agent)

        if qt == infty and q._time < infty :
            self.queues.append(q)

        self.queues.sort(reverse=True)


    def next_event_type(self) :
        """Returns whether the next next event is either an arrival or a departure."""
        return self.queues[-1].next_event_type()


    def _next_event(self, slow=True) :
        n   = len(self.queues)
        if n == 0 :
            self.t  = infty
            return

        q1  = self.queues.pop()
        q1t = q1._time
        e1  = q1.edge[2]

        event  = q1.next_event_type()
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
                        oneBisectSort(self.queues, q1, q2t, len(self.queues))
                    else :
                        if q1._time < q2._time :
                            self.queues.append(q1)
                        else :
                            self.queues.insert(0, q1)
                elif q2._time < q2t and e2 != e1 :
                    if n == 1 :
                        if q1._time < q2._time :
                            self.queues.append(q2)
                            self.queues.append(q1)
                        else :
                            self.queues.append(q1)
                            self.queues.append(q2)
                    else :
                        twoSort(self.queues, q1, q2, len(self.queues))
                else :
                    if n == 1 :
                        self.queues.append(q1)
                    else :
                        bisectSort(self.queues, q1, len(self.queues))
            else :
                if q2._time < q2t < infty :
                    if n > 2 :
                        oneSort(self.queues, q2t, len(self.queues))
                elif q2._time < q2t :
                    if n == 1 :
                        self.queues.append(q2)
                    else :
                        bisectSort(self.queues, q2, len(self.queues))

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
                    self.queues.append(q1)
                else :
                    bisectSort(self.queues, q1, len(self.queues) )

        if self._to_animate :
            self._window.graph.regenerate_surface(lazy=False)
            self._window.graph.queue_draw()
            return True


    def animate(self, out_size=(750, 750), **kwargs) :
        """Animates the network as it's simulating.

        Closing the window ends the animation.

        Parameters
        ----------
        out_size : tuple (optional, the default is (750, 750)).
            The size of the canvas for the animation.
        **kwargs :
            Any extra parameters are passed to :class:`~graph_tool.draw.GraphWindow`.

        Raises
        ------
        RuntimeError
            Will raise a :exc:`~RuntimeError` if the ``QueueNetwork`` has not been initialized. Call
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
                edge_text=self.g.ep['edge_text'],
                edge_font_size=self.g.ep['edge_font_size'],
                edge_text_distance=self.g.ep['edge_text_distance'],
                edge_text_parallel=self.g.ep['edge_text_parallel'],
                edge_text_color=self.g.ep['edge_text_color'],
                vertex_color=self.g.vp['vertex_color'],
                vertex_fill_color=self.g.vp['vertex_fill_color'],
                vertex_halo=self.g.vp['vertex_halo'],
                vertex_halo_color=self.g.vp['vertex_halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                vertex_text=self.g.vp['vertex_text'],
                vertex_text_position=self.g.vp['vertex_text_position'],
                vertex_font_size=self.g.vp['vertex_font_size'],
                vertex_size=self.g.vp['vertex_size'], **kwargs)

        cid = GObject.idle_add(self._next_event)
        self._window.connect("delete_event", Gtk.main_quit)
        self._window.show_all()
        Gtk.main()
        self._to_animate = False


    def simulate(self, n=None, t=25) :
        """Simulates the network forward.

        This method simulates the network forward for a specified amount of *system time* ``t``,
        or for a specific number of events ``n``.

        Parameters
        ----------
        n : int (optional)
            The number of events to simulate.
        t : float (optional, the default is 25)
            The amount of system time to simulate forward. If ``n`` is ``None`` then this parameter is used.

        Raises
        ------
        RuntimeError
            Will raise a :exc:`~RuntimeError` if the ``QueueNetwork`` has not been initialized. Call
            :meth:`~queueing_tool.network.QueueNetwork.initialize` before running.
        """
        if not self._initialized :
            raise RuntimeError("Network has not been initialized. Call '.initialize()' first.")
        if n is None :
            now = self.t
            while self.t < now + t :
                self._next_event(slow=False)
        elif isinstance(n, int) :
            for k in range(n) :
                self._next_event(slow=False)


    def reset_colors(self) :
        """Sets all edge and vertex colors to their default values."""
        for k, e in enumerate(self.g.edges()) :
            self.g.ep['edge_color'][e]    = self.edge2queue[k].colors['edge_normal']
        for k, v in enumerate(self.g.vertices()) :
            self.g.vp['vertex_fill_color'][v] = self.colors['vertex_normal']
            self.g.vp['vertex_halo_color'][v] = self.colors['halo_normal']
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
    
    """
    def __init__(self, g=None, q_classes=None, q_args=None, seed=None) :
        QueueNetwork.__init__(self, g, seed, calcpath)


    def __repr__(self) :
        return 'CongestionNetwork. # nodes: %s, edges: %s, agents: %s' % (self.nV, self.nE, np.sum(self.nAgents))


    def _next_event(self, slow=True) :
        q1  = self.queues.pop()
        q1t = q1._time
        e1  = q1.edge[2]

        event  = q1.next_event_type()
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
