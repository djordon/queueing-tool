import numpy            as np
import graph_tool.all   as gt
import copy

from numpy              import infty
from gi.repository      import Gtk, GObject
from .. generation      import generate_random_graph, prepare_graph

from .sorting           import oneBisectSort, bisectSort, oneSort, twoSort

# Garages changed to FCQ (finite capacity queue)
# each edge and vertex has an eType and vType respectively now. 
#   The default edge type is an arc type, which has given the number 0. 
#   FCQs are assumed to be of type 1.
#   Other queues, like the old destination queues, have a type of 2 or higher;
#       by default destination queues have a type of 2

# Need to check that queues is nonempty before running _next_event

class QueueNetwork :

    def __init__(self, g=None, nVertices=100, pDest=0.1, pFCQ=1, seed=None, graph_type="osm", calcpath=False) :
        self.nEvents      = 0
        self.t            = 0
        self.to_animate   = False
        self.initialized  = False
        self.agent_cap    = 100
        self.prev_edge    = None
        self.colors       =  {'edge_departure'   : [0, 0, 0, 1], 
                              'edge_normal'      : [0.7, 0.7, 0.7, 0.50],
                              'vertex' :     { 0 : [0.9, 0.9, 0.9, 1.0],        # normal
                                               1 : [0.9, 0.9, 0.9, 1.0],        # garages aka fcq
                                               2 : [0.9, 0.9, 0.9, 1.0],        # destination
                                               3 : [0.9, 0.9, 0.9, 1.0]},       # light
                              'vertex_pen' : { 0 : [0.0, 0.5, 1.0, 1.0],        # normal vertex
                                               1 : [0.133, 0.545, 0.133, 1.0],  # garages aka fcq
                                               2 : [0.282, 0.239, 0.545, 1.0],  # destination
                                               3 : [1.0, 0.135, 0.0, 1.0]},     # light
                              'halo_normal'      : [0, 0, 0, 0],
                              'halo_arrival'     : [0.1, 0.8, 0.8, 0.25],
                              'halo_departure'   : [0.9, 0.9, 0.9, 0.25],
                              'text_normal'      : [1, 1, 1, 0.5],
                              'bg_color'         : [1.0, 1.0, 1.0, 1.0]}

        if isinstance(seed, int) :
            np.random.seed(seed)
            gt.seed_rng(seed)

        if graph_type != 'copy' :
            if g == None :
                g     = generate_random_graph(nVertices, pDest, pFCQ)
                g, qs = prepare_graph(g, colors=self.colors, graph_type=None)
            elif isinstance(g, str) or isinstance(g, gt.Graph) :
                g, qs = prepare_graph(g, colors=self.colors, graph_type=graph_type)
            else :
                raise Exception("A proper graph (or graph location) was not given.")

            self.shortest_path = calculate_shortest_path(g) if calcpath else 0

            def edge_index(e) :
                return g.edge_index[e]

            self.adjacency  = [ [i for i in map(edge_index, list(v.out_edges()))] for v in g.vertices()]
            self.in_edges   = [ [i for i in map(edge_index, list(v.in_edges()))] for v in g.vertices() ]
            self.edge2queue = qs
            self.nAgents    = np.zeros( g.num_edges() )

            self.g  = g
            self.nV = g.num_vertices()
            self.nE = g.num_edges()


    def __repr__(self) :
        return 'QueueNetwork. # nodes: %s, edges: %s, agents: %s' % (self.nV, self.nE, np.sum(self.nAgents))


    def initialize(self, nActive=1, queues=None) :
        initialized = False
        if queues == None :
            queues = np.arange(self.nE)  
            np.random.shuffle(queues)
            queues = queues[:nActive]

        for ei in queues :
            self.edge2queue[ei].initialize()

        self.queues = [q for q in self.edge2queue]
        self.queues.sort()
        while self.queues[-1].time == infty :
            self.queues.pop()

        self.queues.sort(reverse=True)
        self._queues      = set([q.edge[2] for q in self.queues])
        self.initialized  = True


    def blocked(self) :
        ans = [q.lossed() for q in self.edge2queue]
        return ans


    def draw(self, outSize=(750, 750), output=None, update_colors=True) :
        if update_colors :
            self.update_graph_colors()

        kwargs = {'output_size': outSize , 'output' : output} if output != None else {'geometry' : outSize}

        ans = gt.graph_draw(self.g, pos=self.g.vp['pos'],
                    bg_color=self.colors['bg_color'],
                    edge_color=self.g.ep['edge_color'],
                    edge_control_points=self.g.ep['control'],
                    edge_marker_size=self.g.ep['arrow_width'],
                    edge_pen_width=self.g.ep['edge_width'],
                    edge_font_size=self.g.ep['edge_t_size'],
                    edge_text=self.g.ep['text'],
                    edge_text_distance=self.g.ep['edge_t_distance'],
                    edge_text_parallel=self.g.ep['edge_t_parallel'],
                    edge_text_color=self.g.ep['edge_t_color'],
                    vertex_color=self.g.vp['vertex_pen_color'],
                    vertex_fill_color=self.g.vp['vertex_color'],
                    vertex_halo=self.g.vp['halo'],
                    vertex_halo_color=self.g.vp['halo_color'],
                    vertex_halo_size=self.g.vp['vertex_halo_size'],
                    vertex_pen_width=self.g.vp['vertex_pen_width'],
                    vertex_text=self.g.vp['text'],
                    vertex_text_position=self.g.vp['vertex_t_pos'],
                    vertex_font_size=self.g.vp['vertex_t_size'],
                    vertex_size=self.g.vp['vertex_size'], **kwargs)


    def update_graph_colors(self) :
        ep  = self.g.ep
        vp  = self.g.vp
        do  = [True for v in range(self.nV)]
        for q in self.edge2queue :
            e = self.g.edge(q.edge[0], q.edge[1])
            v = self.g.vertex(q.edge[1])
            if q.edge[0] == q.edge[1] :
                vp['vertex_color'][v] = q.current_color()
                ep['edge_color'][e]   = q.current_color('edge')
            else :
                ep['edge_color'][e]   = q.current_color()
                if do[q.edge[1]] :
                    nSy = 0
                    cap = 0
                    for vi in self.in_edges[q.edge[1]] :
                        nSy += self.edge2queue[vi].nSystem
                        cap += self.edge2queue[vi].nServers

                    tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

                    color    = [ i * tmp / 0.9 for i in self.colors['vertex'][0] ]
                    color[3] = 1.0 - tmp
                    vp['vertex_color'][v] = color
                    do[q.edge[1]] = False


    def _update_graph_colors(self, ad, qedge) :
        e   = self.g.edge(qedge[0], qedge[1])
        v   = e.target()
        ep  = self.g.ep
        vp  = self.g.vp

        if self.prev_edge != None :
            pe  = self.g.edge(self.prev_edge[0], self.prev_edge[1])
            pv  = self.g.vertex(self.prev_edge[1])

            if pe.target() == pe.source() :
                ep['edge_color'][pe]   = self.edge2queue[self.prev_edge[2]].current_color('edge')
                vp['vertex_color'][pv] = self.edge2queue[self.prev_edge[2]].current_color()
                vp['halo_color'][pv]   = self.colors['halo_normal']
                vp['halo'][pv]  = False
            else :
                ep['edge_color'][pe] = self.edge2queue[self.prev_edge[2]].current_color()
                nSy = 0
                cap = 0
                for vi in self.in_edges[self.prev_edge[1]] :
                    nSy += self.edge2queue[vi].nSystem
                    cap += self.edge2queue[vi].nServers

                tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

                color    = [ i * tmp / 0.9 for i in self.colors['vertex'][0] ]
                color[3] = 1.0 - tmp
                vp['vertex_color'][v] = color

        if qedge[0] == qedge[1] :
            ep['edge_color'][e]   = self.edge2queue[qedge[2]].current_color('edge')
            vp['vertex_color'][v] = self.edge2queue[qedge[2]].current_color()
            vp['halo'][v]  = True

            if ad == 'arrival' :
                vp['halo_color'][v] = self.colors['halo_arrival']
            elif ad == 'departure' :
                vp['halo_color'][v] = self.colors['halo_departure']
        else :
            ep['edge_color'][e] = self.edge2queue[qedge[2]].current_color()
            nSy = 0
            cap = 0
            for vi in self.in_edges[qedge[1]] :
                nSy += self.edge2queue[vi].nSystem
                cap += self.edge2queue[vi].nServers

            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            color    = [ i * tmp / 0.9 for i in self.colors['vertex'][0] ]
            color[3] = 1.0 - tmp
            vp['vertex_color'][v] = color


    def append_departure(self, ei, agent, t) :
        q   = self.edge2queue[ei]
        q.append_departure(agent, t)

        if q.edge[2] not in self._queues and q.time < infty :
            self._queues.add(q.edge[2])
            self.queues.append(q)

        self.queues.sort(reverse=True)


    def add_arrival(self, ei, agent, t=None) :
        q = self.edge2queue[ei]
        if t == None :
            t = q.time + 1 if q.time < infty else self.queues[-1].time + 1

        agent.set_arrival(t)
        q._add_arrival(agent)

        if q.edge[2] not in self._queues and q.time < infty :
            self._queues.add(q.edge[2])
            self.queues.append(q)

        self.queues.sort(reverse=True)


    def next_event_type(self) :
        return self.queues[-1].next_event_type()


    def _next_event(self, slow=True) :
        q = self.queues.pop()
        t = q.time
        j = q.edge[2]

        event  = q.next_event_type()
        self.t = t

        if event == 2 : # This is a departure
            agent           = q.next_event()
            self.nAgents[j] = q.nTotal
            self.nEvents   += 1

            ei  = agent.desired_destination(self, q.edge) # expects the network, and current location
            q2  = self.edge2queue[ei]
            q2t = q2.time
            agent.set_arrival(t)

            q2._add_arrival(agent)
            self.nAgents[q2.edge[2]] = q2.nTotal

            if slow :
                self._update_graph_colors(ad='departure', qedge=q.edge)
                self.prev_edge = q.edge

            if q2.active and np.sum(self.nAgents) > self.agent_cap - 1 :
                q2.active = False

            if q2.departures[0].time <= q2.arrivals[0].time :
                print("WHOA! THIS NEEDS CHANGING! %s %s" % (q2.departures[0].time, q2.arrivals[0].time) )

            q2.next_event()

            if slow :
                self._update_graph_colors(ad='arrival', qedge=q2.edge)
                self.prev_edge = q2.edge

            if q.time < infty :
                if q2.edge[2] in self._queues :
                    if q2.time < q2t :
                        oneBisectSort(self.queues, q, q2t, len(self.queues))
                    else :
                        bisectSort(self.queues, q, len(self.queues))
                elif q2.time < infty :
                    self._queues.add(q2.edge[2])
                    twoSort(self.queues, q, q2, len(self.queues))
                else :
                    bisectSort(self.queues, q, len(self.queues))
            else :
                self._queues.remove(j)
                if q2.edge[2] in self._queues :
                    if q2.time < q2t :
                        oneSort(self.queues, q2t, len(self.queues))
                elif q2.time < infty :
                    self._queues.add(q2.edge[2])
                    bisectSort(self.queues, q2, len(self.queues))

        elif event == 1 : # This is an arrival
            if q.active and np.sum(self.nAgents) > self.agent_cap - 1 :
                q.active = False

            q.next_event()
            self.nAgents[j] = q.nTotal
            self.nEvents   += 1

            if slow :
                self._update_graph_colors(ad='arrival', qedge=q.edge)
                self.prev_edge  = q.edge

            if q.time < infty :
                bisectSort(self.queues, q, len(self.queues) )

        if self.to_animate :
            self.win.graph.regenerate_surface(lazy=False)
            self.win.graph.queue_draw()
            return True


    def animation(self, outSize=(750, 750)) :
        if not self.initialized :
            raise Exception("Network has not been initialized. Call 'initialize()' first.")

        self.to_animate = True
        self.update_graph_colors()
        self.win = gt.GraphWindow(self.g, self.g.vp['pos'], geometry=outSize,
                bg_color=self.colors['bg_color'],
                edge_color=self.g.ep['edge_color'],
                edge_control_points=self.g.ep['control'],
                edge_marker_size=self.g.ep['arrow_width'],
                edge_pen_width=self.g.ep['edge_width'],
                edge_text=self.g.ep['text'],
                edge_font_size=self.g.ep['edge_t_size'],
                edge_text_distance=self.g.ep['edge_t_distance'],
                edge_text_parallel=self.g.ep['edge_t_parallel'],
                edge_text_color=self.g.ep['edge_t_color'],
                vertex_color=self.g.vp['vertex_pen_color'],
                vertex_fill_color=self.g.vp['vertex_color'],
                vertex_halo=self.g.vp['halo'],
                vertex_halo_color=self.g.vp['halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                vertex_text=self.g.vp['text'],
                vertex_text_position=self.g.vp['vertex_t_pos'],
                vertex_font_size=self.g.vp['vertex_t_size'],
                vertex_size=self.g.vp['vertex_size'])

        cid = GObject.idle_add(self._next_event)
        self.win.connect("delete_event", Gtk.main_quit)
        self.win.show_all()
        Gtk.main()
        self.to_animate = False


    def simulate(self, t=25, n=None) :
        if not self.initialized :
            raise Exception("Network has not been initialized. Call 'initialize()' first.")
        if n == None :
            now = self.t
            while self.t < now + t :
                self._next_event(slow=False)
        elif isinstance(n, int) :
            for k in range(n) :
                self._next_event(slow=False)


    def reset_colors(self) :
        for e in self.g.edges() :
            self.g.ep['edge_color'][e]    = self.colors['edge_normal']
        for v in self.g.vertices() :
            self.g.vp['vertex_color'][v]  = self.colors['vertex'][0]
            self.g.vp['halo_color'][v]    = self.colors['halo_normal']
            self.g.vp['halo'][v]          = False
            self.g.vp['text'][v]          = ''


    def clear(self) :
        self.t            = 0
        self.nEvents      = 0
        self.nAgents      = np.zeros(self.nE)
        self.to_animate   = False
        self.initialized  = False
        self.reset_colors()
        for q in self.edge2queue :
            q.clear()


    def copy(self) :
        net               = QueueNetwork(graph_type='copy')
        net.g             = self.g.copy()
        net.t             = copy.copy(self.t)
        net.agent_cap     = copy.copy(self.agent_cap)
        net.nV            = copy.copy(self.nV)
        net.nE            = copy.copy(self.nE)
        net.nAgents       = copy.copy(self.nAgents)
        net.nEvents       = copy.copy(self.nEvents)
        net.initialized   = copy.copy(self.initialized)
        net.prev_edge     = copy.copy(self.prev_edge)
        net.shortest_path = copy.copy(self.shortest_path)
        net.to_animate    = copy.copy(self.to_animate)
        net._queues       = copy.copy(self._queues)
        net.colors        = copy.deepcopy(self.colors)
        net.adjacency     = copy.deepcopy(self.adjacency)
        net.in_edges      = copy.deepcopy(self.in_edges)
        net.edge2queue    = copy.deepcopy(self.edge2queue)
        net.queues        = [q for q in net.edge2queue if q.edge[2] in net._queues]
        net.queues.sort(reverse=True)
        return net


def calculate_shortest_path(g) :
    nV  = g.num_vertices()
    vertex_props  = set()
    shortest_path = np.ones( (nV, nV), int)

    for key in g.vertex_properties.keys() :
        vertex_props = vertex_props.union([key])

    if 'shortest_path' in vertex_props :
        for v in g.vertices() :
            shortest_path[int(v), :] = g.vp['shortest_path'][v].a

    else :
        spath = np.ones( (nV, nV), int) * -1

        for v in g.vertices() :
            vi  = int(v)
            for u in g.vertices() :
                ui  = int(u)
                if ui == vi or spath[vi, ui] != -1 :
                    continue

                path  = gt.shortest_path(g, v, u, weights=g.ep['edge_length'])[0]
                path  = [int(z) for z in path]
                spath[path[:-1], path[-1]] = path[1:]

                for j in range(1,len(path)-1) :
                    pa  = path[:-j]
                    spath[pa[:-1], pa[-1]] = pa[1:]

                if not g.is_directed() :
                    path.reverse()
                    spath[path[:-1], path[-1]] = path[1:]

                    for j in range(1, len(path)-1) :
                        pa  = path[:-j]
                        spath[pa[:-1], pa[-1]] = pa[1:]

            shortest_path[vi, :] = spath[vi, :]

        r = np.arange(nV)
        shortest_path[r, r] = r

    return shortest_path
