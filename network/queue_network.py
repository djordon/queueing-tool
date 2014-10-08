import numpy                as np
import graph_tool.all       as gt
import time
import copy

from numpy                  import ones, zeros, array, arange, logical_and, infty
from heapq                  import heappush, heappop, heapify
from gi.repository          import Gtk, GObject
from .. agents.queue_agents import LearningAgent, ResourceAgent

from .. servers import queue_server as qs

from sorting import insertionsort_special

# Garages changed to FCQ (finite capacity queue)
# each edge and vertex has an eType and vType respectively now. 
#   The default edge type is an arc type, which has given the number 0. 
#   FCQs are assumed to be of type 1.
#   Other queues, like the old destination queues, have a type of 2 or higher;
#       by default destination queues have a type of 2

class QueueNetwork :

    def __init__(self, g=None, nVertices=100, pDest=None, pFCQ=None, seed=None, graph_type="osm", calcpath=False) :
        self.nEvents    = 0
        self.t          = 0
        self.to_animate = False
        self.undirected = False
        self.nAgents    = [0]
        self.agent_cap  = 100
        self.fcq_count  = 0
        self.prev_issn  = None
        self.colors     =  {'edge_departure'   : [0, 0, 0, 1], 
                            'edge_normal'      : array([0.7, 0.7, 0.7, 0.50]),
                            'vertex' :     { 0 : [0.9, 0.9, 0.9, 1.0],          # normal
                                             1 : [0.9, 0.9, 0.9, 1.0],          # garages aka fcq
                                             2 : [0.9, 0.9, 0.9, 1.0],          # destination
                                             3 : [0.9, 0.9, 0.9, 1.0]},         # light
                            'vertex_pen' : { 0 : [0.0, 0.5, 1.0, 1.0],          # normal vertex
                                             1 : [0.133, 0.545, 0.133, 1.0],    # garages aka fcq
                                             2 : [0.282, 0.239, 0.545, 1.0],    # destination
                                             3 : [1.0, 0.135, 0.0, 1.0]},       # light
                            'vertex_arrival'   : [0.4, 0.8, 0.4, 0.75], 
                            'vertex_departure' : [0.4, 0.8, 0.4, 0.75],
                            'halo_normal'      : [0, 0, 0, 0],
                            'halo_arrival'     : [0.1, 0.8, 0.8, 0.25],
                            'halo_departure'   : [0.9, 0.9, 0.9, 0.25],
                            'text_normal'      : [1, 1, 1, 0.5],
                            'bg_color'         : [1.95, 1.95, 1.95, 1.0]}


        if isinstance(seed, int) :
            np.random.seed(seed)
            gt.seed_rng(seed)

        if g == None :
            self.active_graph(nVertices, pDest, pFCQ)
        elif isinstance(g, str) or isinstance(g, gt.Graph) :
            self.set_graph(g, graph_type, calcpath)
        else :
            pass


    def set_graph(self, g, graph_type=None, calc_shortest_path=False) :
        if isinstance(g, str) :
            g = gt.load_graph(g, fmt='xml')
        elif not isinstance(g, gt.Graph) :
            raise Exception("Need to supply a graph when initializing")

        g.reindex_edges()
        vertex_t_color    = g.new_vertex_property("vector<double>")
        vertex_pen_color  = g.new_vertex_property("vector<double>")
        vertex_color      = g.new_vertex_property("vector<double>")
        halo_color        = g.new_vertex_property("vector<double>")
        vertex_t_pos      = g.new_vertex_property("double")
        vertex_t_size     = g.new_vertex_property("double")
        halo              = g.new_vertex_property("bool")
        state             = g.new_vertex_property("int")
        vertex_type       = g.new_vertex_property("int")
        vertex_halo_size  = g.new_vertex_property("double")
        vertex_pen_width  = g.new_vertex_property("double")
        vertex_size       = g.new_vertex_property("double")

        control           = g.new_edge_property("vector<double>")
        edge_color        = g.new_edge_property("vector<double>")
        edge_t_color      = g.new_edge_property("vector<double>")
        edge_width        = g.new_edge_property("double")
        arrow_width       = g.new_edge_property("double")
        edge_length       = g.new_edge_property("double")
        edge_times        = g.new_edge_property("double")
        edge_t_size       = g.new_edge_property("double")
        edge_t_distance   = g.new_edge_property("double")
        edge_t_parallel   = g.new_edge_property("bool")
        edge_state        = g.new_edge_property("int")
        queues            = g.new_edge_property("python::object")

        vertex_props = set()
        for key in g.vertex_properties.keys() :
            vertex_props = vertex_props.union([key])

        edge_props = set()
        for key in g.edge_properties.keys() :
            edge_props = edge_props.union([key])

        has_garage  = 'garage' in vertex_props
        has_destin  = 'destination' in vertex_props
        has_light   = 'light' in vertex_props
        has_egarage = 'garage' in edge_props
        has_edestin = 'destination' in edge_props
        has_elight  = 'light' in edge_props

        if graph_type == 'osm' :
            vType   = g.new_vertex_property("int")
            eType   = g.new_edge_property("int")
            fcq_count   = 0
            dest_count  = 0
            for v in g.vertices() :
                if has_garage and g.vp['garage'][v] :
                    vType[v]    = 1
                    fcq_count  += 1
                if has_destin and g.vp['destination'][v] :
                    vType[v]    = 2
                    dest_count += 1
                if has_light and g.vp['light'][v] :
                    vType[v]    = 3

            for e in g.edges() :
                if has_egarage and g.ep['garage'][e] :
                    eType[e]    = 1
                if has_edestin and g.ep['destination'][e] :
                    eType[e]    = 2
                if has_elight  and g.ep['light'][e] :
                    eType[e]    = 3

            g.vp['vType']   = vType
            g.ep['eType']   = eType
            self.dest_count = dest_count
            self.fcq_count  = fcq_count


        if 'pos' not in vertex_props :
            g.vp['pos'] = gt.sfdp_layout(g, epsilon=1e-2, cooling_step=0.95)

        self.nV       = g.num_vertices()
        self.nE       = g.num_edges()
        self.nAgents  = [0 for k in range(self.nE)]
        self.edges    = [e for e in g.edges()]

        has_length    = 'edge_length' in edge_props
        has_lanes     = 'lanes' in vertex_props
        has_cap       = 'cap' in vertex_props

        for e in g.edges() :
            qissn = (int(e.source()), int(e.target()), g.edge_index[e])
            if g.ep['eType'][e] == 1 : #qs.LossQueue(cap, issn=qissn, net_size=self.nE)
                cap       = g.vp['cap'][e.target()] if has_cap else 4
                #if qissn[0] == qissn[1] :
                #    queues[e] = qs.ResourceQueue(20, issn=qissn, net_size=self.nE)
                #else :
                queues[e] = qs.LossQueue(20, issn=qissn, net_size=self.nE)#, AgentClass=ResourceAgent)
                edge_length[e]  = g.ep['edge_length'][e] if has_length else 1 ## Needs editing
            else : #qs.QueueServer(lanes, issn=qissn, net_size=self.nE)
                lanes     = g.vp['lanes'][e.target()] if has_lanes else 8
                lanes     = lanes if lanes > 10 else max([lanes // 2, 1])
                #if qissn[0] == qissn[1] :
                #    queues[e] = qs.ResourceQueue(20, issn=qissn, net_size=self.nE)
                #else :
                queues[e] = qs.QueueServer(20, issn=qissn, net_size=self.nE)#, AgentClass=ResourceAgent)
                edge_length[e]  = g.ep['edge_length'][e] if has_length else 1 ## Needs editing

            if qissn[0] == qissn[1] :
                edge_color[e]   = [0, 0, 0, 0]
            else :
                control[e]      = [0, 0, 0, 0]
                edge_color[e]   = self.colors['edge_normal']
            edge_width[e]   = 1.25
            arrow_width[e]  = 8
            edge_state[e]   = 0
            edge_times[e]   = 1
            edge_t_size[e]  = 8
            edge_t_color[e] = [0.0, 0.0, 0.0, 1.0]
            edge_t_distance[e]  = 8
            edge_t_parallel[e]  = False

        for v in g.vertices() :
            e = g.edge(v, v)
            if isinstance(e, gt.Edge) :
                vertex_pen_color[v] = queues[e].color('pen')
                vertex_color[v]     = queues[e].color()
            vertex_t_color[v]   = self.colors['text_normal']
            halo_color[v]       = self.colors['halo_normal']
            vertex_t_pos[v]     = 0 * np.pi / 4
            vertex_t_size[v]    = 8
            vertex_halo_size[v] = 1.3
            vertex_pen_width[v] = 0.8
            vertex_size[v]      = 7
            halo[v]             = False
            state[v]            = 0

        if calc_shortest_path and 'shortest_path' not in vertex_props :
            shortest_path = np.ones( (self.nV, self.nV), int)
            spath         = np.ones( (self.nV, self.nV), int) * -1

            for v in g.vertices() :
                vi  = int(v)
                for u in g.vertices() :
                    ui  = int(u)
                    if ui == vi or spath[vi, ui] != -1 :
                        continue

                    path  = gt.shortest_path(g, v, u, weights=edge_length)[0]
                    path  = [int(z) for z in path]
                    spath[path[:-1], path[-1]] = path[1:]

                    for j in range(1,len(path)-1) :
                        pa  = path[:-j]
                        spath[pa[:-1], pa[-1]] = pa[1:]

                    if self.undirected :
                        path.reverse()
                        spath[path[:-1], path[-1]] = path[1:]

                        for j in range(1, len(path)-1) :
                            pa  = path[:-j]
                            spath[pa[:-1], pa[-1]] = pa[1:]

                shortest_path[vi, :] = spath[vi, :]

            self.shortest_path = shortest_path

        node_dict = {'fcq' : [], 'des' : [], 'arc' : [], 'dest_arc' : []}
        for v in g.vertices() :
            if g.vp['vType'][v] == 1 :
                node_dict['fcq'].append(int(v))
            elif g.vp['vType'][v] == 2 :
                node_dict['des'].append(int(v))
            else :
                node_dict['arc'].append(int(v))
        node_dict['dest_arc'] = copy.copy(node_dict['des'])
        node_dict['dest_arc'].extend(node_dict['arc'])
        self.node_index = node_dict

        g.vp['vertex_t_color']   = vertex_t_color
        g.vp['vertex_pen_color'] = vertex_pen_color
        g.vp['vertex_color']     = vertex_color
        g.vp['halo_color']       = halo_color
        g.vp['vertex_t_pos']     = vertex_t_pos
        g.vp['vertex_t_size']    = vertex_t_size
        g.vp['halo']             = halo
        g.vp['state']            = state
        g.vp['vertex_type']      = vertex_type 
        g.vp['vertex_halo_size'] = vertex_halo_size
        g.vp['vertex_pen_width'] = vertex_pen_width
        g.vp['vertex_size']      = vertex_size

        g.ep['edge_t_size']      = edge_t_size
        g.ep['edge_t_distance']  = edge_t_distance
        g.ep['edge_t_parallel']  = edge_t_parallel
        g.ep['edge_t_color']     = edge_t_color
        g.ep['state']            = edge_state
        g.ep['control']          = control
        g.ep['edge_color']       = edge_color
        g.ep['edge_width']       = edge_width
        g.ep['edge_length']      = edge_length
        g.ep['arrow_width']      = arrow_width
        g.ep['edge_times']       = edge_times
        g.ep['queues']           = queues

        self.g  = g
        self.queue_heap = [self.g.ep['queues'][e] for e in self.g.edges()]
        #heapify(self.queue_heap)


    def active_graph(self, nVertices=250, pDest=None, pGarage=None) :

        points  = np.random.random((nVertices, 2)) * 2
        radii   = [(4 + k) / 200 for k in range(560)]

        for r in radii :
            g, pos  = gt.geometric_graph(points, r, [(0,2), (0,2)])
            comp, a = gt.label_components(g)
            if max(comp.a) == 0 :
                break

        pos       = gt.sfdp_layout(g, epsilon=1e-2, cooling_step=0.95)
        pos_array = array([pos[v] for v in g.vertices()])
        pos_array = pos_array / (100*(np.max(pos_array,0) - np.min(pos_array,0)))

        for v in g.vertices() :
            pos[v]  = pos_array[int(v), :]

        g.vp['pos']     = pos
        self.undirected = True

        g = self.set_special_nodes(g, pDest, pGarage)
        self.set_graph(g)



    def set_special_nodes(self, g, pDest=None, pFCQ=None) :
        if pDest == None :
            pDest = 0.1
        if pFCQ == None :
            pFCQ = 1
        pDest = pDest * 100

        def calculate_distance(latlon1, latlon2) :
            lat1, lon1  = latlon1
            lat2, lon2  = latlon2
            R     = 6371          # radius of the earth in kilometers
            dlon  = lon2 - lon1
            dlat  = lat2 - lat1
            a     = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
            c     = 2 * np.pi * R * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) / 180
            return c

        pagerank    = gt.pagerank(g)
        tmp         = np.array(pagerank.a)
        tmp.sort()
        nDests      = int(np.ceil(g.num_vertices()/pDest))
        dests       = np.where(pagerank.a >= tmp[-nDests])[0]

        dest_pos    = array([g.vp['pos'][g.vertex(k)] for k in dests])
        nFCQ        = int(pFCQ * np.size(dests))
        min_g_dist  = ones(nFCQ) * infty
        ind_g_dist  = ones(nFCQ, int)

        r, theta    = np.random.random(nFCQ) / 500, np.random.random(nFCQ) * 360
        xy_pos      = array([r * np.cos(theta), r * np.sin(theta)]).transpose()
        g_pos       = xy_pos + dest_pos[ np.array( np.mod(np.arange(nFCQ), nDests), int) ]

        for v in g.vertices() :
            if int(v) not in dests :
                tmp = array([calculate_distance(g.vp['pos'][v], g_pos[k, :]) for k in range(nFCQ)])
                min_g_dist = np.min((tmp, min_g_dist), 0)
                ind_g_dist[min_g_dist == tmp] = int(v)

        ind_g_dist  = np.unique(ind_g_dist)
        fcqs        = ind_g_dist[:min( (nFCQ, len(ind_g_dist)) )]

        if not g.is_directed() :
            g.set_directed(True)

            g2  = g.copy()
            for e in g2.edges() :
                e1  = g.add_edge(source=int(e.target()), target=int(e.source()))

        vType   = g.new_vertex_property("int")
        eType   = g.new_edge_property("int")
        elength = g.new_edge_property("double")

        for v in g.vertices() :
            if int(v) in dests :
                vType[v] = 2
            if int(v) in fcqs :
                vType[v] = 1

        for e in g.edges() :
            latlon1     = g.vp['pos'][e.target()]
            latlon2     = g.vp['pos'][e.source()]
            elength[e]  = np.round(calculate_distance(latlon1, latlon2), 3)

        for v in g.vertices() :
            e = g.add_edge(source=v, target=v)
            if vType[v] == 1 :
                eType[e] = 1

        g.vp['vType'] = vType
        g.ep['eType'] = eType
        return g


    def initialize(self, nActive=1, queues=None) :
        self.g.reindex_edges()
        initialized = False
        if queues == None :
            queues = np.arange(self.nV)  
            np.random.shuffle(queues)
            queues = queues[:nActive]

        for vi in queues :
            for e in self.g.vertex(vi).out_edges() :
                if e.source() == e.target() :
                    self.g.ep['queues'][e].initialize()
                    initialized = True
                    break
            if not initialized :
                self.g.ep['queues'][e].initialize()

        self.queue_heap.sort()


    def __repr__(self) :
        return 'QueueNetwork. # nodes: %s, edges: %s, agents: %s' % (self.nV, self.nE, sum(self.nAgents))


    def agent_stats(self) :
        ans     = zeros(7)
        rested  = 0
        spaces  = 0
        for e in self.g.edges() :
            q        = self.g.ep['queues'][e]
            ans[:4] += q.travel_stats()
            if isinstance(q, qs.LossQueue) :
                ans[4] += q.lossed()
                rested += q.nSystem
                spaces += q.nServers
        ans[5] = self.fcq_count
        ans[6] = rested/spaces
        return ans
            

    def information_stats(self) :
        real  = zeros(self.nE)
        data  = zeros(self.nE)
        a     = arange(self.nE)

        for e in self.g.edges() :
            q   = self.g.ep['queues'][e]
            k   = self.g.edge_index[e]
            tmp = q.net_data[:,2]
            ind = logical_and(a!=k, tmp != -1)
            real[q.issn[2]] = q.nSystem / q.nServers    # q.issn[2] is the edge_index for that edge
            data[ind]      += (tmp[ind] - real[ind])**2

        g_index = array(self.node_index['fcq'])
        d_index = array(self.node_index['destination'])
        r_index = array(self.node_index['arc'])
        data   /= (self.nE - 1)

        return array( ( (np.mean(data[r_index]), np.std(data[r_index])), \
                        (np.mean(data[d_index]), np.std(data[d_index])), \
                        (np.mean(data[g_index]), np.std(data[g_index])), \
                        (np.mean(data),          np.std(data)) ) )


    def _information_stats(self) :
        g   = self.node_index['fcq'][0]
        d   = self.node_index['des'][0]
        r   = self.node_index['arc'][1]

        ans = zeros( (3,2) )
        ct  = 0

        for q in g,d,r :
            data  = infty * ones(self.nE)
            a     = np.arange(self.nE)
            a     = a[a!=q]
            qnew  = self.g.ep['queues'][self.edges[q]]
            real  = qnew.nSystem / qnew.nServers

            for k in a :
                tmpdata = self.g.ep['queues'][self.edges[k]].net_data[q,2]
                if tmpdata == -1 : 
                    continue
                data[k] = (tmpdata - real)
         
            data      = data[ data != infty ]
            ans[ct, :] = np.mean( data ), np.std( data )
            ct       += 1

        return ans


    def blocked(self) :
        ans = [q.lossed() for q in self.queue_heap]
        return ans


    def reset_colors(self) :
        for e in self.g.edges() :
            self.g.ep['edge_color'][e]    = self.colors['edge_normal']
        for v in self.g.vertices() :
            self.g.vp['vertex_color'][v]  = self.colors['vertex_normal']
            self.g.vp['halo_color'][v]    = self.colors['halo_normal']
            self.g.vp['halo'][v]          = False
            self.g.vp['state'][v]         = 0


    def draw(self, file_name=None, update_colors=True) :
        if update_colors :
            self.update_graph_colors()

        if file_name == None :
            ans = gt.graph_draw(self.g, self.g.vp['pos'], geometry=(750, 750),
                        bg_color=self.colors['bg_color'],
                        edge_color=self.g.ep['edge_color'],
                        edge_control_points=self.g.ep['control'],
                        edge_marker_size=self.g.ep['arrow_width'],
                        edge_pen_width=self.g.ep['edge_width'],
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
                        #vertex_text=self.g.vp['state'],
                        vertex_text_position=self.g.vp['vertex_t_pos'],
                        vertex_font_size=self.g.vp['vertex_t_size'],
                        vertex_size=self.g.vp['vertex_size'])
        else :
            ans = gt.graph_draw(self.g, self.g.vp['pos'], geometry=(750, 750), output=file_name,
                        bg_color=self.colors['bg_color'],
                        edge_color=self.g.ep['edge_color'],
                        edge_control_points=self.g.ep['control'],
                        edge_marker_size=self.g.ep['arrow_width'],
                        edge_pen_width=self.g.ep['edge_width'],
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
                        #vertex_text=self.g.vp['state'],
                        vertex_text_position=self.g.vp['vertex_t_pos'],
                        vertex_font_size=self.g.vp['vertex_t_size'],
                        vertex_size=self.g.vp['vertex_size'])




    def update_graph_colors(self) :
        ep  = self.g.ep
        vp  = self.g.vp
        for e in self.g.edges() :
            nSy = ep['queues'][e].nSystem
            ep['state'][e] = nSy

            if e.target() == e.source() :
                v = e.target()
                vp['state'][v] = nSy
                vp['vertex_color'][v] = ep['queues'][e].color()
                ep['edge_color'][e]   = ep['queues'][e].color('edge')
            else :
                ep['edge_color'][e]   = ep['queues'][e].color()



    def _update_graph_colors(self, ad, qissn) :
        e   = self.g.edge(qissn[0], qissn[1])
        v   = e.target()
        ep  = self.g.ep
        vp  = self.g.vp

        if self.prev_issn != None :
            pe  = self.g.edge(self.prev_issn[0], self.prev_issn[1])
            pv  = self.g.vertex(self.prev_issn[1])
            nSy = ep['queues'][pe].nSystem

            ep['state'][pe] = nSy

            if pe.target() == pe.source() :
                ep['edge_color'][pe]   = ep['queues'][pe].color('edge')
                vp['vertex_color'][pv] = ep['queues'][pe].color()
                vp['halo_color'][pv]   = self.colors['halo_normal']
                vp['halo'][pv]  = False
                vp['state'][pv] = nSy
            else :
                ep['edge_color'][pe] = ep['queues'][e].color()

        nSy = ep['queues'][e].nSystem
        ep['state'][e] = nSy

        if e.target() == e.source() :
            ep['edge_color'][e]   = ep['queues'][e].color('edge')
            vp['vertex_color'][v] = ep['queues'][e].color()
            vp['halo'][v]  = True
            vp['state'][v] = nSy

            if ad == 'arrival' :
                vp['halo_color'][v] = self.colors['halo_arrival']
            elif ad == 'departure' :
                vp['halo_color'][v] = self.colors['halo_departure']
        else :
            ep['edge_color'][e] = ep['queues'][e].color()



    def next_event_type(self) :
        self.queue_heap.sort()
        return self.queue_heap[0].next_event_type()


    def next_event(self, Slow=False, STOP_LEARNER=False) :
        insertionsort_special( self.queue_heap)
        q = self.queue_heap[0]
        #qissn = self.queue_heap[0][1:]#.issn
        #q = self.g.ep['queues'][self.g.edge(qissn[0], qissn[1])]
        t = q.next_time
        j = q.issn[2]

        if t == infty :
            self.t == infty
            return

        event_type  = q.next_event_type()

        if STOP_LEARNER and event_type == "departure" :
            if isinstance(q.departures[0], qs.LearningAgent) :
                return "STOP"

        self.t = t

        if event_type == "departure" :
            agent           = q.next_event()
            self.nAgents[j] = q.nTotal

            e   = agent.desired_destination(self, q.issn) # expects the network, and current location
            q2  = self.g.ep['queues'][e]
            agent.set_arrival(t)

            q2._add_arrival(agent)
            self.nAgents[q2.issn[2]] = q2.nTotal

            if Slow :
                self._update_graph_colors(ad='departure', qissn=q.issn)
                self.prev_issn = q.issn

            if q2.active and sum(self.nAgents) > self.agent_cap - 1 :
                q2.active = False

            if q2.departures[0].time <= q2.arrivals[0].time :
                print("WHOA! THIS NEEDS CHANGING!")

            q2.next_event()
            self.nEvents += 2

            if Slow :
                self._update_graph_colors(ad='arrival', qissn=q2.issn)
                self.prev_issn = q2.issn

        else :
            if q.active and sum(self.nAgents) > self.agent_cap - 1 :
                q.active = False

            q.next_event()
            self.nEvents   += 1
            self.nAgents[j] = q.nTotal

            if Slow :
                self._update_graph(ad='arrival', qissn=q.issn)
                self.prev_issn  = q.issn

        if self.to_animate :
            #future = self.queue_heap[1].next_time
            #time.sleep( future - self.t )
            self.win.graph.regenerate_surface(lazy=False)
            self.win.graph.queue_draw()
            return True


    def animation(self) :
        self.to_animate = True
        self.win = gt.GraphWindow(self.g, self.g.vp['pos'], geometry=(750, 750),
                bg_color=self.colors['bg_color'],
                edge_color=self.g.ep['edge_color'],
                edge_control_points=self.g.ep['control'],
                edge_marker_size=self.g.ep['arrow_width'],
                edge_pen_width=self.g.ep['edge_width'],
                #edge_text=self.g.ep['state'],
                edge_font_size=self.g.ep['edge_t_size'],
                #edge_text_distance=self.g.ep['edge_t_distance'],
                #edge_text_parallel=self.g.ep['edge_t_parallel'],
                #edge_text_color=self.g.ep['edge_t_color'],
                vertex_color=self.g.vp['vertex_pen_color'],
                vertex_fill_color=self.g.vp['vertex_color'],
                vertex_halo=self.g.vp['halo'],
                vertex_halo_color=self.g.vp['halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                #vertex_text=self.g.vp['state'],
                vertex_text_position=self.g.vp['vertex_t_pos'],
                vertex_font_size=self.g.vp['vertex_t_size'],
                vertex_size=self.g.vp['vertex_size'])

        cid = GObject.idle_add(self.next_event)
        self.win.connect("delete_event", Gtk.main_quit)
        self.win.show_all()
        Gtk.main()
        self.to_animate = False


    def simulate(self, T=25) :
        now = self.t
        while self.t < now + T :
            self.next_event(Slow=False)


    def reset(self) :
        self.t          = 0
        self.nEvents    = 0
        self.nAgents    = [0 for k in range( self.nE )]
        self.to_animate = False
        self.reset_colors()
        for e in self.g.edges() :
            self.g.ep['queues'][e].reset()


    def copy(self) :
        net               = QueueNetwork(0)
        net.t             = copy.deepcopy(self.t)
        net.agent_cap     = copy.deepcopy(self.agent_cap)
        net.nV            = copy.deepcopy(self.nV)
        net.nE            = copy.deepcopy(self.nE)
        net.to_animate    = False
        net.g             = self.g.copy()
        net.fcq_count     = copy.deepcopy(self.fcq_count)
        net.dest_count    = copy.deepcopy(self.dest_count)
        net.shortest_path = copy.deepcopy(self.shortest_path)
        net.prev_issn     = copy.deepcopy(self.prev_issn)
        net.nEvents       = copy.deepcopy(self.nEvents)
        net.nAgents       = copy.deepcopy(self.nAgents)
        net.colors        = copy.deepcopy(self.colors)
        net.edges         = [e for e in net.g.edges()]
        queues            = net.g.new_edge_property("python::object")

        for e in net.g.edges() :
            queues[e]   = copy.deepcopy( net.g.ep['queues'][e] )

        net.g.ep['queues']  = queues
        net.queue_heap      = [net.g.ep['queues'][e] for e in net.g.edges()]
        heapify(net.queue_heap)
        return net
