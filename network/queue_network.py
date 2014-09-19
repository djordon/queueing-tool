import numpy                as np
import graph_tool.all       as gt
import time
import copy

from numpy                  import ones, zeros, array, arange, logical_and, infty
from heapq                  import heappush, heappop, heapify
from gi.repository          import Gtk, GObject
from .. agents.queue_agents import Agent, SmartAgent, LearningAgent, RandomAgent

from .. servers import queue_server as qs

class QueueNetwork :

    def __init__(self, g=None, nVertices=250, pDest=None, pGarage=None, seed=None, graph_type="normal") :
        self.nEvents    = 0
        self.t          = 0
        self.to_animate = False
        self.undirected = False
        self.nAgents    = [0]
        self.agent_cap  = 100
        self.colors     =  {'edge_departure'     : [0, 0, 0, 1], 
                            'edge_normal'        : [0.339, 0.3063, 0.3170, 0.290],
                            'vertex_normal'      : [0.0, 0.5, 1.0, 0.85],
                            'vertex_light'       : [1.0, 0.135, 0.0, 1.0],
                            'vertex_garage'      : [0.133, 0.545, 0.133, 1.0],
                            'vertex_destination' : [0.282, 0.239, 0.545, 1.0],
                            'vertex_arrival'     : [0.4, 0.8, 0.4, 0.75], 
                            'vertex_departure'   : [0.4, 0.8, 0.4, 0.75],
                            'halo_normal'        : [0, 0, 0, 0],
                            'halo_arrival'       : [0.1, 0.8, 0.8, 0.25],
                            'halo_departure'     : [0.9, 0.9, 0.9, 0.25],
                            'text_normal'        : [1, 1, 1, 0.5],
                            'bg_color'           : [1.95, 1.95, 1.95, 1.0]}


        if isinstance(seed, int) :
            np.random.seed(seed)
            gt.seed_rng(seed)

        if g == None :
            self.active_graph(nVertices, pDest, pGarage)
        elif isinstance(g, str) :
            self.set_graph(g)
        elif isinstance(g, gt.Graph) :
            self.set_graph(g)
        else :
            pass


    def set_graph(self, g) :
        if isinstance(g, str) :
            g = gt.load_graph(g, fmt='xml')
        elif not isinstance(g, gt.Graph) :
            raise Exception("Need to supply a graph when initializing")

        vertex_t_color    = g.new_vertex_property("vector<double>")
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

        garage_count      = g.new_graph_property("int")
        dest_count        = g.new_graph_property("int")
        node_index        = g.new_graph_property("python::object")

        dest_count[g]     = sum(g.vp['destination'].a)
        garage_count[g]   = sum(g.vp['garage'].a)

        vertex_props = set()
        for key in g.vertex_properties.keys() :
            vertex_props = vertex_props.union([key])

        HAS_LIGHTS  = 'light' in vertex_props

        for v in g.vertices() :
            if g.vp['destination'][v] :
                vertex_color[v] = self.colors['vertex_destination']
            elif g.vp['garage'][v] :
                vertex_color[v] = self.colors['vertex_garage']
            else :
                vertex_color[v] = self.colors['vertex_normal']
            if HAS_LIGHTS :
                if g.vp['light'][v] :
                    vertex_color[v] = self.colors['vertex_light']

            vertex_t_color[v]   = self.colors['text_normal']
            halo_color[v]       = self.colors['halo_normal']
            vertex_t_pos[v]     = 0 * np.pi / 4
            vertex_t_size[v]    = 8
            vertex_halo_size[v] = 1.3
            vertex_pen_width[v] = 0.8
            vertex_size[v]      = 7
            halo[v]             = False
            state[v]            = 0

        if 'pos' not in vertex_props :
            g.vp['pos'] = gt.sfdp_layout(g, epsilon=1e-2, cooling_step=0.95)

        edge_props = set()
        for key in g.edge_properties.keys() :
            edge_props = edge_props.union([key])

        self.nV       = g.num_vertices()
        self.nE       = g.num_edges()
        self.nAgents  = [0 for k in range(self.nE)]
        self.edges    = [e for e in g.edges()]

        HAS_LENGTH    = 'edge_length' in edge_props
        HAS_LANES     = 'lanes' in vertex_props
        HAS_CAP       = 'cap' in vertex_props

        for e in g.edges() :
            qissn   = (int(e.source()), int(e.target()), g.edge_index[e])
            if g.ep['garage'][e] :
                cap             = g.vp['cap'][e.target()] if HAS_CAP else 1
                queues[e]       = qs.LossQueue(cap, issn=qissn, net_size=self.nE)
                edge_length[e]  = g.ep['edge_length'][e] if HAS_LENGTH else 0.1
            else : #QueueServer MarkovianQueue
                lanes           = g.vp['lanes'][e.target()] if HAS_LANES else 1
                lanes           = lanes if lanes > 10 else max([int(lanes / 2), 1])
                queues[e]       = qs.QueueServer(lanes, issn=qissn, net_size=self.nE)
                edge_length[e]  = g.ep['edge_length'][e] if HAS_LENGTH else 1
                control[e]      = [0, 0, 0, 0]
            
            edge_color[e]       = self.colors['edge_normal']
            edge_width[e]       = 1.25
            arrow_width[e]      = 8
            edge_times[e]       = 1
            edge_t_size[e]      = 8
            edge_t_distance[e]  = 8
            edge_t_parallel[e]  = False
            edge_t_color[e]     = [0.0, 0.0, 0.0, 1.0]
            edge_state[e]       = 0


        if 'shortest_path' not in vertex_props :
            shortest_path = g.new_vertex_property("vector<int>")
            spath         = np.ones( (self.nV, self.nV), int) * -1

            for v in g.vertices() :
                for u in g.vertices() :
                    if u == v or spath[int(v), int(u)] != -1 :
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

                shortest_path[v] = spath[path[0], :]

            g.vp['shortest_path'] = shortest_path

        node_dict = {'garage' : [], 'destination' : [], 'road' : [], 'dest_road' : []}
        for v in g.vertices() :
            if g.vp['garage'][v] :
                node_dict['garage'].append(int(v))
            elif g.vp['destination'][v] :
                node_dict['destination'].append(int(v))
            else :
                node_dict['road'].append(int(v))
        node_dict['dest_road'] = copy.copy(node_dict['destination'])
        node_dict['dest_road'].extend(node_dict['road'])
        node_index[g]  = node_dict

        g.vp['vertex_t_color']   = vertex_t_color
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

        g.gp['garage_count']     = garage_count
        g.gp['dest_count']       = dest_count
        g.gp['node_index']       = node_index

        self.g  = g
        self.queue_heap = [self.g.ep['queues'][e] for e in self.g.edges()]
        heapify(self.queue_heap)



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



    def set_special_nodes(self, g, pDest=None, pGarage=None) :
        if pDest == None :
            pDest = 0.1
        if pGarage == None :
            pGarage = 1
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
        nGarages    = int(pGarage * np.size(dests))
        min_g_dist  = ones(nGarages) * infty
        ind_g_dist  = ones(nGarages, int)

        r, theta    = np.random.random(nGarages) / 500, np.random.random(nGarages) * 360
        xy_pos      = array([r * np.cos(theta), r * np.sin(theta)]).transpose()
        g_pos       = xy_pos + dest_pos[ np.array( np.mod(np.arange(nGarages), nDests), int) ]

        for v in g.vertices() :
            if int(v) not in dests :
                tmp = array([calculate_distance(g.vp['pos'][v], g_pos[k, :]) for k in range(nGarages)])
                min_g_dist = np.min((tmp, min_g_dist), 0)
                ind_g_dist[min_g_dist == tmp] = int(v)

        ind_g_dist  = np.unique(ind_g_dist)
        garas       = ind_g_dist[:min( (nGarages,len(ind_g_dist)) )]

        if not g.is_directed() :
            g.set_directed(True)

            g2  = g.copy()
            for e in g2.edges() :
                e = g.add_edge(source=int(e.target()), target=int(e.source()))

        destination = g.new_vertex_property("bool")
        garage      = g.new_vertex_property("bool")
        egarage     = g.new_edge_property("bool")
        edge_length = g.new_edge_property('double')

        for v in g.vertices() :
            if int(v) in dests :
                destination[v]  = True
            if int(v) in garas :
                garage[v] = True

        for e in g.edges() :
            latlon1         = g.vp['pos'][e.target()]
            latlon2         = g.vp['pos'][e.source()]
            edge_length[e]  = np.round(calculate_distance(latlon1, latlon2), 3)

        for v in g.vertices() :
            if garage[v] :
                e           = g.add_edge(source=v, target=v)
                egarage[e]  = True

        g.vp['destination'] = destination
        g.vp['garage']      = garage
        g.ep['garage']      = egarage
        return g


    def initialize(self, nActive=1, queues=None) :
        self.g.reindex_edges()
        initialized = False
        if queues == None :
            queues = np.random.randint(0, self.nV, nActive)

        for vi in queues :
            for e in self.g.vertex(vi).out_edges() :
                if e.source() != e.target() :
                    self.g.ep['queues'][e].active   = True
                    self.g.ep['queues'][e].active_p = 0
                    self.g.ep['queues'][e].add_arrival()
                    initialized = True
                    break

        if not initialized :
            print("Could not initialize, trying again")
            if not hasattr(self, tmp) :
                self.tmp  = 1
            else :
                self.tmp += 1
            if self.tmp < 10 :
                self.initialize()
            else :
                raise Exception("Could not initialize, try again.")


    def __repr__(self) :
        return 'QueueNetwork. # nodes : %s, # edges : %s, # agents : %s' % (self.nV, self.nE, sum(self.nAgents))


    def agent_stats(self) :
        ans     = zeros(7)
        parked  = 0
        spaces  = 0
        for e in self.g.edges() :
            q        = self.g.ep['queues'][e]
            ans[:4] += q.travel_stats()
            if isinstance(q, qs.LossQueue) :
                ans[4] += q.lossed()
                parked += q.nSystem
                spaces += q.nServers
        ans[5] = self.g.gp['garage_count']
        ans[6] = parked/spaces
        return ans
            

    def information_stats(self) :
        real  = zeros(self.nE)
        data  = zeros(self.nE)
        a     = arange(self.nE)

        for e in self.g.edges() :
            q   = self.g.ep['queues'][e]
            key = self.g.edge(q.issn[0], q.issn[1]) 
            real[q.issn[2]] = q.nSystem / q.nServers    # q.issn[2] is the edge_index for that edge

        g_index = array(self.g.gp['node_index']['garage'])
        d_index = array(self.g.gp['node_index']['destination'])
        r_index = array(self.g.gp['node_index']['road'])

        for e in self.g.edges() :
            q           = self.g.ep['queues'][e]
            k           = self.g.edge_index[e]
            tmpdata     = q.net_data[:,2]
            ind         = logical_and(a!=k, tmpdata != -1)
            data[ind]  += (tmpdata[ind] - real[ind])**2
     
        data   /= (self.nE - 1)

        return array( ( (np.mean(data[r_index]), np.std(data[r_index])), \
                        (np.mean(data[d_index]), np.std(data[d_index])), \
                        (np.mean(data[g_index]), np.std(data[g_index])), \
                        (np.mean(data),          np.std(data)) ) )


    def _information_stats(self) :
        g   = self.g.gp['node_index']['garage'][0]
        d   = self.g.gp['node_index']['destination'][0]
        r   = self.g.gp['node_index']['road'][1]

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


    def lossed(self) :
        ans = 0
        for e in self.g.edges() :
            q   = self.g.ep['queues'][e]
            if isinstance(q, qs.LossQueue) :
                ans += q.lossed()
        return ans


    def reset_colors(self, which_colors=('all',)) :
        if 'all' in which_colors :
            for e in self.g.edges() :
                self.g.ep['edge_color'][e]    = self.colors['edge_normal']
            for v in self.g.vertices() :
                self.g.vp['vertex_color'][v]  = self.colors['vertex_normal']
                self.g.vp['halo_color'][v]    = self.colors['halo_normal']
                self.g.vp['halo'][v]          = False
                self.g.vp['state'][v]         = 0
        if 'edges' in which_colors :
            for e in self.g.edges() :
                self.g.ep['edge_color'][e]    = self.colors['edge_normal']
        if 'vertices' in which_colors :
            for v in self.g.vertices() :
                self.g.vp['vertex_color'][v]  = self.colors['vertex_normal']
                self.g.vp['halo_color'][v]    = self.colors['halo_normal']
                self.g.vp['halo'][v]          = False
                self.g.vp['state'][v]         = 0
        if 'halos' in which_colors :
            for v in self.g.vertices() :
                self.g.vp['halo_color'][v]    = self.colors['halo_normal']
                self.g.vp['halo'][v]          = False
        if 'basic' in which_colors :
            for v in self.g.vertices() :
                if self.g.vp['destination'][v] :
                    self.g.vp['vertex_color'][v]  = self.colors['vertex_destination']
                elif self.g.vp['garage'][v] :
                    self.g.vp['vertex_color'][v]  = self.colors['vertex_garage']
                else :
                    self.g.vp['vertex_color'][v]  = self.colors['vertex_normal']
                self.g.vp['halo_color'][v]        = self.colors['halo_normal']
                self.g.vp['halo'][v]    = False
                self.g.vp['state'][v]   = 0



    def draw(self, file_name=None) :
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
                        vertex_fill_color=self.g.vp['vertex_color'],
                        vertex_halo=self.g.vp['halo'],
                        vertex_halo_color=self.g.vp['halo_color'],
                        vertex_halo_size=self.g.vp['vertex_halo_size'],
                        vertex_pen_width=self.g.vp['vertex_pen_width'],
                        #vertex_text=self.g.vp['state'],
                        vertex_text_position=self.g.vp['vertex_t_pos'],
                        vertex_font_size=self.g.vp['vertex_t_size'],
                        vertex_size=self.g.vp['vertex_size'])




    def _update_graph(self, ad, vertex, edge ) :
        v, e    = vertex, edge
        self.reset_colors(('edges','halos'))
        
        if ad == 'arrival' :
            cap = self.g.ep['queues'][e].nServers
            tmp = self.g.vp['state'][v]/( 2*cap if cap < infty else self.agent_cap / self.nV )

            if self.g.vp['destination'][v] :
                a     = list(self.colors['vertex_destination'])
                a[2] += (1-a[2]) * tmp
            elif self.g.vp['garage'][v] :
                a     = list(self.colors['vertex_garage'])
                a[1] += (1-a[1]) * tmp
            else :
                a     = [tmp, tmp, tmp, 0.85]

            self.g.vp['vertex_color'][v]  = a
            self.g.vp['halo_color'][v]    = self.colors['halo_arrival']
            self.g.vp['halo'][v]          = True
            self.g.ep['state'][e]         = self.g.ep['queues'][e].nSystem
            if self.g.ep['garage'][e] :
                self.g.vp['state'][v]     = self.g.ep['queues'][e].nSystem

        elif ad == 'departure' :
            cap = self.g.ep['queues'][e].nServers
            tmp = self.g.vp['state'][v]/(2*cap if cap < infty else self.agent_cap/self.nV)

            if self.g.vp['destination'][v] :
                a     = list(self.colors['vertex_destination'])
                a[2] += (1-a[2]) * tmp
            elif self.g.vp['garage'][v] :
                a     = list(self.colors['vertex_garage'])
                a[1] += (1-a[1]) * tmp
            else :
                a     = [tmp, tmp, tmp, 0.85]

            self.g.vp['vertex_color'][v]  = a
            self.g.ep['state'][e]         = self.g.ep['queues'][e].nSystem
            self.g.ep['edge_color'][e]    = self.colors['edge_departure']
            self.g.vp['halo_color'][v]    = self.colors['halo_departure']
            self.g.vp['halo'][v]          = True
            if self.g.ep['garage'][e] :
                self.g.vp['state'][v]     = self.g.ep['queues'][e].nSystem


    def next_time(self) :
        self.queue_heap.sort()
        t   = self.queue_heap[0].next_time
        que = self.queue_heap[0].issn
        e   = self.g.edge(que[0], que[1])
        return t, e


    def next_event_type(self) :
        t, e0   = self.next_time()
        return self.g.ep['queues'][e0].next_event_type()


    def next_event(self, Fast=False, STOP_LEARNER=False) :
        t, e0 = self.next_time()
        queue = self.g.ep['queues'][e0]
        j     = self.g.edge_index[e0] #vertex(j)        
        event_type  = queue.next_event_type()

        if event_type == "departure" and STOP_LEARNER :
            if isinstance(queue.departures[0], qs.LearningAgent) :
                return "STOP"

        self.nEvents += 1
        self.t  = t

        if event_type == "departure" :
            agent           = queue.next_event()
            self.nAgents[j] = queue.nTotal

            if e0.target().out_degree() > 0 :
                e1  = agent.desired_destination(self, e0) # expects the network, and current location
                agent.set_arrival(t + 0.25 * self.g.ep['edge_length'][e1])

                self.g.ep['queues'][e1].add_arrival(agent)
                self.nAgents[self.g.edge_index[e1]] = self.g.ep['queues'][e1].nTotal

                if not Fast :
                    self._update_graph(ad='departure', vertex=e1.target(), edge=e1)

        else :
            if queue.active and sum(self.nAgents) > self.agent_cap - 1 :
                queue.active = False

            queue.next_event()
            self.nAgents[j] = queue.nTotal

            if not Fast :
                self._update_graph(ad='arrival', vertex=e0.target(), edge=e0)

        if self.to_animate :
            future = self.next_time()[0]
            time.sleep( future - self.t )
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
                edge_text=self.g.ep['state'],
                edge_font_size=self.g.ep['edge_t_size'],
                edge_text_distance=self.g.ep['edge_t_distance'],
                edge_text_parallel=self.g.ep['edge_t_parallel'],
                edge_text_color=self.g.ep['edge_t_color'],
                vertex_fill_color=self.g.vp['vertex_color'],
                vertex_halo=self.g.vp['halo'],
                vertex_halo_color=self.g.vp['halo_color'],
                vertex_halo_size=self.g.vp['vertex_halo_size'],
                vertex_pen_width=self.g.vp['vertex_pen_width'],
                vertex_text=self.g.vp['state'],
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
            self.next_event(Fast=True)


    def reset(self) :
        self.t          = 0
        self.nEvents    = 0
        self.nAgents    = [0 for k in range( self.nE )]
        self.to_animate = False
        self.reset_colors()
        for e in self.g.edges() :
            self.g.ep['queues'][e].reset()


    def copy(self) :
        net             = QueueNetwork(0)
        net.t           = copy.deepcopy(self.t)
        net.agent_cap   = copy.deepcopy(self.agent_cap)
        net.nV          = copy.deepcopy(self.nV)
        net.nE          = copy.deepcopy(self.nE)
        net.to_animate  = False
        net.g           = self.g.copy()
        net.nEvents     = copy.deepcopy(self.nEvents)
        net.nAgents     = copy.deepcopy(self.nAgents)
        net.colors      = copy.deepcopy(self.colors)
        net.edges       = [e for e in net.g.edges()]
        queues          = net.g.new_edge_property("python::object")

        for e in net.g.edges() :
            queues[e]   = copy.deepcopy( net.g.ep['queues'][e] )

        net.g.ep['queues']  = queues
        net.queue_heap      = [net.g.ep['queues'][e] for e in net.g.edges()]
        heapify(net.queue_heap)
        return net






"""
class QueueNetwork :

    def __init__(self, g=None, nVertices=250, pDest=None, pGarage=None, seed=None, graph_type="normal") :
        a = 1

    def next_event(self, Fast=False, STOP_LEARNER=False) :
        t, e0 = self.next_time()
        queue = self.g.ep['queues'][e0]
        j     = self.g.edge_index[e0] #vertex(j)        
        event_type  = queue.next_event_type()

        if event_type == "departure" and STOP_LEARNER :
            if isinstance(queue.departures[0], qs.LearningAgent) :
                return "STOP"

        self.nEvents += 1
        self.t  = t

        if event_type == "departure" :
            agent           = queue.next_event()
            self.nAgents[j] = queue.nTotal

            if len(list(e0.target().out_edges())) > 0 :
                e1   = agent.desired_destination(self, e0) # expects the network, and current location
                agent.set_arrival(t + 0.25 * self.g.ep['edge_length'][e1] )

                self.g.ep['queues'][e1].add_arrival(agent)
                self.nAgents[self.g.edge_index[e1]] = self.g.ep['queues'][e1].nTotal

                if not Fast :
                    self._update_graph(ad='departure', vertex=e1.target(), edge=e1)

        else :
            if queue.active and sum(self.nAgents) > self.agent_cap - 1 :
                queue.active = False

            queue.next_event()
            self.nAgents[j] = queue.nTotal

            if not Fast :
                self._update_graph(ad='arrival', vertex=e0.target(), edge=e0)

        if self.to_animate :
            future = self.next_time()[0]
            time.sleep( future - self.t )
            self.win.graph.regenerate_surface(lazy=False)
            self.win.graph.queue_draw()
            return True

"""


