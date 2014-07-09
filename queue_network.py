import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import graph_tool.all       as gt
import queue_server         as qs
import time
import copy

from numpy                  import ones, zeros, array, arange, logical_and, infty
from heapq                  import heappush, heappop, heapify
from collections            import deque
from gi.repository          import Gtk, Gdk, GdkPixbuf, GObject
from queue_agents           import Agent, Smart_Agent, Learning_Agent, Random_Agent

class Queue_network :

    def __init__(self, ab=None, graph_type="normal", p_dest=None, p_garage=None, seed=None) :

        self.t          = 0
        self.to_animate = False
        self.nAgents    = [0]
        self.agent_cap  = 100
        self.colors     =  {'edge_departure'    : [0, 0, 0, 1], 
                            'edge_normal'       : [0.339, 0.3063, 0.3170, 0.290],
                            'vertex_normal'     : [0.0, 0.5, 1.0, 0.85],
                            'vertex_light'      : [1.0, 0.135, 0.0, 1.0],
                            'vertex_garage'     : [0.133, 0.545, 0.133, 1.0],
                            'vertex_destination': [0.282, 0.239, 0.545, 1.0],
                            'vertex_arrival'    : [0.4, 0.8, 0.4, 0.75], 
                            'vertex_departure'  : [0.4, 0.8, 0.4, 0.75],
                            'halo_normal'       : [0, 0, 0, 0],
                            'halo_arrival'      : [0.1, 0.8, 0.8, 0.25],
                            'halo_departure'    : [0.9, 0.9, 0.9, 0.25],
                            'text_normal'       : [1, 1, 1, 0.5],
                            'bg_color'          : [1.95, 1.95, 1.95, 1.0] }


        if isinstance(seed, int) :
            np.random.seed(seed)
            gt.seed_rng(seed)

        if ab == None :
            if graph_type == "periodic" :
                self.create_graph2(p_dest=p_dest, p_garage=p_garage)
            else :
                self.create_graph(p=p_dest)
        elif isinstance(ab, list) :
            if graph_type == "periodic" :
                self.create_graph2(ab, p_dest, p_garage)
            else :
                self.create_graph(ab, p_dest)
        elif isinstance(ab, str) :
            self.set_graph(ab)
        elif isinstance(ab, gt.Graph) :
            self.set_graph(ab)
        elif isinstance(ab, gt.Graph) and False :
            self.g      = ab
            self.nV     = ab.num_vertices()
            self.nE     = ab.num_edges()
            self.nAgents= [0 for k in range(self.nE)]
            e_props = set()
            for key in self.g.edge_properties.keys() :
                e_props = e_props.union( [key] )
            if 'queues' not in e_props :
                for e in self.g.edges() :
                    qissn   = (int(e.source()), int(e.target()), self.g.edge_index[e])
                    self.g.ep['queues'][e]  = qs.Queue_server(2, issn=qissn, net_size=self.nE)
        elif ab == 0 :
            pass


    def set_graph(self, g) :
        if isinstance(g, str) :
            self.g = gt.load_graph(g, fmt='xml')
        elif isinstance(g, gt.Graph) :
            self.g = g

        vertex_t_color      = self.g.new_vertex_property("vector<double>")
        vertex_color        = self.g.new_vertex_property("vector<double>")
        halo_color          = self.g.new_vertex_property("vector<double>")
        vertex_t_pos        = self.g.new_vertex_property("double")
        vertex_t_size       = self.g.new_vertex_property("double")
        halo                = self.g.new_vertex_property("bool")
        state               = self.g.new_vertex_property("int")
        vertex_type         = self.g.new_vertex_property("int")
        vertex_halo_size    = self.g.new_vertex_property("double")
        vertex_pen_width    = self.g.new_vertex_property("double")
        vertex_size         = self.g.new_vertex_property("double")

        control             = self.g.new_edge_property("vector<double>")
        edge_color          = self.g.new_edge_property("vector<double>")
        edge_t_color        = self.g.new_edge_property("vector<double>")
        edge_width          = self.g.new_edge_property("double")
        arrow_width         = self.g.new_edge_property("double")
        edge_length         = self.g.new_edge_property("double")
        edge_times          = self.g.new_edge_property("double")
        edge_t_size         = self.g.new_edge_property("double")
        edge_t_distance     = self.g.new_edge_property("double")
        edge_t_parallel     = self.g.new_edge_property("bool")
        edge_state          = self.g.new_edge_property("int")
        queues              = self.g.new_edge_property("python::object")

        garage_count        = self.g.new_graph_property("int")
        dest_count          = self.g.new_graph_property("int")
        node_index          = self.g.new_graph_property("python::object")

        dest_count[self.g]  = sum( self.g.vp['destination'].a )
        garage_count[self.g]= sum( self.g.vp['garage'].a )

        if garage_count[self.g] == 0 :
            print("Need more than 0 garages")
            return

        v_props = set()
        for key in self.g.vertex_properties.keys() :
            v_props = v_props.union( [key] )

        HAS_LIGHTS  = 'light' in v_props

        for v in self.g.vertices() :
            if self.g.vp['destination'][v] :
                vertex_color[v] = self.colors['vertex_destination']
            elif self.g.vp['garage'][v] :
                vertex_color[v] = self.colors['vertex_garage']
            else :
                vertex_color[v] = self.colors['vertex_normal']
            if HAS_LIGHTS :
                if self.g.vp['light'][v] :
                    vertex_color[v] = self.colors['vertex_light']

            vertex_t_color[v]   = self.colors['text_normal']
            halo_color[v]       = self.colors['halo_normal']
            vertex_t_pos[v]     = 0 * np.pi / 4
            vertex_t_size[v]    = 8
            halo[v]             = False
            state[v]            = 0
            vertex_halo_size[v] = 1.3
            vertex_pen_width[v] = 0.8
            vertex_size[v]      = 7


        if 'pos' not in v_props :
            self.g.vp['pos']    = gt.sfdp_layout(self.g, epsilon=1e-2, cooling_step=0.95)

        e_props = set()
        for key in self.g.edge_properties.keys() :
            e_props = e_props.union( [key] )

        self.nV         = self.g.num_vertices()
        self.nE         = self.g.num_edges()
        self.nAgents    = [0 for k in range(self.nE)]
        self.edges      = [e for e in self.g.edges()]

        HAS_LENGTH  = 'edge_length' in e_props
        HAS_CAP     = 'cap' in v_props
        HAS_LANES   = 'lanes' in v_props

        for e in self.g.edges() :
            qissn   = (int(e.source()), int(e.target()), self.g.edge_index[e])
            if self.g.ep['garage'][e] :
                cap             = self.g.vp['cap'][e.target()] if HAS_CAP else 1
                queues[e]       = qs.Loss_Queue(cap, issn=qissn, net_size=self.nE)
                edge_length[e]  = self.g.ep['edge_length'][e] if HAS_LENGTH else 0.1
            else :
                lanes           = self.g.vp['lanes'][e.target()] if HAS_LANES else 1
                lanes           = lanes if lanes > 10 else max([int(lanes / 2), 1])
                queues[e]       = qs.Queue_server(lanes, issn=qissn, net_size=self.nE)
                edge_length[e]  = self.g.ep['edge_length'][e] if HAS_LENGTH else 1
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


        if 'shortest_path' not in v_props :
            shortest_path       = self.g.new_vertex_property("vector<int>")
            node_index[self.g]  = dict({'garage' : [], 'destination' : [], 'road' : []})
            tmp                 = [0 for k in range(self.nV)]

            for k in range(self.nV) :
                tmp[k]  = 0
                vk      = self.g.vertex(k)
                if self.g.vp['garage'].a[k] :
                    node_index[self.g]['garage'].append(k)
                elif self.g.vp['destination'].a[k] :
                    node_index[self.g]['destination'].append(k)
                else :
                    node_index[self.g]['road'].append(k)
                for j in range(self.nV) :
                    if k == j : 
                        continue
                    tmp[j]  = int(gt.shortest_path(self.g, vk, 
                                  self.g.vertex(j), weights=edge_length)[0][1])
                shortest_path[vk] = tmp
            self.g.vp['shortest_path']  = shortest_path
        else :
            node_dict   = {'garage' : [], 'destination' : [], 'road' : [], 'dest_road' : []}
            for v in self.g.vertices() :
                if self.g.vp['garage'][v] :
                    node_dict['garage'].append( int(v) )
                elif self.g.vp['destination'][v] :
                    node_dict['destination'].append( int(v) )
                else :
                    node_dict['road'].append( int(v) )
            node_dict['dest_road'] = copy.copy( node_dict['destination'] )
            node_dict['dest_road'].extend( node_dict['road'] )
            node_index[self.g]  = node_dict


        self.g.vp['vertex_t_color']     = vertex_t_color
        self.g.vp['vertex_color']       = vertex_color
        self.g.vp['halo_color']         = halo_color
        self.g.vp['vertex_t_pos']       = vertex_t_pos
        self.g.vp['vertex_t_size']      = vertex_t_size
        self.g.vp['halo']               = halo
        self.g.vp['state']              = state
        self.g.vp['vertex_type']        = vertex_type 
        self.g.vp['vertex_halo_size']   = vertex_halo_size
        self.g.vp['vertex_pen_width']   = vertex_pen_width
        self.g.vp['vertex_size']        = vertex_size

            
        self.g.ep['edge_t_size']        = edge_t_size
        self.g.ep['edge_t_distance']    = edge_t_distance
        self.g.ep['edge_t_parallel']    = edge_t_parallel
        self.g.ep['edge_t_color']       = edge_t_color
        self.g.ep['state']              = edge_state
        self.g.ep['control']            = control
        self.g.ep['edge_color']         = edge_color
        self.g.ep['edge_width']         = edge_width
        self.g.ep['edge_length']        = edge_length
        self.g.ep['arrow_width']        = arrow_width
        self.g.ep['edge_times']         = edge_times
        self.g.ep['queues']             = queues

        self.g.gp['garage_count']       = garage_count
        self.g.gp['dest_count']         = dest_count
        self.g.gp['node_index']         = node_index

        self.queue_heap = [ self.g.ep['queues'][e] for e in self.g.edges()]
        heapify(self.queue_heap)



    def create_graph(self, ab=None,  p=None) :
        if p == None :
            p = 0.45            
        if ab == None :
            ab  = [np.random.randint(2, 5), np.random.randint(2, 5)]

        g   = gt.lattice(ab)

        destination = g.new_vertex_property("bool")
        garage      = g.new_vertex_property("bool")
        tmp_set     = set([2, 3])

        g2  = g.copy()
        for v in g2.vertices() :
            if np.random.uniform() < p:
                y       = g.vertex( int(v) )
                v_iter  = g.add_vertex( int(np.random.multinomial(1, [0, 0.35, 0.55, 0.10]).argmax()) )
                destination[y]  = True
                if isinstance( v_iter, gt.Vertex) :
                    garage[v_iter]   = True
                    g.add_edge(y, v_iter)
                else :
                    for z in v_iter :
                        garage[z]   = True
                        g.add_edge(y, z)

        g2  = g.copy()
        for v1 in g2.vertices() :
            if destination[v1] :
                for v2 in v1.all_neighbours() :
                    if destination[v2] :
                        for v3 in v2.all_neighbours() :
                            if garage[v3] :
                                if np.random.uniform() < 0.6 :
                                    g.add_edge(v1, v3)


        to_remove   = g.new_vertex_property("bool")
        for v in g.vertices() :
            if v.out_degree() in tmp_set and not destination[v] and not garage[v] :
                to_remove[v] = True
                for y in v.all_neighbours() :
                    if destination[y] :
                        to_remove[v] = False

        g.set_vertex_filter(to_remove, inverted=True)        
        g.purge_vertices(in_place=False)
        g.set_directed(True)

        g2  = g.copy()
        for e in g2.edges() :
            target1 = int(e.target())
            source1 = int(e.source())
            g.add_edge(source=target1, target=source1)

        for v in g.vertices() :
            if garage[v] :
                g.add_edge(source=int(v), target=int(v))

        egarage = g.new_edge_property("bool")
        for e in g.edges() :
            if e.target() == e.source() :
                egarage[e]  = True
            else :
                egarage[e]  = False

        g.vp['destination'] = destination
        g.vp['garage']      = garage
        g.ep['garage']      = egarage

        if sum(destination.a) < 2 :
            self.create_graph(ab=ab, p=p)
        else :
            self.set_graph(g)



    def create_graph2(self, ab=None,  p_dest=None, p_garage=None) :
        if p_dest == None :
            p_dest  = 0.2
        if p_garage == None :
            p_garage = 0.55
        if ab == None :
            ab  = [np.random.randint(4, 7), np.random.randint(4, 7)]

        g           = gt.lattice(ab, periodic=True)
        g.set_directed(True)

        destination = g.new_vertex_property("bool")
        garage      = g.new_vertex_property("bool")
        egarage     = g.new_edge_property("bool")

        for v in g.vertices() :
            if np.random.uniform() < p_dest :
                destination[v]  = True
                for w in v.all_neighbours() :
                    if np.random.uniform() < p_garage and not destination[w] :
                        garage[w]   = True

        g2  = g.copy()
        for e in g2.edges() :
            if np.random.uniform() < 10.75 :
                g.add_edge( source=int(e.target()), target=int(e.source()) )

        for v in g.vertices() :
            if garage[v] :
                g.add_edge(source=v, target=v)

        for e in g.edges() :
            if e.target() == e.source() :
                egarage[e]  = True
            else :
                egarage[e]  = False

        g.vp['destination'] = destination
        g.vp['garage']      = garage
        g.ep['garage']      = egarage

        if sum(destination.a) < 2 :
            self.create_graph2(ab, p_dest, p_garage)
        else :
            self.set_graph(g)


    def __repr__(self) :
        return 'Queue_network. # nodes: %s, # edges: %s, # agents: %s' % (self.nV, self.nE, sum(self.nAgents))


    def agent_stats(self) :
        ans     = zeros(7)
        parked  = 0
        spaces  = 0
        for e in self.g.edges() :
            q       = self.g.ep['queues'][e]
            ans[:4]+= q.travel_stats()
            if isinstance(q, qs.Loss_Queue) :
                ans[4] += q.lossed()
                parked += q.nSystem
                spaces += q.nServers
        ans[5] = self.g.gp['garage_count']
        ans[6] = parked/spaces
        return ans
            

    def information_stats(self) :
        real    = zeros(self.nE)
        data    = zeros(self.nE)
        a       = arange(self.nE)

        for e in self.g.edges() :
            q   = self.g.ep['queues'][e]
            key = self.g.edge( q.issn[0], q.issn[1] ) 
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
            data    = infty * ones(self.nE)
            a       = np.arange(self.nE)
            a       = a[a!=q]
            qnew    = self.g.ep['queues'][self.edges[q]]
            real    = qnew.nSystem / qnew.nServers

            for k in a :
                tmpdata = self.g.ep['queues'][self.edges[k]].net_data[q,2]
                if tmpdata == -1 : 
                    continue
                data[k] = (tmpdata - real)
         
            data        = data[ data != infty ]
            ans[ct,:]   = np.mean( data ), np.std( data )
            ct         += 1

        return ans


    def lossed(self) :
        ans = 0
        for e in self.g.edges() :
            q   = self.g.ep['queues'][e]
            if isinstance(q, qs.Loss_Queue) :
                ans += q.lossed()
        return ans


    def reset_colors(self, which_colors=('all',)) :
        if 'all' in which_colors :
            for e in self.g.edges() :
                self.g.ep['edge_color'][e]          = self.colors['edge_normal']
            for v in self.g.vertices() :
                self.g.vp['vertex_color'][v]        = self.colors['vertex_normal']
                self.g.vp['halo_color'][v]          = self.colors['halo_normal']
                self.g.vp['halo'][v]                = False
                self.g.vp['state'][v]               = 0
        if 'edges' in which_colors :
            for e in self.g.edges() :
                self.g.ep['edge_color'][e]          = self.colors['edge_normal']
        if 'vertices' in which_colors :
            for v in self.g.vertices() :
                self.g.vp['vertex_color'][v]        = self.colors['vertex_normal']
                self.g.vp['halo_color'][v]          = self.colors['halo_normal']
                self.g.vp['halo'][v]                = False
                self.g.vp['state'][v]               = 0
        if 'halos' in which_colors :
            for v in self.g.vertices() :
                self.g.vp['halo_color'][v]          = self.colors['halo_normal']
                self.g.vp['halo'][v]                = False
        if 'basic' in which_colors :
            for v in self.g.vertices() :
                if self.g.vp['destination'][v] :
                    self.g.vp['vertex_color'][v]    = self.colors['vertex_destination']
                elif self.g.vp['garage'][v] :
                    self.g.vp['vertex_color'][v]    = self.colors['vertex_garage']
                else :
                    self.g.vp['vertex_color'][v]    = self.colors['vertex_normal']
                self.g.vp['halo_color'][v]          = self.colors['halo_normal']
                self.g.vp['halo'][v]                = False
                self.g.vp['state'][v]               = 0



    def draw(self, file_name=None) :
        if file_name == None :
            gt.graph_draw(self.g, self.g.vp['pos'], geometry=(750, 750),
                        bg_color=self.colors['bg_color'],
                        edge_color=self.g.ep['edge_color'],
                        edge_control_points=self.g.ep['control'],
                        edge_marker_size=self.g.ep['arrow_width'],
                        edge_pen_width=self.g.ep['edge_width'],
                        #edge_text=self.g.ep['state'],
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
            gt.graph_draw(self.g, self.g.vp['pos'], geometry=(750, 750), output=file_name,
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




    def _update_graph(self, ad, vertex, edge ) :
        v, e    = vertex, edge
        self.reset_colors(('edges','halos'))
        
        if ad == 'arrival' :
            cap = self.g.ep['queues'][e].nServers
            tmp = self.g.vp['state'][v]/( 2*cap if cap < infty else self.agent_cap / self.nV )

            if self.g.vp['destination'][v] :
                a       = list(self.colors['vertex_destination'])
                a[2]   += (1-a[2]) * tmp
                self.g.vp['vertex_color'][v]    = a
            elif self.g.vp['garage'][v] :
                a       = list(self.colors['vertex_garage'])
                a[1]   += (1-a[1]) * tmp
                self.g.vp['vertex_color'][v]    = a
            else :
                self.g.vp['vertex_color'][v]    = [tmp, tmp, tmp, 0.85]
            self.g.vp['halo_color'][v]          = self.colors['halo_arrival']
            self.g.vp['halo'][v]                = True
            self.g.ep['state'][e]               = self.g.ep['queues'][e].nSystem
            if self.g.ep['garage'][e] :
                self.g.vp['state'][v]           = self.g.ep['queues'][e].nSystem

        elif ad == 'departure' :
            cap = self.g.ep['queues'][e].nServers
            tmp = self.g.vp['state'][v]/(2*cap if cap < infty else self.agent_cap/self.nV)

            if self.g.vp['destination'][v] :
                a       = list(self.colors['vertex_destination'])
                a[2]   += (1-a[2]) * tmp
                self.g.vp['vertex_color'][v]    = a
            elif self.g.vp['garage'][v] :
                a       = list(self.colors['vertex_garage'])
                a[1]   += (1-a[1]) * tmp
                self.g.vp['vertex_color'][v]    = a
            else :
                self.g.vp['vertex_color'][v]    = [tmp, tmp, tmp, 0.85]
            self.g.ep['state'][e]               = self.g.ep['queues'][e].nSystem
            self.g.ep['edge_color'][e]          = self.colors['edge_departure']
            self.g.vp['halo_color'][v]          = self.colors['halo_departure']
            self.g.vp['halo'][v]                = True
            if self.g.ep['garage'][e] :
                self.g.vp['state'][v]           = self.g.ep['queues'][e].nSystem


    def next_time(self) :
        self.queue_heap.sort()
        t   = self.queue_heap[0].next_time()
        que = self.queue_heap[0].issn
        e   = self.g.edge(que[0], que[1])
        return t, e


    def next_event_type(self) :
        t, e0       = self.next_time()
        return self.g.ep['queues'][e0].next_event_type()


    def next_event(self, Fast=False, STOP_LEARNER=False) :
        t, e0       = self.next_time()
        j           = self.g.edge_index[e0] #vertex(j)        
        event_type  = self.g.ep['queues'][e0].next_event_type()

        if event_type == "departure" and STOP_LEARNER:
            if isinstance(self.g.ep['queues'][e0].departures[0], qs.Learning_Agent) :
                return "STOP"

        if self.t < t :
            self.t  = t

        if event_type == "departure" :
            agent           = self.g.ep['queues'][e0].next_event()
            self.nAgents[j] = self.g.ep['queues'][e0].nTotal

            if len(list(e0.target().out_edges())) > 0 :
                e1   = agent.desired_destination(self, e0) # expects the network, and current location
                agent.set_arrival(t + 0.25 * self.g.ep['edge_length'][e1] )

                self.g.ep['queues'][e1].add_arrival(agent)
                self.nAgents[self.g.edge_index[e1]] = self.g.ep['queues'][e1].nTotal

                if not Fast :
                    self._update_graph(ad='departure', vertex=e1.target(), edge=e1)

        else :
            if self.g.ep['queues'][e0].CREATE and sum(self.nAgents) > self.agent_cap - 1 :
                self.g.ep['queues'][e0].CREATE = False

            self.g.ep['queues'][e0].next_event()
            self.nAgents[j] = self.g.ep['queues'][e0].nTotal

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
        self.nAgents    = [0 for k in range( self.nE )]
        self.to_animate = False
        self.reset_colors()
        for e in self.g.edges() :
            self.g.ep['queues'][e].reset()


    def copy(self) :
        net             = Queue_network(0)
        net.t           = copy.deepcopy(self.t)
        net.agent_cap   = copy.deepcopy(self.agent_cap)
        net.nV          = copy.deepcopy(self.nV)
        net.nE          = copy.deepcopy(self.nE)
        net.to_animate  = False
        net.g           = self.g.copy()
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

