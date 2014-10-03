import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import graph_tool.all       as gt
import queueing_tool        as qt
import heapq
import time
import copy
import os

from numpy.random   import uniform, multinomial
from numpy          import size, ones, zeros, array, ndarray, transpose, vstack, arange
from numpy          import logical_and, logical_or, logical_not, infty, log
from heapq          import heappush, heappop
from collections    import deque
from gi.repository  import Gtk, Gdk, GdkPixbuf, GObject

## Incorporate expected_sojourn calculation into QueueNetwork, perhaps by doing the 
## calculation at every, departure event. Find out how much this will slow things down


# basis_function1(S, QN)
# Calculates the actual shortest path between current location and destination

# basis_function2(S, QN, RETURN_PATH=False)
# Calculates the expected travel times between nodes in the graph and returns the
# time of the stochastic shortest path between our current location and destination

# basis_function3(S, QN, PRE_SET=False)
# Calculates the expected travel times between nodes in the graph and returns the
# time of the stochastic shortest path between our current location and the garage
# that is 'closest' to the destination


class approximate_dynamic_program :

    def __init__(self, Qn=None, seed=None ) :

        self.parameters         = {'N': 10, 'M':10, 'T': 40, 'gamma':0.95}
        self.t                  = 0
        self.animate            = False
        self.agent_cap          = 50
        self.dist               = None
        self.parking_penalty    = None
        self.nFeatures          = 5
        self.beta               = 0.5 * ones( self.nFeatures )
        self.agent_variables    = {}
        self.parked             = {}
        self.dir                = {'frames' : './figures/frames/'}

        if Qn == None :
            self.Qn = self.active_network(agent_cap=self.agent_cap, seed=seed)
        elif isinstance(Qn, gt.Graph) :
            self.Qn = qt.QueueNetwork(Qn)
            tmp0    = self.Qn.g.vp['destination'].a + self.Qn.g.vp['garage'].a
            self.ce = np.arange( self.Qn.nV )[tmp0==min(tmp0)]

        self.edge_text    = self.Qn.g.new_edge_property("string")
        self.vertex_text  = self.Qn.g.new_vertex_property("string")
        self.calculate_parking_penalty2()


    def active_network(self, agent_cap, net_size=150, seed=None) :
        Qn  = qt.QueueNetwork(nVertices=net_size, graph_type="periodic", seed=seed)
        Qn.agent_cap = agent_cap
        for q in Qn.queue_heap :
            q.xArrival    = lambda x : x - log(uniform()) / 1
            q.xDepart     = lambda x : x - log(uniform()) / 3
            q.xDepart_mu  = lambda x : 1/3 

        tmp0    = Qn.g.vp['vType'].a
        self.ce = np.arange( Qn.nV )[tmp0==min(tmp0)] # Creation edge
        
        garage_cap  = int( np.ceil( agent_cap / (1.5 * Qn.fcq_count) ) )
        garage_sum  = [[] for k in range(Qn.nV)]

        garage_cap  = np.random.uniform(0, 1, Qn.garage_count) / Qn.garage_count
        garage_cap  = np.floor( garage_cap * agent_cap * 8 )
        garage_cap  = [max([int(k),2]) for k in garage_cap]
        ct          = 0
        print( (garage_cap, sum(garage_cap), agent_cap) )

        for v in Qn.g.vertices() :
            if Qn.g.vp['vType'][v] not in (1,2) :
                for e in v.in_edges() :
                    Qn.g.ep['queues'][e].xArrival   = lambda x : x - log(uniform()) / 8
                    Qn.g.ep['queues'][e].xDepart    = lambda x : x - log(uniform()) / 3
                    Qn.g.ep['queues'][e].xDepart_mu = lambda x : 1/3
                    Qn.g.ep['queues'][e].initialize()

        for e in Qn.g.edges() :
            if Qn.g.ep['eType'][e] == 1 :
                Qn.g.ep['queues'][e].set_nServers(garage_cap[ct])
                Qn.g.ep['queues'][e].xDepart    = lambda x : x - log(uniform()) / 0.5
                Qn.g.ep['queues'][e].xDepart_mu = lambda x : 2
                ct += 1

        return Qn



    def __repr__(self) :
        return 'whatever'


    def time2cost(self, t) :
        return t


    def length2cost(self, l) :
        return l

    ## Calculates it incorrectly, fix
    ## if time2cost was a different function, make sure to apply it here
    def expected_sojourn(self, e, QN, S=None, time2cost=True) : 
        exp_s   = copy.deepcopy(QN.t)

        if S == None :
            kk  = max((QN.g.ep['queues'][e].nSystem - QN.g.ep['queues'][e].nServers, 0)) + 1
        else :
            kk  = max((S[QN.g.edge_index[e]+1] - QN.g.ep['queues'][e].nServers, 0)) + 1

        for k in range(kk) :
            exp_s   += QN.g.ep['queues'][e].xDepart_mu(exp_s)

        exp_s   += 0.25 * QN.g.ep['edge_length'][e]
        return exp_s-QN.t


    def simple_model(self, act, QN, state) :

        exp_t       = self.expected_sojourn(act, QN, state)
        exp_depart  = QN.g.new_edge_property("int")
        exp_state   = QN.g.new_edge_property("int")

        for e in QN.g.edges() :
            exp_depart[e] = -1

        for e in QN.g.edges() :
            exp_state[e]   += state[QN.g.edge_index[e]+1]
            dum_t           = QN.t

            while dum_t <= exp_t :
                dum_t          += QN.g.ep['queues'][e].xDepart_mu(dum_t)
                exp_depart[e]  += 1

            if exp_depart[e] > state[QN.g.edge_index[e]+1] :
                exp_depart[e] = state[QN.g.edge_index[e]+1]

            if exp_depart[e] > 0 :
                exp_state[e]  -= exp_depart[e]
                
                od  = int(e.target().out_degree())
                ed  = int(np.floor(exp_depart[e] / od))
                c   = 0
                depart_list = zeros( od, int )

                if ed < 1 :
                    depart_list[:exp_depart[e]] = 1
                else :
                    depart_list[:]  = ed
                    if od * ed < exp_depart[e] :
                        tmp = exp_depart[e] - od * ed
                        depart_list[:tmp] += 1
            
                np.random.shuffle( depart_list )

                for w in e.target().out_edges() :
                    exp_state[w] += depart_list[c]
                    c  += 1

        ans = [state[0]]
        ans.extend([exp_state[e] for e in QN.g.edges()])
        return ans


    def post_decision_state(self, state, e, QN) :
        # S: stands for state, a: stands for action
        S       = copy.deepcopy(state)
        k       = S[0][0]
        a       = QN.g.edge_index[e]
        S[k+1] -= 0 ### Fix later
        S[a+1] += 1
        S[0][0] = int(e.target())
        return S


    def advise_agent(self, action, QN) :
        t, e  = QN.next_time()
        QN.g.ep['queues'][e].departures[0].dest = action


    def simulate_forward(self, QN):
        stop  = QN.next_event(Fast=True, STOP_LEARNER=False)
        stop  = None
        while stop == None :
            stop  = QN.next_event(Fast=True, STOP_LEARNER=True)
        return


    def calculate_parking_penalty(self) :
        dist = zeros( (self.Qn.nV, self.Qn.nV) )
        for ve in self.Qn.g.vertices() :
            for we in self.Qn.g.vertices() :
                v,w  = int(ve), int(we)
                if v == w or dist[w, v] != 0 or dist[v, w] != 0 :
                    continue
                u           = self.Qn.shortest_path[w, v]
                dist[w, v] += self.Qn.g.ep['edge_length'][ self.Qn.g.edge(w, u) ]
                while v != u :
                    u_new       = self.Qn.shortest_path[u, v]
                    dist[w, v] += self.Qn.g.ep['edge_length'][self.Qn.g.edge(u, u_new)]
                    u           = u_new

        dist       += np.transpose(dist) 
        self.dist   = copy.deepcopy(dist)

        for k in range(self.Qn.nV) :
            for j in range(self.Qn.nV) :
                if self.Qn.g.vp['vType'][self.Qn.g.vertex(k)] != 1 :
                    dist[k, j] = 8000
                else :
                    dist[k, j] = min((0.5 * np.exp(1.5 * dist[k, j]) - 2, 1000))

        self.parking_penalty  = np.abs(dist)
        self.full_penalty     = 10


    def calculate_parking_penalty2(self) :

        v_props = set()
        for key in self.Qn.g.vertex_properties.keys() :
            v_props = v_props.union([key])

        dist    = zeros((self.Qn.nV, self.Qn.nV))

        if 'dist' not in v_props :        
            dist    = zeros((self.Qn.nV, self.Qn.nV))
            for ve in self.Qn.g.vertices() :
                for we in self.Qn.g.vertices() :
                    v,w  = int(ve), int(we)
                    if v == w or dist[w, v] != 0 or dist[v, w] != 0 :
                        continue
                    tmp     = gt.shortest_path(self.Qn.g, ve, we, weights=self.Qn.g.ep['edge_length'])
                    path    = [int(v) for v in tmp[0]]
                    elen    = [self.Qn.g.ep['edge_length'][e] for e in tmp[1]]
                    for i in range(len(path) - 1):
                        for j in range(i+1, len(path)):
                            dist[path[i], path[j]] = sum(elen[i:j])

            dist       += np.transpose( dist ) 
        else :
            for v in self.Qn.g.vertices() :
                dist[int(v),:] = self.Qn.g.vp['dist'][v].a

        self.dist   = dist
        pp          = 0.5 * np.exp(1.5 * dist) - 2
        pp[pp>1000] = 1000

        for v in self.Qn.g.vertices() :
            if self.Qn.g.vp['vType'][v] != 1 :
                pp[int(v), :] = 8000

        self.parking_penalty  = np.abs(pp)
        self.full_penalty     = 10


    def parking_value(self, origin, destination, S, QN) :
        e   = QN.g.edge(origin, origin)
        if isinstance(e, gt.Edge) :
            cap = QN.g.ep['queues'][e].nServers
            ei  = QN.g.edge_index[e]
            r   = S[ei+1] / cap
            p   = 1.0 / ( 1.0 + np.exp(-40*(r-19/20)) )
            ans = self.parking_penalty[origin, destination] + p * self.full_penalty
        else :
            ans = self.parking_penalty[origin, destination]

        return ans


    # algorithm from pg 405 of Powell 2011
    # theta update from pg 349 of Powell 2011

    def approximate_policy_iteration(self, orig, dest, COMPLETE_INFORMATION=True, save_frames=False) :
        if save_frames and not os.path.exists(self.dir['frames']) :
            os.mkdir(self.dir['frames'])

        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        gamma   = self.parameters['gamma']

        self.setup_adp( len(orig) )

        max_out_degree  = 1
        for v in self.Qn.g.vertices() :
            max_out_degree  = max( (max_out_degree, v.out_degree()) )
        max_out_degree += 1

        cost        = zeros(max_out_degree)
        target_node = zeros(max_out_degree, int)
        value_es    = zeros(max_out_degree)
        obj_func    = zeros(max_out_degree)
        basis_es    = np.mat(zeros( (self.nFeatures, max_out_degree) ))
        for_state   = [0 for k in range(self.Qn.nE)]

        if hasattr(dest, '__iter__') :
            print(self.parking_penalty[:, dest[0]])
        else :
            print(self.parking_penalty[:, dest] )


        for n in range(N) :
            self.random_state(orig, dest, (10,20) )
            print( sum(self.Qn.nAgents) )

            for m in range(M) :
                QN  = self.Qn.copy()

                for agent_var in self.agent_variables.values() :
                    agent_var.tau   = 0

                for t in range(T) :
                    t0, e0  = QN.next_time()
                    issn    = QN.g.ep['queues'][e0].departures[0].issn
                    agent   = self.agent_variables[issn]
                    tau     = agent.tau
                    s0      = QN.g.ep['queues'][e0].departures[0].od
                    state   = [s0]
                    print( "Frame: %s, %s, %s, %s" % (agent.issn, n, m, tau) )

                    if self.parked[issn] :
                        print("parked or something")
                        #agent.A_D  = [np.infty, np.infty]
                        #self.simulate_forward( QN )
                        #continue

                    state.extend(for_state)

                    if not COMPLETE_INFORMATION :
                        data  = agent.net_data[:, 1:3]
                        data  = data[:, 0] * data[:, 1]
                        for e in QN.g.edges() :
                            state[QN.g.edge_index[e] + 1] = data[QN.g.edge_index[e]]
                    else :
                        for e in QN.g.edges() :
                            state[QN.g.edge_index[e] + 1] = QN.g.ep['queues'][e].nSystem

                    obj_func[0] = self.parking_value(state[0][0], state[0][1], state, QN)
                    if QN.g.vp['vType'][QN.g.vertex( state[0][0] )] == 1 :
                        nServers = QN.g.ep['queues'][QN.g.edge(state[0][0], state[0][0])].nServers
                        if state[state[0][0]+1] == nServers :
                            obj_func[0] += self.full_penalty 
                    ct  = 1

                    for e in QN.g.vertex(state[0][0]).out_edges() :
                        if e.target() == QN.g.vertex(state[0][0]) :
                            continue
                        Sa              = self.post_decision_state(state, e, QN)
                        v, b            = self.value_function(Sa, agent.beta, QN)
                        value_es[ct]    = v
                        basis_es[:,ct]  = b
                        tmp             = self.expected_sojourn(e, QN, state)
                        obj_func[ct]    = tmp + value_es[ct] * gamma
                        target_node[ct] = int(e.target())
                        ct             += 1
                        print( "One step cost: %s\nNum in system: %s, %s" 
                            % (tmp, QN.g.ep['queues'][e].nSystem, int(e.target()) )  )

                    
                    old_t                   = QN.t
                    policy                  = int( np.argmin(obj_func[:ct]) )
                    agent.v_est[n,m,tau]    = value_es[policy]
                    agent.basis[n,m,tau,:]  = array( basis_es[:, policy].T )
                    print( "Options: %s" % (obj_func[:ct]) )
                    if policy == 0 :
                        print( 'parked' )
                        next_M  = True
                        agent.costs[n,m,tau]        = obj_func[0]
                        agent.parking               = True
                        self.agent_variables[issn]  = agent
                        self.parked[issn]           = True  ## Fix later
                        for parked in self.parked.values() :
                            if not parked :
                                next_M  = False
                                break
                        if next_M :
                            break

                    if save_frames :
                        self._update_graph(state, QN)
                        self.save_frame(self.dir['frames']+'sirs_%s_%s_%s_%s-0.png' % (agent.issn,n,m,tau), QN )
                        self._update_graph(state, QN, target_node[policy])
                        self.save_frame(self.dir['frames']+'sirs_%s_%s_%s_%s-1.png' % (agent.issn,n,m,tau), QN )

                    QN.g.ep['queues'][e0].departures[0].dest    = target_node[policy]
                    QN.g.ep['queues'][e0].departures[0].od[0]   = target_node[policy]
                    self.simulate_forward( QN )

                    agent.costs[n,m,tau]  = self.time2cost(QN.t - old_t)
                    agent.tau  += 1
                    print( "Realized cost: %s" % (agent.costs[n,m,tau-1]) )

                for agent in self.agent_variables.values() :
                    for t in range(T-1, -1,-1) :
                        agent.values[n,m,t] = agent.costs[n,m,t] + gamma * agent.values[n,m,t+1]

                    a, b  = self.update_theta(agent.beta, agent.Bmat, agent.basis[n,m,:,:], 
                                              agent.values[n,m,:], agent.v_est[n,m,:])
                    agent.beta_history[n,m,:]   = a
                    agent.value_history[n,m,:]  = agent.values[n,m,0], agent.v_est[n,m,0]
                    agent.beta  = a.T 
                    agent.Bmat  = b
                    print( array([agent.issn, agent.values[n,m,0], agent.v_est[n,m,0], np.sum(QN.nAgents)]) )
                    print("Agent : %s, Weights : %s" % (agent.issn, a))


    def setup_adp(self, nLearners) :
        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        for k in range(nLearners) :
            issn    = self.agent_cap+k+1
            self.agent_variables[issn] = Variable_Struct(issn, self.nFeatures, N, M, T)


    def random_state(self, orig, dest, H=(25, 75) ) :

        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        np.random.shuffle(self.ce)
        
        self.Qn.edges[self.ce[0]]
        self.Qn.reset()
        self.Qn.g.ep['queues'][self.Qn.edges[self.ce[0]]].active = True
        self.Qn.g.ep['queues'][self.Qn.edges[self.ce[0]]].add_arrival()
        self.Qn.simulate( np.random.randint(H[0], H[1]) )

        t, e0   = self.Qn.next_time()
        if not hasattr(orig, '__iter__') :
            orig    = [orig]
        if not hasattr(dest, '__iter__') :
            dest    = [dest]

        count   = 1
        for i in range(len(orig)) :
            for e in self.Qn.g.vertex(orig[i]).in_edges() :
                q   = self.Qn.g.ep['queues'][e]
                ei  = e
                break

            agent       = qt.LearningAgent(self.agent_cap+count, self.Qn.nE)
            agent.od    = [orig[i], dest[i]]
            agent.set_type(1)
            agent.set_dest(dest=dest[i])
            agent.update_information(q.net_data)
            agent.stamp(q.issn, q.nSystem, q.nServers, q.t)
            q.append_departure(agent, self.Qn.t)
            self.parked[self.agent_cap+count] = False
            print( self.parked )
            count  += 1

        if not np.array([self.Qn.edges[i] == e0 for i in orig]).any():
            self.simulate_forward( self.Qn )

        return



    ## time weighted updating
    def update_theta(self, v, B, basis, value, value_est ) :
        lam = 0.95
        x   = np.mat(basis[0,:]).T
        gam = lam + x.T * B * x
        H   = B / gam
        B   = (B - B * x * x.T * B / gam) / lam
        ips = value_est[0] - value[0]
        return np.squeeze(np.asarray(np.mat(v).T - ips * H * x)), B


    def update_theta2(self, v, B, basis, value, value_est ) :
        lam = 0.95
        ans = zeros( v.shape )
        for j in range(value_est.shape[0]) :
            if value_est[j] == 0:
                break
        for k in range(j) :
            ans += basis[k,:] * ( value[k] - np.sum(value_est[k:]) )
        return v - lam * ans, B


    def value_function(self, state, theta, QN) :
        #if len(theta) != self.nFeatures :
        #    print(("theta not of correct length: value_function", theta.shape))
        #    return

        value    = zeros(self.nFeatures)
        value[0] = theta[0]
        value[1] = theta[1] * self.basis_function1(state, QN)
        value[2] = theta[2] * self.basis_function2(state, QN)
        value[3] = theta[3] * self.basis_function3(state, QN)
        value[4] = theta[4] * self.basis_function4(state, QN)
        return sum(value), np.mat(value).T


    def basis_function1(self, S, QN) :
        return 1 * 0.25 * self.dist[ S[0][0], S[0][1] ]


    def basis_function2(self, S, QN, RETURN_PATH=False) :

        #for e in QN.g.edges() :
        #    if e.source() != e.target() :
        #        sojourn                     = self.expected_sojourn(e, QN, S)
        #        QN.g.ep['edge_times'][e]    = sojourn + 0.25 * QN.g.ep['edge_length'][e]

        origin  = QN.g.vertex(S[0][0])
        destin  = QN.g.vertex(S[0][1])
        #answer  = gt.shortest_path(QN.g, origin, destin, weights=QN.g.ep['edge_times'])[1]

        if RETURN_PATH :
            an  = answer
        else :
            an  = 0
            an  = 1
            #for e in answer :
            #    an += self.time2cost( QN.g.ep['edge_times'][e] )
        return an


    def basis_function3(self, S, QN, PRE_SET=False) :

        #indices = np.argsort( self.dist[S[0][1], QN.node_index['fcq']] )
        #garages = [QN.node_index['fcq'][k] for k in indices]

        #if len( garages ) > 4 :
        #    garages = garages[:4]

        #if not PRE_SET :
        #    for e in QN.g.edges() :
        #        if e.source() != e.target() :
        #            sojourn = self.expected_sojourn(e, QN, S)
        #            QN.g.ep['edge_times'][e] = sojourn + 0.25 * QN.g.ep['edge_length'][e]

        #an  = zeros( len(garages) )
        ct  = 0
        origin  = QN.g.vertex(S[0][0])

        for g in (1,):# garages:
            #destin  = QN.g.vertex(g)
            #answer  = gt.shortest_path(QN.g, origin, destin, weights=QN.g.ep['edge_times'])[1]
            #for e in answer :
            #    an[ct] += self.time2cost(QN.g.ep['edge_times'][e])
            #an[ct] += self.parking_value(g, S[0][1], S, QN)
            ct     += 1

        an = (1,)

        return min(an)


    def basis_function4(self, S, QN) :
        destination = QN.g.vertex(S[0][1]) 
        for v in QN.g.vertex(S[0][0]).out_neighbours():
            if v == destination :
                return 0
        v   = [self.parking_value(g, S[0][1], S, QN) for g in QN.node_index['fcq']]
        ans = np.min(v) 
        return ans



    def _update_graph(self, state, QN, target=None ) :
        QN.reset_colors(('all','basic'))

        v_props = set()
        for key in QN.g.vertex_properties.keys() :
            v_props = v_props.union( [key] )

        HAS_LIGHT = 'light' in v_props

        i, j    = state[0]
        vi, vj  = QN.g.vertex(i), QN.g.vertex(j)
        self.edge_text       = QN.g.new_edge_property("string")
        self.vertex_text     = QN.g.new_vertex_property("string")
        self.vertex_text[vj] = QN.g.vp['name'][vj]

        for e in QN.g.edges() :
            QN.g.ep['edge_width'][e]  = 1.25
            QN.g.ep['arrow_width'][e] = 4
            if e.target() != e.source() :
                QN.g.ep['edge_color'][e]    = [0.5, 0.5, 0.5, 0.45]
                QN.g.ep['edge_t_color'][e]  = [0.0, 0.0, 0.0, 0.45]
            else :
                QN.g.ep['edge_color'][e]    = [0.5, 0.5, 0.5, 0.0]
                QN.g.ep['edge_t_color'][e]  = [0.0, 0.0, 0.0, 0.0]
            if QN.g.ep['queues'][e].nSystem > 0 :
                self.edge_text[e] = str(QN.g.ep['queues'][e].nSystem)
            else :
                self.edge_text[e] = ''

        dest_color      = list(QN.colors['vertex'][2])
        gara_color      = list(QN.colors['vertex'][1])
        road_color      = list(QN.colors['vertex'][0])
        ligh_color      = list(QN.colors['vertex'][3])
        dest_color[-1]  = 1.25
        gara_color[-1]  = 1.25
        road_color[-1]  = 0.5
        ligh_color[-1]  = 1.25
        pen_width       = 1.2

        for v in QN.g.vertices() :
            QN.g.vp['vertex_pen_width'][v]  = pen_width
            if v == vj or v == vi :
                if v == vi :
                    QN.g.vp['vertex_pen_width'][v]  = 1.8
                    QN.g.vp['vertex_color'][v]      = [0.941, 0.502, 0.502, 1.0]
                continue
            if QN.g.vp['destination'][v] :
                QN.g.vp['vertex_color'][v]  = dest_color
            elif QN.g.vp['vType'][v] == 1 :
                QN.g.vp['vertex_color'][v]  = gara_color
            elif HAS_LIGHT and QN.g.vp['vType'][v] == 3 :
                QN.g.vp['vertex_color'][v]  = ligh_color
            else :
                QN.g.vp['vertex_color'][v]  = road_color

            if QN.g.vp['vType'][v] == 1 :
                QN.g.vp['vertex_t_color'][v]    = [0.0, 0.0, 0.0, 1.0]
                QN.g.vp['state'][v]             = QN.g.ep['queues'][QN.g.edge(v,v)].nSystem
            else :
                QN.g.vp['vertex_t_color'][v]    = [0.0, 0.0, 0.0, 0.0]

            halo_alpha                  = 1/self.parking_penalty[int(v), int(vj)]
            QN.g.vp['halo'][v]          = QN.g.vp['vType'][v] == 1
            QN.g.vp['halo_color'][v]    = [0.0, 0.502, 0.502, 0.15]
            #QN.g.vp['halo_color'][v]    = [0.0, 1.0, 0.0, halo_alpha/2]

        if target != None :
            target  = QN.g.vertex(target)
            QN.g.vp['halo'][target]         = True 
            QN.g.vp['halo_color'][target]   = [1.0, 1.0, 0.0, 0.5]
            QN.g.vp['halo'][vj]             = True

        for e in QN.g.vertex(state[0][1]).all_edges() :
            if QN.g.vp['vType'][e.target()] == 1 :
                QN.g.ep['edge_color'][e]    = [0.0, 0.5, 1.0, 1.0]
                QN.g.ep['edge_t_color'][e]  = [0.0, 0.0, 0.0, 1.0]

        a           = i
        edge_list   = []
        while a != j :
            a0  = QN.shortest_path[a, int(j)]
            edge_list.append( QN.g.edge(a, a0) )
            a   = a0

        for e in edge_list :
            QN.g.ep['edge_color'][e]    = [0.094, 0.180, 0.275, 1.0]
            QN.g.ep['edge_t_color'][e]  = [0.0, 0.0, 0.0, 1.0]
            QN.g.ep['edge_width'][e]    = 4
            QN.g.ep['arrow_width'][e]   = 9

        path = self.basis_function2(state, QN, True)
        for e in path :
            QN.g.ep['edge_color'][e]    = [0.008, 0.388, 0.820, 1.0] #[0.0, 0.5451, 0.5451, 1.0]
            QN.g.ep['edge_t_color'][e]  = [0.0, 0.0, 0.0, 1.0]
            QN.g.ep['edge_width'][e]    = 4
            QN.g.ep['arrow_width'][e]   = 9



    def save_frame(self, filename, QN) :
        gt.graph_draw(QN.g, QN.g.vp['pos'], 
                output_size=(1200, 1200), output=filename,
                bg_color=[1,1,1,1],#QN.colors['bg_color'],
                edge_color=QN.g.ep['edge_color'],
                edge_control_points=QN.g.ep['control'],
                edge_marker_size=QN.g.ep['arrow_width'],
                edge_pen_width=QN.g.ep['edge_width'],
                edge_text=self.edge_text,
                edge_font_size=QN.g.ep['edge_t_size'],
                edge_text_distance=QN.g.ep['edge_t_distance'],
                edge_text_parallel=QN.g.ep['edge_t_parallel'],
                edge_text_color=QN.g.ep['edge_t_color'],
                vertex_fill_color=QN.g.vp['vertex_color'],
                vertex_halo=QN.g.vp['halo'],
                vertex_halo_color=QN.g.vp['halo_color'],
                vertex_halo_size=QN.g.vp['vertex_halo_size'],
                vertex_pen_width=QN.g.vp['vertex_pen_width'],
                vertex_text=self.vertex_text,
                vertex_text_position=QN.g.vp['vertex_t_pos'],
                vertex_text_color=QN.g.vp['vertex_t_color'],
                vertex_font_size=QN.g.vp['vertex_t_size'],
                vertex_size=QN.g.vp['vertex_size'])





class Variable_Struct() :

    def __init__(self, issn, nF=1, N=1, M=1, T=1) :
        self.issn   = issn
        self.tau    = 0                 # Keeps track of iteration number in ADP
        self.setup_adp(nF, N, M, T)

    def setup_adp(self, nFeatures, N, M, T) :
        self.beta   = np.ones( nFeatures )
        self.Bmat   = np.mat( np.eye( nFeatures ) ) / 8
        self.values = zeros( (N,M,T+1) )
        self.v_est  = zeros( (N,M,T) )
        self.costs  = zeros( (N,M,T+1) )
        self.basis  = zeros( (N,M,T, nFeatures) )

        self.beta_history   = zeros( (N,M, nFeatures) )
        self.value_history  = zeros( (N,M,2) )




