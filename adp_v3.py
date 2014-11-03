import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import graph_tool.all       as gt
import queueing_tool        as qt
import datetime
import time
import copy
import os

from numpy.random   import uniform, multinomial
from numpy.linalg   import pinv
from numpy          import size, ones, zeros, array, ndarray, transpose, vstack, arange
from numpy          import logical_and, logical_or, logical_not, infty, log, dot
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

    def __init__(self, g=None, nVertices=125, seed=None) :
        self.parameters         = {'N': 10, 'M':10, 'T': 25, 'gamma':0.975}
        self.t                  = 0
        self.animate            = False
        self.agent_cap          = 50
        self.dist               = None
        self.parking_penalty    = None
        self.agent_variables    = {}
        self.parked             = {}
        self.dir                = {'frames' : './figures/frames/'}
        self.nInteractions      = 0

        if g == None :
            #self.Qn = self.activate_network(agent_cap=self.agent_cap, seed=seed)
            self.Qn = qt.QueueNetwork(nVertices=nVertices, calcpath=True, seed=seed)
        elif isinstance(g, gt.Graph) or isinstance(g, str) :
            self.Qn = qt.QueueNetwork(g, calcpath=True, seed=seed)

        self.nE = self.Qn.nE
        self.ce = np.arange( self.Qn.nV )[self.Qn.g.vp['vType'].a==0]

        self.node_dict = {'fcq' : [], 'des' : [], 'arc' : []}
        for v in self.Qn.g.vertices() :
            if self.Qn.g.vp['vType'][v] == 1 :
                self.node_dict['fcq'].append(int(v))
            elif self.Qn.g.vp['vType'][v] == 2 :
                self.node_dict['des'].append(int(v))
            else :
                self.node_dict['arc'].append(int(v))

        self.nGar       = min( [len(self.node_dict['fcq']), 4] )
        self.nFeatures  = 3 * self.nGar + 4
        self.beta       = 0.5 * ones(self.nFeatures)
        self.basis      = zeros(self.nFeatures)

        def edge_index(e) :
            return self.Qn.g.edge_index[e]

        self.in_edges   = [ [i for i in map(edge_index, list(v.in_edges()))] for v in self.Qn.g.vertices() ]
        self.all_edges  = [ [i for i in map(edge_index, list(v.all_edges()))] for v in self.Qn.g.vertices() ]
        self.locations  = { self.Qn.g.edge_index[e] : set() for e in self.Qn.g.edges() }
        self.calculate_parking_penalty2()


    def __repr__(self) :
        return 'whatever'


    def activate_network(self, agent_cap, net_size=150, seed=None) :
        Qn  = qt.QueueNetwork(nVertices=net_size, graph_type='periodic', seed=seed)
        Qn.agent_cap = agent_cap
        for q in [Qn.g.ep['queues'][e] for e in Qn.g.edges()] :
            q.fArrival    = lambda x : x + exponential(1.0)
            q.fDepart     = lambda x : x + exponential(0.333)
            q.fDepart_mu  = lambda x : 1/3 

        tmp0    = Qn.g.vp['vType'].a
        self.ce = np.arange( Qn.nV )[tmp0==min(tmp0)] # Creation edge
        Qn.initialize(queues=self.ce)
        
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
                    Qn.g.ep['queues'][e].fArrival   = lambda x : x + exponential(0.125)
                    Qn.g.ep['queues'][e].fDepart    = lambda x : x + exponential(0.333)
                    Qn.g.ep['queues'][e].fDepart_mu = lambda x : 0.333

        for e in Qn.g.edges() :
            if Qn.g.ep['eType'][e] == 1 :
                Qn.g.ep['queues'][e].set_nServers(garage_cap[ct])
                Qn.g.ep['queues'][e].fDepart    = lambda x : x + exponential(2)
                Qn.g.ep['queues'][e].fDepart_mu = lambda x : 2
                ct += 1

        return Qn


    def time2cost(self, t) :
        return t


    def length2cost(self, l) :
        return l

    ## Calculates it incorrectly, fix
    def expected_sojourn(self, ei, QN, S=None, time2cost=True) : 
        exp_s   = copy.deepcopy(QN.t)

        if S == None :
            kk  = max((QN.edge2queue[ei].nSystem - QN.edge2queue[ei].nServers, 0)) + 1
        else :
            kk  = max((S[ei+1] - QN.edge2queue[ei].nServers, 0)) + 1

        for k in range(int(kk)) :
            exp_s   += QN.edge2queue[ei].fDepart_mu(exp_s)

        return exp_s - QN.t


    def simple_model(self, act, QN, state) :

        exp_t       = self.expected_sojourn(act, QN, state)
        exp_depart  = QN.g.new_edge_property("int")
        exp_state   = QN.g.new_edge_property("int")

        for e in QN.g.edges() :
            exp_depart[e] = -1

        for e in QN.g.edges() :
            exp_state[e] += state[QN.g.edge_index[e]+1]
            dum_t         = QN.t

            while dum_t <= exp_t :
                dum_t          += QN.g.ep['queues'][e].fDepart_mu(dum_t)
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
                    dist[k, j] = min((np.exp(1.5 * dist[k, j]) - 1, 1000))

        self.parking_penalty  = np.abs(dist) * 10
        self.full_penalty     = 10


    def calculate_parking_penalty2(self) :

        v_props = set()
        for key in self.Qn.g.vertex_properties.keys() :
            v_props = v_props.union([key])

        dist    = zeros((self.Qn.nV, self.Qn.nV))

        if 'dist' not in v_props :        
            dist  = zeros((self.Qn.nV, self.Qn.nV))
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

            dist += np.transpose( dist ) 
        else :
            for v in self.Qn.g.vertices() :
                dist[int(v),:] = self.Qn.g.vp['dist'][v].a

        self.dist   = dist * 10
        pp          = np.exp(6 * dist) - 1
        pp[pp>100]  = 100

        for v in self.Qn.g.vertices() :
            if self.Qn.g.vp['vType'][v] not in [1, 2] :
                pp[int(v), :] = 800

        self.parking_penalty  = np.abs(pp) * 10
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


    def random_state(self, nLearners, H=(25, 75) ) :

        dest  = list( np.random.choice(self.node_dict['des'], nLearners) )
        orig  = list( np.random.choice(self.node_dict['arc'], nLearners) )

        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        np.random.shuffle(self.ce)

        starting_qs = self.ce[:(self.Qn.nV // 3)].tolist()
        starting_qs.extend( orig )
        starting_qs = np.unique(starting_qs).tolist()

        self.Qn.clear()
        self.Qn.initialize(queues=starting_qs)
        self.Qn.simulate( np.random.randint(H[0], H[1]) )

        count   = 1
        for i in range(nLearners) :
            aissn = self.agent_cap + count
            agent = qt.LearningAgent(aissn)

            for ei in self.in_edges[ orig[i] ] :
                self.locations[ei].add( aissn )

            q = self.Qn.edge2queue[ei]
            t = q.time + 1 if q.time < infty else self.Qn.queues[0].time + 1
            agent.od  = [orig[i], dest[i]]
            agent.set_type(1)
            agent.set_dest(dest=dest[i])

            self.agent_variables[aissn].agent = [agent]
            self.Qn.add_arrival(ei, agent, t)
            self.parked[self.agent_cap+count] = False
            count  += 1

        if not isinstance(self.Qn.queues[0].departures[0], qt.LearningAgent) :
            self.simulate_forward( self.Qn )

        return


    def _setup_adp(self, nLearners, save_frames=False) :
        if save_frames and not os.path.exists(self.dir['frames']) :
            os.mkdir(self.dir['frames'])
        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        for k in range(nLearners) :
            issn    = self.agent_cap+k+1
            self.agent_variables[issn] = AgentStruct(issn, self.nFeatures, N, M, T)


    # algorithm from pg 405 of Powell 2011
    # theta update from pg 349 of Powell 2011

    def approximate_policy_iteration(self, nLearners, complete_info=True, save_frames=False, verbose=False) :
        self.before = datetime.datetime.today()
        self._setup_adp( nLearners, save_frames )

        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        gamma   = self.parameters['gamma']

        cost      = zeros(1 + self.nE)
        value_es  = zeros(1 + self.nE)
        obj_func  = zeros(1 + self.nE)
        edges     = zeros(1 + self.nE, int)
        basis_es  = np.mat(zeros( (self.nFeatures, 1 + self.nE) ))
        for_state = [0 for k in range(self.Qn.nE)]

        for n in range(N) :
            self.random_state(nLearners, (30, 50) )

            if verbose :
                print( sum(self.Qn.nAgents) )

            for m in range(M) :
                QN  = self.Qn.copy()

                finished = np.zeros(nLearners, bool)

                while not finished.all() :
                    issn    = QN.queues[0].departures[0].issn
                    aStruct = self.agent_variables[issn]
                    tau     = aStruct.tau
                    state   = [aStruct.agent[0].od]

                    if finished[issn - self.agent_cap - 1] :
                        self.simulate_forward(QN)
                        continue

                    if aStruct.parked[n, m] :
                        self.exchange_information( aStruct.agent[0].od[0] ) 
                        self.simulate_forward(QN)
                        aStruct.tau  += 1
                        finished[issn - self.agent_cap - 1] = aStruct.tau >= T
                        continue

                    if verbose :
                        print("Frame: %s, %s, %s, %s" % (issn, n, m, tau) )

                    if not complete_info :
                        data  = aStruct.agent[0].net_data[:, 1] * aStruct.agent[0].net_data[:, 2]
                        state.extend( data )
                    else :
                        state.extend( QN.nAgents )

                    obj_func[0] = self.parking_value(state[0][0], state[0][1], state, QN)
                    if QN.g.vp['vType'].a[ state[0][0] ] in [1, 2] :
                        nServers = QN.g.ep['queues'][QN.g.edge(state[0][0], state[0][0])].nServers
                        if state[state[0][0]+1] == nServers :
                            obj_func[0] += self.full_penalty 
                    ct  = 0

                    for ei in QN.adjacency[ state[0][0] ] :
                        if QN.edge2queue[ei].issn[0] == QN.edge2queue[ei].issn[1] :
                            continue

                        ct   += 1
                        Sa    = self.post_decision_state(state, ei, QN)
                        v, b  = self.value_function(Sa, aStruct.beta, QN)

                        obj_func[ct]    = v * gamma
                        value_es[ct]    = v
                        basis_es[:, ct] = b
                        edges[ct]       = ei
                        if verbose :
                            print( "Value function: %s\nNum in queue %s: %s" 
                            % (v, ei, QN.edge2queue[ei].nSystem) )

                    
                    if np.random.uniform() < 0.15 and m < 100 :
                        if obj_func[0] < 1000 :
                            policy = np.random.randint(0, ct+1)
                        else :  
                            policy = np.random.randint(1, ct+1)
                    else :
                        policy  = np.argmin( obj_func[:ct+1] )

                    aStruct.t[n,m,tau]        = QN.t
                    aStruct.v_est[n,m,tau]    = value_es[policy]
                    aStruct.basis[n,m,tau,:]  = array( basis_es[:, policy].T )
                    target_node = QN.edge2queue[ edges[policy] ].issn[1]

                    if policy == 0 :
                        if verbose :
                            print( "Options: %s\nParked!" % (obj_func[:ct+1]) )

                        aStruct.costs[n, m, tau]  = obj_func[0]
                        aStruct.parked[n, m]      = True

                    if save_frames and m == M - 1 :
                        self._update_graph(state, QN)
                        self.save_frame(self.dir['frames']+'sirs_%s_%s_%s_%s-0.png' % (aStruct.issn,n,m,tau), QN)
                        self._update_graph(state, QN, target_node)
                        self.save_frame(self.dir['frames']+'sirs_%s_%s_%s_%s-1.png' % (aStruct.issn,n,m,tau), QN)

                    for ei in self.in_edges[ state[0][0] ] :
                        self.locations[ei].remove( aStruct.issn )

                    for ei in self.in_edges[ target_node ] :
                        self.locations[ei].add( aStruct.issn )

                    self.exchange_information( target_node ) 

                    aStruct.agent[0].dest  = edges[policy]
                    aStruct.agent[0].od[0] = target_node
                    aStruct.tau  += 1

                    self.simulate_forward(QN)

                    finished[issn - self.agent_cap - 1] = aStruct.tau >= T
                    if tau + 1 == T and policy != 0:
                        aStruct.costs[n, m, tau] = 100

                    if verbose and tau > 0 :
                        print( "Options: %s\nPolicy, edge: %s, %s" % (obj_func[:ct+1], policy, edges[policy]) )

                for aStruct in self.agent_variables.values() :
                    for t in range(aStruct.tau - 1, -1, -1) :
                        aStruct.values[n,m,t] = aStruct.costs[n,m,t] + gamma * aStruct.values[n,m,t+1]

                    print([n, m, aStruct.values[n,m,0]])
                    aStruct = self.update_beta(aStruct, n, m)

                    aStruct.tau = 0
                    aStruct.beta_history[n,m,:]   = aStruct.beta
                    aStruct.value_history[n,m,:]  = aStruct.values[n,m,0], aStruct.v_est[n,m,0]
                    if verbose :
                        print( array([aStruct.issn, aStruct.values[n,m,0], aStruct.v_est[n,m,0]]) )
                        print("Agent : %s, Weights : %s" % (aStruct.issn, aStruct.beta))

                nn  = []
                for aStruct in self.agent_variables.values() :
                    nn.append(aStruct.n)

                nnn = min(nn)
                for aStruct in self.agent_variables.values() :
                    aStruct.n = aStruct.n // nnn

        self.after = datetime.datetime.today()


    def post_decision_state(self, state, ei, QN) :
        S     = copy.deepcopy(state)
        k     = S[0][0] + 1
        a     = ei + 1
        S[k]  = S[k] - 1 if S[k] > 0 else 0
        S[a] += 1
        S[0][0] = QN.edge2queue[ei].issn[1]
        return S


    def simulate_forward(self, QN):
        QN.simulate(N=1)
        event = QN.next_event_type()
        while not (event == 2 and isinstance(QN.queues[0].departures[0], qt.LearningAgent) ):
            QN.simulate(N=1)
            event = QN.next_event_type()


    def exchange_information(self, node) :
        issns = set()

        for ei in self.in_edges[node] :
            for n in self.locations[ei] :
                issns.add(n)

        if len(issns) > 1 :
            self.nInteractions += 1
            A = np.zeros( (self.nFeatures, self.nFeatures) )
            z = np.zeros( self.nFeatures )
            n = 0
            p = 0
            for issn in issns :
                n  += self.agent_variables[issn].n

            for issn in issns :
                aStruct = self.agent_variables[issn]
                p   = aStruct.n / n
                A  += aStruct.A * p
                z  += aStruct.z * p

            for issn in issns :
                self.agent_variables[issn].A = A
                self.agent_variables[issn].z = z
                self.agent_variables[issn].n = n


    def update_beta(self, aStruct, n, m) :
        A   = aStruct.A
        z   = aStruct.z
        Phi = aStruct.basis[n,m,0,:]
        c   = aStruct.values[n,m,0]

        aStruct.A     = A + np.outer(Phi, Phi)
        aStruct.z     = z + c * Phi
        aStruct.beta  = dot(pinv(aStruct.A), aStruct.z)
        return aStruct


    def value_function(self, S, beta, QN) :
        indices = np.argsort( self.dist[S[0][1], self.node_dict['fcq']] )
        garages = [self.node_dict['fcq'][k] for k in indices][:self.nGar]
        c_dists = self.dist[S[0][0], garages]
        g_dists = self.dist[S[0][1], garages]
        
        val  = zeros(self.nFeatures)
        ej   = QN.g.edge_index[ QN.g.edge(S[0][1], S[0][1]) ]
        nSer = QN.edge2queue[ej].nServers
        nSys = QN.edge2queue[ej].nSystem

        p    = 1 / (1 + max(nSer - nSys, 0))
        l    = sum([QN.edge2queue[ei].nSystem for ei in self.in_edges[S[0][1]] if ei != S[0][1]]) / np.sum(QN.nAgents)
        cd   = self.dist[S[0][0], S[0][1]]

        self.basis[-4] = cd * p * self.full_penalty
        self.basis[-3] = cd * l
        self.basis[-2] = cd * l * p * self.full_penalty
        self.basis[-1] = 1

        for k in range(self.nGar) :
            g    = garages[k]
            gd   = g_dists[k]
            cd   = c_dists[k]
            nSer = QN.edge2queue[g].nServers
            nSys = QN.edge2queue[g].nSystem
            p    = 1 / (1 + max(nSer - nSys, 0))
            l    = sum([QN.edge2queue[ei].nSystem for ei in self.in_edges[g] if ei != g]) / np.sum(QN.nAgents)
            self.basis[3*k]    = cd * gd * p * self.full_penalty + self.parking_penalty[g, S[0][1]]
            self.basis[3*k+1]  = cd * gd * l + self.parking_penalty[g, S[0][1]]
            self.basis[3*k+2]  = cd * gd * l * p * self.full_penalty + self.parking_penalty[g, S[0][1]]

        return np.dot(self.basis, beta), np.mat(self.basis).T


    def _update_graph(self, state, QN, target=None ) :
        QN.update_graph_colors()
        QN.g.ep['edge_width'].a  = 1.25
        QN.g.ep['arrow_width'].a = 8
        QN.g.vp['halo'].a        = False

        i, j  = state[0]
        vj    = QN.g.vertex(j)

        if target != None :
            target  = QN.g.vertex(target)
            QN.g.vp['halo'][target]         = True 
            QN.g.vp['halo_color'][target]   = [1.0, 0.5, 0.0, 0.5]
            QN.g.vp['halo'][vj]             = True

        a     = i
        eList = []
        while a != j :
            a0  = QN.shortest_path[a, int(j)]
            eList.append( QN.g.edge(a, a0) )
            a   = a0

        for e in eList :
            QN.g.ep['edge_color'][e]    = [0.094, 0.180, 0.275, 1.0]
            QN.g.ep['edge_width'][e]    = 4
            QN.g.ep['arrow_width'][e]   = 9


    def save_frame(self, filename, QN) :
        QN.draw(output=filename, update_colors=False)



class AgentStruct() :

    def __init__(self, issn, nF=1, N=1, M=1, T=1, agent=None) :
        self.agent  = [agent]
        self.issn   = issn
        self.tau    = 0                 # Keeps track of iteration number in ADP
        self.t      = np.zeros( (N,M,T) )
        self.A      = np.eye(nF) * 2
        self.z      = np.zeros(nF)
        self.n      = 1
        self.beta   = np.ones(nF) / nF # np.array([0.34, 0.013, 0.006, 0.351, 0.006, 0.003, 0.607, 0.011, 0.011, 0.285, 0.044, 0.001, 0.35, 0.003, 0.0, 0.351])
        self.Bmat   = np.mat( np.eye(nF) ) / 8
        self.values = zeros( (N,M,T+1) )
        self.v_est  = zeros( (N,M,T) )
        self.costs  = zeros( (N,M,T+1) )
        self.basis  = zeros( (N,M,T, nF) )
        self.parked = zeros( (N,M), bool )

        self.scale_vector   = zeros(nF)
        self.beta_cov       = np.eye(nF) * 2
        self.beta_history   = zeros( (N, M, nF) )
        self.value_history  = zeros( (N, M, 2) )

# np.array([0.34, 0.013, 0.006, 0.351, 0.006, 0.003, 0.607, 0.011, 0.011, 0.285, 0.044, 0.001, 0.35, 0.003, 0.0, 0.351])


