import numpy                as np
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import graph_tool.all       as gt
import queueing_tool        as qt
import datetime
import time
import copy
import os

from numpy.random   import uniform, multinomial, exponential
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


class LearningAgent(qt.Agent) :

    def __init__(self, issn=0) :
        qt.Agent.__init__(self, issn)

    def __repr__(self) :
        return "LearningAgent. issn: %s, time: %s" % (self.issn, self.time)


    def get_beliefs(self) :
        return self.net_data[:, 2]


    def desired_destination(self, *info) :
        return self.dest


    def set_dest(self, net=None, dest=None) :
        self.dest = int(dest)



class approximate_dynamic_program :

    def __init__(self, g=None, nVertices=125, seed=None, agent_cap=1000) :
        self.parameters         = {'N': 10, 'M':10, 'T': 25, 'gamma': 0.975}
        self.t                  = 0
        self.animate            = False
        self.agent_cap          = agent_cap
        self.dist               = None
        self.parking_penalty    = None
        self.agent_variables    = {}
        self.parked             = {}
        self.dir                = {'frames' : './figures/frames/'}
        self.nInteractions      = 0

        if g == None :
            self.Qn = qt.QueueNetwork(nVertices=nVertices, calcpath=False, pDest=0.1, seed=seed)
            self.modify_network()
        elif isinstance(g, gt.Graph) or isinstance(g, str) :
            self.Qn = qt.QueueNetwork(g, calcpath=False, pDest=0.05, seed=seed)

        self.ce        = np.arange( self.Qn.nE )[self.Qn.g.ep['eType'].a==0]
        self.node_dict = {'fcq' : [], 'des' : [], 'arc' : [], 'fcq-des' : []}
        for v in self.Qn.g.vertices() :
            vi  = int(v)
            e   = self.Qn.g.edge(v, v)
            if self.Qn.g.vp['vType'][v] == 1 :
                self.node_dict['fcq'].append(vi)
                if isinstance(e, gt.Edge) :
                    ei = self.Qn.g.edge_index[e]
                    self.node_dict['fcq-des'].append(ei)
            elif self.Qn.g.vp['vType'][v] == 2 :
                self.node_dict['des'].append(vi)
                if isinstance(e, gt.Edge) :
                    ei = self.Qn.g.edge_index[e]
                    self.node_dict['fcq-des'].append(ei)
            else :
                self.node_dict['arc'].append(vi)

        self.nFeatures  = 4
        self.beta       = ones(self.nFeatures)
        self.basis      = zeros(self.nFeatures)

        def edge_index(e) :
            return self.Qn.g.edge_index[e]

        self.edge2node  = {edge_index(e) : int(e.target()) for e in self.Qn.g.edges()}
        self.node2edge  = {int(e.target()) : edge_index(e) for e in self.Qn.g.edges() if e.target() == e.source() }
        self.all_edges  = [ [i for i in map(edge_index, list(v.all_edges()))] for v in self.Qn.g.vertices() ]
        self.locations  = { self.Qn.g.edge_index[e] : set() for e in self.Qn.g.edges() }
        self.Qn.agent_cap = self.agent_cap
        self.calculate_parking_penalty2()


    def __repr__(self) :
        return 'whatever'


    def modify_network(self) :
        ep  = self.Qn.g.ep
        vp  = self.Qn.g.vp
        n   = sum( vp['vType'].a == 1 )
        m   = sum( vp['vType'].a == 2 )
        for q in self.Qn.edge2queue :
            if q.edge[0] == q.edge[1] :
                if vp['vType'].a[q.edge[0]] == 1 :
                    q.fArrival  = lambda x : x + exponential(1.0)
                    q.fDepart   = lambda x : x + exponential(1.333)
                    q.nServers  = max(self.agent_cap // (50 * n), 1)
                elif vp['vType'].a[q.edge[0]] == 2 :
                    q.fArrival  = lambda x : x + exponential(1.0)
                    q.fDepart   = lambda x : x + exponential(1.333)
                    q.nServers  = max(self.agent_cap // (125 * m), 1)


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

        nV    = self.Qn.nV
        dist  = zeros((nV, nV))
        short = np.ones( (nV, nV), int)
        spath = np.ones( (nV, nV), int)

        if 'dist' not in v_props :        
            dist  = zeros((nV, nV))
            for ve in self.Qn.g.vertices() :
                for we in self.Qn.g.vertices() :
                    v,w  = int(ve), int(we)
                    if v == w or dist[v, w] != 0 :
                        continue
                    tmp     = gt.shortest_path(self.Qn.g, ve, we, weights=self.Qn.g.ep['edge_length'])
                    path    = [int(v) for v in tmp[0]]
                    elen    = [self.Qn.g.ep['edge_length'][e] for e in tmp[1]]
                    for i in range(len(path) - 1):
                        for j in range(i+1, len(path)):
                            dist[path[i], path[j]] = sum(elen[i:j])

                    spath[path[:-1], path[-1]] = path[1:]

                    for j in range(1,len(path)-1) :
                        pa  = path[:-j]
                        spath[pa[:-1], pa[-1]] = pa[1:]

                    if not self.Qn.g.is_directed() :
                        path.reverse()
                        spath[path[:-1], path[-1]] = path[1:]

                        for j in range(1, len(path)-1) :
                            pa  = path[:-j]
                            spath[pa[:-1], pa[-1]] = pa[1:]

                short[v, :] = spath[v, :]

            r = np.arange(nV)
            short[r, r] = r
            self.Qn.shortest_path = short
        else :
            for v in self.Qn.g.vertices() :
                dist[int(v),:] = self.Qn.g.vp['dist'][v].a

        self.dist   = dist * 10
        pp          = np.exp(3 * dist) - 0.95
        pp[pp>100]  = 100

        for v in self.Qn.g.vertices() :
            if self.Qn.g.vp['vType'][v] not in [1, 2] :
                pp[int(v), :] = np.infty

        self.parking_penalty  = pp
        self.full_penalty     = 10


    def parking_value(self, origin, destination, S, QN) :
        e   = QN.g.edge(origin, origin)
        if isinstance(e, gt.Edge) :
            ei  = QN.g.edge_index[e]
            cap = QN.edge2queue[ei].nServers
            r   = S[ei+1] / cap
            p   = 1.0 / ( 1.0 + np.exp(-40*(r-19/20)) )
            ans = self.parking_penalty[origin, destination] + p * self.full_penalty
        else :
            ans = self.parking_penalty[origin, destination]

        return ans


    def random_state(self, nLearners, H=(25, 75) ) :

        dest  = list( np.random.choice(self.node_dict['des'], nLearners) )
        orig  = [np.argmax( self.dist[:, d] ) for d in dest] #list( np.random.choice(self.node_dict['arc'], nLearners) )

        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        np.random.shuffle(self.ce)

        starting_qs = self.ce[:(self.Qn.nV // 3)].tolist()
        starting_qs.extend( [self.Qn.in_edges[k][-1] for k in orig] )
        starting_qs = np.unique(starting_qs).tolist()

        self.Qn.clear()
        self.Qn.initialize(queues=starting_qs)
        self.Qn.simulate( np.random.randint(H[0], H[1]) )

        for i in range(nLearners) :
            aissn = self.agent_cap + i + 1
            agent = LearningAgent(aissn)

            for ei in self.Qn.in_edges[ orig[i] ] :
                self.locations[ei].add( aissn )

            q = self.Qn.edge2queue[ei]
            t = q.time + 1 if q.time < infty else self.Qn.queues[-1].time + 1
            agent.od  = [orig[i], dest[i]]
            agent.set_type(1)

            self.agent_variables[aissn].agent = agent
            self.Qn.add_arrival(ei, agent, t)
            self.parked[aissn] = False

        if not isinstance(self.Qn.queues[-1].departures[0], LearningAgent) and nLearners > 0 :
            self.simulate_forward( self.Qn )


    def _setup_adp(self, nLearners, save_frames=False) :
        if save_frames and not os.path.exists(self.dir['frames']) :
            os.mkdir(self.dir['frames'])
        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        for k in range(nLearners) :
            issn    = self.agent_cap+k+1
            self.agent_variables[issn] = AgentStruct(issn, self.nFeatures, N, M, T)


    # algorithm from pg 405 of Powell 2011
    # theta update from pg 350 of Powell 2011

    def approximate_policy_iteration(self, nLearners, complete_info=True, save_frames=False, verbose=False) :
        self.before = datetime.datetime.today()
        self._setup_adp( nLearners, save_frames )

        N, M, T = self.parameters['N'], self.parameters['M'], self.parameters['T']
        gamma   = self.parameters['gamma']
        nFCQs   = len(self.node_dict['fcq-des'])

        cost      = zeros(1 + nFCQs)
        value_es  = zeros(1 + nFCQs)
        obj_func  = zeros(1 + nFCQs)
        edges     = zeros(1 + nFCQs, int)
        basis_es  = np.mat(zeros( (self.nFeatures, 1 + nFCQs) ))
        for_state = [0 for k in range(self.Qn.nE)]

        for n in range(N) :
            self.random_state(nLearners, (30, 50) )

            if verbose :
                print( sum(self.Qn.nAgents) )

            for m in range(M) :
                QN  = self.Qn.copy()
                loc = copy.deepcopy( self.locations )

                finished = np.zeros(nLearners, bool)

                while not finished.all() :
                    agent   = QN.queues[-1].departures[0]
                    issn    = agent.issn
                    aStruct = self.agent_variables[issn]
                    tau     = aStruct.tau
                    state   = [agent.od]

                    if finished[issn - self.agent_cap - 1] :
                        self.simulate_forward(QN)
                        continue

                    if aStruct.parked[n, m] :
                        self.exchange_information( agent.od[0] ) 
                        self.simulate_forward(QN)
                        aStruct.tau  += 1
                        finished[issn - self.agent_cap - 1] = aStruct.tau >= T
                        continue

                    if verbose :
                        print("Frame: %s, %s, %s, %s" % (issn, n, m, tau) )

                    if not complete_info :
                        data  = agent.net_data[:, 1] * agent.net_data[:, 2]
                        state.extend( data )
                    else :
                        state.extend( QN.nAgents )

                    obj_func[0] = self.parking_penalty[state[0][0], state[0][1]]
                    if QN.g.vp['vType'].a[ state[0][0] ] in [1, 2] :
                        ei = QN.g.edge_index[QN.g.edge(state[0][0], state[0][0])]
                        edges[0] = ei
                        nServers = QN.edge2queue[ei].nServers
                        if state[ei+1] == nServers :
                            obj_func[0] += self.full_penalty 

                    for k in range(1, nFCQs + 1) :
                        ei    = self.node_dict['fcq-des'][k-1]
                        v, b  = self.value_function(state, ei, aStruct.beta[tau], QN)

                        obj_func[k]   = v * gamma
                        value_es[k]   = v
                        basis_es[:,k] = b
                        edges[k]      = ei
                        if verbose :
                            print( "Value function: %s\nNum in queue %s: %s" % (v, ei, QN.edge2queue[ei].nSystem) )

                    if 3 > n >= 0 and np.random.uniform() < 0.2 and m < M - 1 and False :
                        if obj_func[0] < np.infty :
                            policy = np.random.randint(0, nFCQs + 1)
                        else :  
                            policy = np.random.randint(1, nFCQs + 1)
                    else :
                        policy  = np.argmin( obj_func )

                    aStruct.t[n,m,tau]        = QN.t
                    aStruct.v_est[n,m,tau]    = value_es[policy]
                    aStruct.basis[n,m,tau,:]  = array( basis_es[:, policy].T )
                    next_node   = QN.shortest_path[ state[0][0], self.edge2node[edges[policy]] ]

                    if policy == 0 :
                        if verbose :
                            print( "Options: %s\nParked!" % (obj_func) )

                        aStruct.costs[n, m, tau]  = obj_func[0]
                        aStruct.parked[n, m]      = True

                    if save_frames and m == M - 1 :
                        self._update_graph(state, QN)
                        self.save_frame(self.dir['frames']+'sirs_%s_%s_%s_%s-0.png' % (issn,n,m,tau), QN)
                        self._update_graph(state, QN, next_node)
                        self.save_frame(self.dir['frames']+'sirs_%s_%s_%s_%s-1.png' % (issn,n,m,tau), QN)

                    if n > 0 :
                        for ei in self.Qn.in_edges[ state[0][0] ] :
                            loc[ei].remove( issn )

                        for ei in self.Qn.in_edges[ next_node ] :
                            loc[ei].add( issn )

                        self.exchange_information( next_node ) 

                    agent.dest   = QN.g.edge_index[ QN.g.edge(agent.od[0], next_node) ]
                    agent.od[0]  = next_node
                    aStruct.tau += 1

                    self.simulate_forward(QN)
                    finished[issn - self.agent_cap - 1] = aStruct.tau >= T
                    if tau + 1 == T and policy != 0:
                        aStruct.costs[n, m, tau] = 100

                    if verbose and tau > 0 :
                        print( "Options: %s\nPolicy, edge: %s, %s" % (obj_func[:ct+1], policy, edges[policy]) )

                for aStruct in self.agent_variables.values() :
                    for t in range(T - 1, -1, -1) :
                        aStruct.values[n,m,t] = aStruct.costs[n,m,t] + gamma * aStruct.values[n,m,t+1]

                    print([n, m, np.argmax(aStruct.values[n,m,:]), max(aStruct.values[n,m,:])])
                    if n > 0 :
                        self.update_beta(aStruct, n, m)
                    elif n == 0 and m == M - 1 :
                        self.initial_update(aStruct)

                    aStruct.tau = 0
                    aStruct.beta_history[n,m,:,:]   = aStruct.beta
                    aStruct.value_history[n,m,:,0]  = aStruct.values[n,m,:T]
                    aStruct.value_history[n,m,:,1]  = aStruct.v_est[n,m,:]
                    if verbose :
                        print( array([aStruct.issn, aStruct.values[n,m,0,0], aStruct.v_est[n,m,0,0]]) )
                        print("Agent : %s, Weights : %s" % (aStruct.issn, aStruct.beta[0]))

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
        S[0][0] = QN.edge2queue[ei].edge[1]
        return S


    def simulate_forward(self, QN) :
        QN.simulate(n=1)
        event = QN.next_event_type()
        while not (event == 2 and isinstance(QN.queues[-1].departures[0], LearningAgent) ):
            QN.simulate(n=1)
            event = QN.next_event_type()


    def exchange_information(self, node) :
        issns = set()

        for ei in self.Qn.in_edges[node] :
            for n in self.locations[ei] :
                issns.add(n)

        if len(issns) > 1 :
            T = self.parameters['T']
            n = 0
            for issn in issns :
                n  += self.agent_variables[issn].n

            for t in range(T) :
                self.nInteractions += 1
                A = np.zeros( (self.nFeatures, self.nFeatures) )
                z = np.zeros( self.nFeatures )
                p = 0

                for issn in issns :
                    aStruct = self.agent_variables[issn]
                    p   = aStruct.n / n
                    A  += aStruct.A[t] * p
                    z  += aStruct.z[t] * p

                for issn in issns :
                    self.agent_variables[issn].A[t] = A
                    self.agent_variables[issn].z[t] = z

            for issn in issns :
                self.agent_variables[issn].n = n


    def initial_update(self, aStruct) :
        T   = self.parameters['T']
        for t in range(T) :
            y   = aStruct.values[0, :, t]
            X   = aStruct.basis[0,:,t,:]
            A   = dot(X.T, X)

            aStruct.A[t]    = A
            aStruct.z[t]    = dot(X.T, y)
            aStruct.beta[t] = dot(pinv(A), aStruct.z[t])


    def update_beta(self, aStruct, n, m) :
        T   = self.parameters['T']
        for t in range(T) :
            A   = aStruct.A[t]
            z   = aStruct.z[t]
            Phi = aStruct.basis[n,m,t,:]
            c   = aStruct.values[n,m,t]

            aStruct.A[t]    = A + np.outer(Phi, Phi)
            aStruct.z[t]    = z + c * Phi
            aStruct.beta[t] = dot(pinv(aStruct.A[t]), aStruct.z[t])


    def value_function(self, S, e, beta, QN) :
        v     = self.edge2node[e]
        cost  = self.parking_penalty[v, S[0][1]]
        
        nSer  = QN.edge2queue[e].nServers
        nSys  = QN.edge2queue[e].nSystem

        p    = 1 / (1 + max(nSer - nSys, 0))
        l    = sum([QN.edge2queue[ei].nSystem for ei in self.Qn.in_edges[v] if ei != e]) / len(self.Qn.in_edges[v])
        penl = self.full_penalty / ( 1.0 + np.exp(-40*(min(l,1)-19/20)) )
        pen  = self.full_penalty / ( 1.0 + np.exp(-40*(p-19/20)) )

        self.basis[0] = 1
        self.basis[1] = cost
        self.basis[2] = pen
        self.basis[3] = penl

        return np.dot(self.basis, beta), np.mat(self.basis).T


    def _update_graph(self, state, QN, target=None ) :
        QN.update_graph_colors()
        QN.g.ep['edge_width'].a  = 1.25
        QN.g.ep['arrow_width'].a = 8
        QN.g.vp['halo'].a        = False

        i, j  = state[0]

        if target != None :
            target  = QN.g.vertex(target)
            QN.g.vp['halo'][target]         = True 
            QN.g.vp['halo_color'][target]   = [0.7, 0.1, 0.1, 1.0]

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
        self.t      = zeros( (N,M,T) )
        self.A      = ones( (T, nF, nF) )
        self.z      = zeros( (T, nF) )
        self.n      = 1
        self.beta   = ones( (T, nF) )
        self.values = zeros( (N,M,T+1) )
        self.v_est  = zeros( (N,M,T) )
        self.costs  = zeros( (N,M,T+1) )
        self.basis  = zeros( (N,M,T, nF) )
        self.parked = zeros( (N,M), bool )

        self.scale_vector   = zeros(nF)
        self.beta_cov       = np.eye(nF) * 2
        self.beta_history   = zeros( (N, M, T, nF) )
        self.value_history  = zeros( (N, M, T, 2) )

        for t in range(T) :
            self.A[t] = np.eye(nF)

        #for k in range(1, nF, 2) :
        #    self.beta[:, k] = 2 / (nF-1)



