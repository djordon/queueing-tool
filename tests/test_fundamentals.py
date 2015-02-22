import numpy    as np
import queueing_tool  as qt
import graph_tool.all as gt

import unittest
import numbers

from numpy.random import randint




class TestQueueServers(unittest.TestCase) :

    def setUp(self) :
        self.lam = np.random.randint(1,10)
        self.rho = np.random.uniform(0.5, 1)


    def test_QueueServer_accounting(self) :

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        arr = lambda t : t + np.random.exponential(1/self.lam)
        ser = lambda t : t + np.random.exponential(1 / mu)

        q   = qt.QueueServer(nServers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        nEvents = 5000
        
        ans = np.zeros((nEvents,2), bool)

        for k in range(nEvents) :
            nt = len(q._departures) + len(q._queue) + len(q._arrivals) - 2
            nS = len(q._departures) + len(q._queue) - 1
            ans[k,0] = nt == q._nTotal
            ans[k,1] = nS == q.nSystem
            q.simulate(n=1)

        self.assertTrue( ans.all() )


    def test_QueueServer_simulation(self) :

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        arr = lambda t : t + np.random.exponential(1/self.lam)
        ser = lambda t : t + np.random.exponential(1 / mu)

        q   = qt.QueueServer(nServers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        nEvents = 5000
        
        ans = np.zeros(4, bool)

        k   = np.random.randint(nEvents * 0.75, nEvents * 1.25)
        nA0 = q._oArrivals
        nD0 = q.nDepartures
        q.simulate(n=k)
        ans[0] = q.nDepartures + q._oArrivals - nA0 - nD0 == k

        k   = np.random.randint(nEvents * 0.75, nEvents * 1.25)
        nA0 = q._oArrivals
        q.simulate(nA=k)
        ans[1] = q._oArrivals - nA0 == k

        k   = np.random.randint(nEvents * 0.75, nEvents * 1.25)
        nD0 = q.nDepartures
        q.simulate(nD=k)
        ans[2] = q.nDepartures - nD0 == k

        t  = 100 * np.random.uniform(0.5, 1)
        t0 = q.current_time
        q.simulate(t=t)
        ans[3] = q.current_time - t0 >= t

        self.assertTrue( ans.all() )


    def test_LossQueue_accounting(self) :

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        arr = lambda t : t + np.random.exponential(1/self.lam)
        ser = lambda t : t + np.random.exponential(1 / mu)

        q   = qt.LossQueue(nServers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        nEvents = 5000
        
        ans = np.zeros((nEvents,2), bool)

        for k in range(nEvents) :
            nt = len(q._departures) + len(q._queue) + len(q._arrivals) - 2
            nS = len(q._departures) + len(q._queue) - 1
            ans[k,0] = nt == q._nTotal
            ans[k,1] = nS == q.nSystem
            q.simulate(n=1)

        self.assertTrue( ans.all() )


    def test_LossQueue_blocking(self) :

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        k   = np.random.randint(5, 15)
        scl = 1 / (mu * k)

        arr = lambda t : t + np.random.exponential(1/self.lam)
        ser = lambda t : t + np.random.gamma(k, scl)

        q  = qt.LossQueue(nServers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        nE = 500
        c  = 0
        ans = np.zeros(nE, bool)

        while c < nE :
            if q.next_event_description() == 1 and q.at_capacity() :
                nB0 = q.nBlocked
                q.simulate(n=1)
                ans[c] = nB0 + 1 == q.nBlocked
                c += 1
            else :
                q.simulate(n=1)

        tmp = np.ones(5)
        self.assertTrue( ans.all() )



class TestQueueNetwork(unittest.TestCase) :

    def setUp(self) :
        self.g   = qt.generate_random_graph(300)
        self.qn  = qt.QueueNetwork(self.g)

    def test_QueueNetwork_sorting(self) :

        self.qn.agent_cap = 3000
        self.qn.initialize(50)
        self.qn.simulate(n=10000)

        nEvents = 1000
        ans = np.zeros(nEvents, bool)
        for k in range(nEvents) :
            net_times   = np.array([q.time for q in self.qn._queues])
            queue_times = [q.time for q in self.qn.edge2queue]
            queue_times.sort()
            while queue_times[-1] == np.infty :
                tmp = queue_times.pop()

            queue_times.sort(reverse=True)

            ans[k] = (queue_times == net_times).all()
            self.qn.simulate(n=1)

        self.assertTrue( ans.all() )


    def test_QueueNetwork_closedness(self) :

        self.qn.agent_cap = 3000
        self.qn.initialize(50)

        nEvents = 2500

        ans = np.zeros(nEvents, bool)
        na  = np.zeros(self.qn.nE, int)
        for q in self.qn.edge2queue :
            na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q._queue) - 2

        for k in range(nEvents) :
            ans[k] = np.sum(self.qn.nAgents) >= np.sum(na)
            for q in self.qn.edge2queue :
                na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q._queue) - 2

            self.qn.simulate(n=1)

        self.assertTrue( ans.all() )


    def test_QueueNetwork_accounting(self) :

        self.qn.agent_cap = 3000
        self.qn.initialize(50)

        nEvents = 2500

        ans = np.zeros(nEvents, bool)
        na  = np.zeros(self.qn.nE, int)
        for q in self.qn.edge2queue :
            na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q._queue) - 2

        for k in range(nEvents) :
            ans[k] = (self.qn.nAgents == na).all()
            self.qn.simulate(n=1)
            for q in self.qn.edge2queue :
                na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q._queue) - 2

        self.assertTrue( ans.all() )


    def test_QueueNetwork_routing(self) :

        adj = {0 : 1, 1 : [2, 3]}
        g   = qt.adjacency2graph(adj)
        qn  = qt.QueueNetwork(g)
        mat = qt.generate_transition_matrix(g)
        qn.set_transitions(mat)

        qn.initialize(edge=(0,1))
        qn.collect_data(edge=[(1,2), (1,3)])

        qn.simulate(500000)

        data = qn.data_queues(edge=[(1,2), (1,3)])
        e1, e2 = qn.out_edges[1]

        p1 = np.sum(data[:, 4] == e1) / data.shape[0]
        p2 = np.sum(data[:, 4] == e2) / data.shape[0]

        trans = qn.transitions(False)
        ans1  = np.round(trans[1], 2)
        ans2  = np.round([p1, p2], 2)

        self.assertTrue( (ans1 == ans2).all() )



def generate_adjacency(a=3, b=25, c=6, n=12) :
    return {k : list(randint(a, b, randint(1,c))) for k in np.unique(randint(a, b, n))}



class TestGraphFunctions(unittest.TestCase) :

    def test_graph2dict(self) :
        adj = generate_adjacency()
        g1  = qt.adjacency2graph(adj,adjust=1)
        aj1 = qt.graph2dict(g1)
        g2  = qt.adjacency2graph(aj1[0],adjust=1)
        self.assertTrue( gt.isomorphism(g1, g2) )


    def test_adjacency2graph(self) :

        adj = {0 : 1, 1 : [2, 3]}
        eTy = {0 : 5, 1 : [9, 14]}
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=1)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [0, 0], 2 : [], 3 : []})

        g   = qt.adjacency2graph(adj, eType=eTy, adjust=0)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [9, 14], 2 : [0], 3 : [0]})

        adj = generate_adjacency()
        g1  = qt.adjacency2graph(adj, adjust=1)
        ans = qt.graph2dict(g1)

        vertices = set()
        for key, value in adj.items() :
            vertices.add(key)
            if isinstance(value, numbers.Integral) :
                vertices.add(value)
            else :
                vertices.update(value)

        for v in vertices :
            if v not in adj :
                adj[v] = []

        m   = min(vertices)
        aj2 = {key-m : [v-m for v in value] for key,value in adj.items()}
        g2  = qt.adjacency2graph(aj2, adjust=1)
        self.assertTrue( gt.isomorphism(g1, g2) )



if __name__ == '__main__':
    unittest.main()
