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
        nEvents = 15000
        
        ans = np.zeros((nEvents,3), bool)

        for k in range(nEvents) :
            nt = len(q._departures) + len(q._queue) + len(q._arrivals) - 2
            nS = len(q._departures) + len(q._queue) - 1
            ans[k,0] = nt == q._nTotal
            ans[k,1] = nS == q.nSystem
            ans[k,2] = len(q._departures) - 1 <= q.nServers
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
        nEvents = 15000
        
        ans = np.zeros((nEvents,3), bool)

        for k in range(nEvents) :
            nt = len(q._departures) + len(q._queue) + len(q._arrivals) - 2
            nS = len(q._departures) + len(q._queue) - 1
            ans[k,0] = nt == q._nTotal
            ans[k,1] = nS == q.nSystem
            ans[k,2] = len(q._departures) - 1 <= q.nServers
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

    @classmethod
    def setUpClass(cls) :
        cls.g   = qt.generate_random_graph(200)
        cls.qn  = qt.QueueNetwork(cls.g)
        cls.qn.max_agents = 2000
        cls.qn.initialize(50)

    def tearDown(self) :
        self.qn.clear()
        self.qn.initialize(50)

    def test_QueueNetwork_sorting(self) :

        nEvents = 1000
        ans = np.zeros(nEvents, bool)
        for k in range(nEvents // 2) :
            net_times   = np.array([q.time for q in self.qn._queues])
            queue_times = [q.time for q in self.qn.edge2queue]
            queue_times.sort()
            while queue_times[-1] == np.infty :
                tmp = queue_times.pop()

            queue_times.sort(reverse=True)

            ans[k] = (queue_times == net_times).all()
            self.qn.simulate(n=1)

        self.qn.simulate(n=10000)

        for k in range(nEvents // 2, nEvents) :
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


    def test_QueueNetwork_max_agents(self) :

        nEvents = 1500
        self.qn.max_agents = 200
        ans = np.zeros(nEvents, bool)

        for k in range(nEvents // 2) :
            ans[k] = np.sum(self.qn.nAgents) <= self.qn.max_agents
            self.qn.simulate(n=1)

        self.qn.simulate(n=20000)

        for k in range(nEvents // 2, nEvents) :
            ans[k] = np.sum(self.qn.nAgents) <= self.qn.max_agents
            self.qn.simulate(n=1)

        self.assertTrue( ans.all() )


    def test_QueueNetwork_accounting(self) :

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


    def test_QueueNetwork_Jackson_routing(self) :

        adj = {0 : 1, 1 : [2, 3]}
        g   = qt.adjacency2graph(adj)
        qn  = qt.QueueNetwork(g)
        mat = qt.generate_transition_matrix(g)
        qn.set_transitions(mat)

        qn.initialize(edge=(0,1))
        qn.collect_data(edge=[(1,2), (1,3)])

        qn.simulate(100000)

        data = qn.data_queues(edge=[(1,2), (1,3)])
        e0, e1 = qn.out_edges[1]

        p0 = np.sum(data[:, 5] == e0) / data.shape[0]
        p1 = np.sum(data[:, 5] == e1) / data.shape[0]

        trans = qn.transitions(False)

        # np.allclose(trans[1], [p0, p1], atol=0.01) )

        self.assertAlmostEqual( trans[1][0], p0, 2)
        self.assertAlmostEqual( trans[1][1], p1, 2)


    def test_QueueNetwork_greedy_routing(self) :

        lam = np.random.randint(1,10)
        rho = np.random.uniform(0.75, 1)
        nSe = np.random.randint(1, 10)
        mu  = lam / (3 * rho * nSe)
        arr = lambda t : t + np.random.exponential(1/lam)
        ser = lambda t : t + np.random.exponential(1/mu)

        adj = {0 : 1, 1 : [2, 3, 4]}
        ety = {0 : 1, 1 : [2, 2, 2]}
        g   = qt.adjacency2graph(adj, eType=ety)

        qcl = {1 : qt.QueueServer, 2 : qt.QueueServer}
        arg = {1 : {'arrival_f' : arr, 'service_f' : lambda t: t,
                    'AgentClass': qt.GreedyAgent}, 
               2 : {'service_f' : ser, 'nServers' : nSe} }

        qn  = qt.QueueNetwork(g, q_classes=qcl, q_args=arg)
        qn.initialize(edge=(0,1))
        qn.max_agents = 5000

        nEvents = 1000
        ans = np.zeros(nEvents, bool)
        e01 = qn.g.edge_index[qn.g.edge(0,1)]
        edg = qn.edge2queue[e01].edge
        c   = 0

        while c < nEvents :
            qn.simulate(n=1)
            if qn.next_event_description() == ('Departure', e01) :
                d0 = qn.edge2queue[e01]._departures[0].desired_destination(qn, edg)
                a1 = np.argmin([qn.edge2queue[e].nQueued() for e in qn.out_edges[1]])
                d1 = qn.out_edges[1][a1]
                ans[c] = d0 == d1
                c += 1

        self.assertTrue( ans.all() )



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
