import numpy as np
import queueing_tool  as qt
import graph_tool.all as gt
import unittest
import numbers
import os


class TestQueueNetwork(unittest.TestCase) :

    @classmethod
    def setUpClass(cls) :
        cls.g  = qt.generate_pagerank_graph(200)
        cls.qn = qt.QueueNetwork(cls.g)
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


    def test_QueueNetwork_sorting2(self) :

        nEvents = 100
        ans = np.zeros(nEvents, bool)
        self.qn.clear()
        self.qn.initialize(1)
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


    def test_QueueNetwork_simulate(self) :

        g  = qt.generate_pagerank_graph(50)
        qn = qt.QueueNetwork(g)
        qn.max_agents = 2000
        qn.initialize(50)
        t0 = np.random.uniform(30, 50)
        qn.max_agents = 2000
        qn.simulate(t=t0)

        self.assertTrue( qn.current_time > t0 )


    def test_QueueNetwork_initialization(self) :

        # Single edge index
        k = np.random.randint(0, self.qn.nE)
        self.qn.clear()
        self.qn.initialize(queues=k)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue(ans == [k])

        # Multiple edge indices
        k = np.unique(np.random.randint(0, self.qn.nE, 5))
        self.qn.clear()
        self.qn.initialize(queues=k)

        ans = np.array([q.edge[2] for q in self.qn.edge2queue if q.active])
        ans.sort()
        self.assertTrue( (ans == k).all() )

        # Single edge as edge
        k = np.random.randint(0, self.qn.nE)
        e = self.g.edge(self.qn.edge2queue[k].edge[0], self.qn.edge2queue[k].edge[1])
        self.qn.clear()
        self.qn.initialize(edge=e)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue(ans == [k])

        # Single edge as tuple
        k  = np.random.randint(0, self.qn.nE)
        e  = self.g.edge(self.qn.edge2queue[k].edge[0], self.qn.edge2queue[k].edge[1])
        ee = (int(e.source()), int(e.target()))
        self.qn.clear()
        self.qn.initialize(edge=ee)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue(ans == [k])

        # Multiple edges as tuples
        k   = np.unique(np.random.randint(0, self.qn.nE, 5))
        es  = [self.g.edge(self.qn.edge2queue[i].edge[0], self.qn.edge2queue[i].edge[1]) for i in k]
        ees = [(int(e.source()), int(e.target())) for e in es]
        self.qn.clear()
        self.qn.initialize(edge=ees)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue( (ans == k).all() )

        # Multple edges as edges
        k   = np.unique(np.random.randint(0, self.qn.nE, 5))
        es  = [self.g.edge(self.qn.edge2queue[i].edge[0], self.qn.edge2queue[i].edge[1]) for i in k]
        self.qn.clear()
        self.qn.initialize(edge=es)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue( (ans == k).all() )

        # Single eType
        k  = np.random.randint(1, 4)
        self.qn.clear()
        self.qn.initialize(eType=k)

        ans = np.array([q.edge[3] == k for q in self.qn.edge2queue if q.active])
        self.assertTrue(ans.all())

        # Multiple eTypes
        k   = np.unique(np.random.randint(1, 4, 3))
        self.qn.clear()
        self.qn.initialize(eType=k)

        ans = np.array([q.edge[3] in k for q in self.qn.edge2queue if q.active])
        self.assertTrue( ans.all() )


    def test_QueueNetwork_add_arrival(self) :

        adj = {0 : 1, 1 : [2, 3]}
        g   = qt.adjacency2graph(adj)
        qn  = qt.QueueNetwork(g)
        mat = qt.generate_transition_matrix(g)
        qn.set_transitions(mat)

        qn.initialize(edge=(0,1))
        qn.start_collecting_data(edge=[(1,2), (1,3)])

        qn.simulate(150000)

        data = qn.data_queues(edge=[(1,2), (1,3)])
        e0, e1 = qn.out_edges[1]

        p0 = np.sum(data[:, 5] == e0, dtype=float) / data.shape[0]
        p1 = np.sum(data[:, 5] == e1, dtype=float) / data.shape[0]

        trans = qn.transitions(False)

        # np.allclose(trans[1], [p0, p1], atol=0.01) )

        self.assertAlmostEqual( trans[1][0], p0, 2)
        self.assertAlmostEqual( trans[1][1], p1, 2)


    def test_QueueNetwork_transitions(self) :

        degree = [len(self.qn.out_edges[k]) for k in range(self.qn.nV)]
        k, deg = np.argmax(degree), max(degree)

        trans  = np.random.uniform(size=deg)
        trans  = trans / sum(trans)

        self.qn.set_transitions({k : trans})
        mat = self.qn.transitions()
        v   = self.qn.g.vertex(k)
        tra = mat[k, [int(e.target()) for e in v.out_edges()]]

        self.assertTrue( (tra == trans).all() )

        tra = self.qn.transitions(return_matrix=False)

        self.assertTrue( (tra[k] == trans).all() )

        mat = qt.generate_transition_matrix(self.g)
        self.qn.set_transitions(mat)
        tra = self.qn.transitions()

        self.assertTrue( np.allclose(tra, mat) )

        mat = qt.generate_transition_matrix(self.g)
        self.qn.set_transitions({k : mat[k]})
        tra = self.qn.transitions()

        self.assertTrue( np.allclose(tra[k], mat[k]) )


    def test_QueueNetwork_data_agents(self) :

        self.qn.clear()
        self.qn.initialize(queues=1)
        self.qn.start_collecting_data()
        self.qn.simulate(n=20000)

        data = self.qn.data_agents()
        dat0 = data[(1,0)]

        a = dat0[:,0]
        b = dat0[dat0[:,1] > 0, 1]
        c = dat0[dat0[:,2] > 0, 2]

        a.sort()
        b.sort()
        c.sort()

        self.assertTrue( (a == dat0[:,0]).all() )
        self.assertTrue( (b == dat0[dat0[:,1] > 0, 1]).all() )
        self.assertTrue( (c == dat0[dat0[:,2] > 0, 2]).all() )
        self.assertTrue( (dat0[1:, 0] == dat0[dat0[:,2] > 0, 2]).all() )


    def test_QueueNetwork_data_queues(self) :

        nV  = 50
        ps  = np.random.uniform(0, 2, size=(nV, 2))

        g, pos = gt.geometric_graph(ps, 1)
        g = qt.set_types_random(g, pTypes={1 : 1})
        q_cls = {1 : qt.QueueServer}

        qn  = qt.QueueNetwork(g, q_classes=q_cls, seed=17)
        k   = np.random.randint(10000, 20000)

        qn.max_agents = 4000
        qn.initialize(queues=range(qn.nE))
        qn.start_collecting_data()
        qn.simulate(n=k)

        data = qn.data_queues()
        self.assertTrue( data.shape == (k, 6) )
        qn.stop_collecting_data()
        qn.clear_data()

        ans = np.array([q.data == {} for q in qn.edge2queue])
        self.assertTrue( ans.all() )


    def test_QueueNetwork_greedy_routing(self) :

        lam = np.random.randint(1,10) + 0.0
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


    def test_QueueNetwork_blocking(self) :

        nV  = 100
        ps  = np.random.uniform(0, 5, size=(nV, 2))

        g, pos = gt.geometric_graph(ps, 1)
        g = qt.set_types_random(g, pTypes={k : 1.0/6 for k in range(1,7)})
        q_cls = {1 : qt.LossQueue, 2 : qt.QueueServer, 3 : qt.InfoQueue, 
                 4 : qt.ResourceQueue, 5 : qt.ResourceQueue, 6 : qt.QueueServer}
        q_arg = {3 : {'net_size' : g.num_edges()}, 4 : {'nServers' : 500}, 6 : {'AgentClass' : qt.GreedyAgent}}

        qn  = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg, seed=17)
        qn.blocking = 'RS'
        self.assertTrue(qn.blocking == 'RS')
        self.assertTrue(qn._blocking == False)

        qn.clear()
        self.assertTrue( qn._initialized == False )


    def test_QueueNetwork_copy(self) :

        nV  = 100
        ps  = np.random.uniform(0, 5, size=(nV, 2))

        g, pos = gt.geometric_graph(ps, 1)
        g = qt.set_types_random(g, pTypes={k : 0.2 for k in range(1,6)})
        q_cls = {1 : qt.LossQueue, 2 : qt.QueueServer, 3 : qt.InfoQueue, 
                 4 : qt.ResourceQueue, 5 : qt.ResourceQueue}
        q_arg = {3 : {'net_size' : g.num_edges()}, 4 : {'nServers' : 500}}

        qn  = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg, seed=17)
        qn.max_agents = np.infty
        qn.initialize(queues=range(g.num_edges()))
        qn.start_collecting_data()

        qn.simulate(n=50000)
        stamp = [(q.nArrivals, q.time) for q in qn.edge2queue]

        qn2 = qn.copy()
        qn.simulate(n=50000)

        self.assertFalse( qn.current_time == qn2.current_time )
        self.assertFalse( qn.time == qn2.time )

        ans = []
        for k, q in enumerate(qn.edge2queue) :
            if stamp[k][1] != q.time :
                ans.append(q.time != qn2.edge2queue[k].time)

        self.assertTrue( np.array(ans).all() )


    def test_QueueNetwork_drawing_animation(self) :

        ct  = np.random.randint(15, 22)
        ans = np.zeros(ct+4, bool)
        self.qn.animate(out='test', n=ct, output_size=(200,200))

        for k in range(ct) :
            ans[k] = os.path.isfile('test%s.png' % (k))
            if ans[k] :
                os.remove('test%s.png' % (k))

        for k in range(4) :
            ans[ct+k] = not os.path.isfile('test%s.png' % (ct+k))

        self.assertTrue( ans.all() )


    def test_QueueNetwork_drawing_animation_time(self) :

        nE0 = self.qn.nEvents
        self.qn.animate(out='test', t=0.01, output_size=(200,200))
        nE1 = self.qn.nEvents
        ct  = nE1 - nE0
        ans = np.zeros(ct+4, bool)

        for k in range(ct) :
            ans[k] = os.path.isfile('test%s.png' % (k))
            if ans[k] :
                os.remove('test%s.png' % (k))

        for k in range(4) :
            ans[ct+k] = not os.path.isfile('test%s.png' % (ct+k))

        self.assertTrue( ans.all() )


    def test_QueueNetwork_show_type_active(self) :

        ans = np.zeros(2, bool)

        self.qn.show_type(2, output='types.png', geometry=(200,200))
        self.qn.show_active(output='active.png', output_size=(200,200))

        ans[0] = os.path.isfile('types.png')
        ans[1] = os.path.isfile('active.png')

        if ans[0] :
            os.remove('types.png')
        if ans[1] :
            os.remove('active.png')

        self.assertTrue( ans.all() )


    def test_QueueNetwork_add_arrival(self) :

        self.qn.simulate(1000)

        ag1 = qt.Agent( (-1,0) )
        ag2 = qt.Agent( (-1,1) )

        t1  = [q.edge[2] for q in self.qn.edge2queue if q.edge[3] == 1]
        t2  = [q.edge[2] for q in self.qn.edge2queue if q.edge[3] == 2]

        q1  = t1[np.random.randint(len(t1))]
        q2  = t2[np.random.randint(len(t2))]
        
        self.qn._add_arrival(q1, ag1)
        self.qn._add_arrival(q2, ag2)

        arrivals = self.qn.edge2queue[q1]._arrivals
        an1 = np.array([ag1.issn == ag.issn for ag in arrivals if isinstance(ag, qt.Agent)])

        arrivals = self.qn.edge2queue[q2]._arrivals
        an2 = np.array([ag2.issn == ag.issn for ag in arrivals if isinstance(ag, qt.Agent)])

        self.assertTrue( an1.any() )
        self.assertTrue( an2.any() )

        net_times   = np.array([q.time for q in self.qn._queues])
        queue_times = [q.time for q in self.qn.edge2queue]
        queue_times.sort()

        while queue_times[-1] == np.infty :
            tmp = queue_times.pop()

        queue_times.sort(reverse=True)

        self.assertTrue( (queue_times == net_times).all() )

        self.qn.simulate(10000)
        net_times   = np.array([q.time for q in self.qn._queues])
        queue_times = [q.time for q in self.qn.edge2queue]
        queue_times.sort()
        while queue_times[-1] == np.infty :
            tmp = queue_times.pop()

        queue_times.sort(reverse=True)
        self.assertTrue( (queue_times == net_times).all() )


if __name__ == '__main__':
    unittest.main()
