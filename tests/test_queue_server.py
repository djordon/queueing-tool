import os
import unittest

import networkx as nx
import numpy as np

import queueing_tool as qt


class TestQueueServers(unittest.TestCase) :

    def setUp(self) :
        self.lam = np.random.randint(1,10) + 0.0
        self.rho = np.random.uniform(0.5, 1)


    def test_QueueServer_set_nServers(self):

        nSe = np.random.randint(1, 10)
        q   = qt.QueueServer(nServers=nSe)

        Se1 = q.nServers
        q.set_nServers(2*nSe)

        Se2 = q.nServers
        q.set_nServers(np.infty)

        self.assertTrue( Se1 == nSe )
        self.assertTrue( Se2 == 2*nSe )
        self.assertTrue( q.nServers is np.inf )


    def test_QueueServer_set_inactive(self) :
        
        q   = qt.QueueServer()
        q.set_active()

        a   = q.active
        q.set_inactive()

        self.assertTrue( a )
        self.assertTrue( not q.active )


    def test_QueueServer_copy(self) :
        
        q1  = qt.QueueServer(seed=15)
        q1.set_active()
        q1.simulate(t=100)

        q2  = q1.copy()
        t   = q1.time
        q2.simulate(t=20)

        self.assertTrue( t < q2.time )


    def test_QueueServer_active_cap(self) :

        r   = lambda t: 2 + np.sin(t)
        arr = lambda t: qt.poisson_random_measure(r, 3, t)
        q   = qt.QueueServer(active_cap=1000, arrival_f=arr, seed=12)
        q.set_active()
        q.simulate(n=3000)

        self.assertTrue( q.nDepartures == 1000 )
        self.assertTrue( q.nArrivals == [1000, 1000] )


    def test_QueueServer_accounting(self) :

        nSe = np.random.randint(1, 10)
        mu  = self.lam / (self.rho * nSe)
        arr = lambda t : t + np.random.exponential(1 / self.lam)
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
        arr = lambda t : t + np.random.exponential(1 / self.lam)
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
        arr = lambda t : t + np.random.exponential(1 / self.lam)
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

        arr = lambda t : t + np.random.exponential(1 / self.lam)
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


    def test_ResourceQueue_network(self) :

        g = nx.random_geometric_graph(100, 0.2).to_directed()
        q_cls = {1 : qt.ResourceQueue, 2 : qt.ResourceQueue}
        q_arg = {1 : {'nServers' : 50}, 2 : {'nServers' : 500}}

        qn  = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg)
        qn.max_agents = 400000
        qn.initialize(queues=range(qn.g.num_edges()))
        qn.simulate(n=50000)

        nServ = {1 : 50, 2 : 500}
        ans   = np.array([q.nServers != nServ[q.edge[3]] for q in qn.edge2queue])
        self.assertTrue( ans.any() )


    def test_ResourceQueue_animation(self) :

        g = nx.random_geometric_graph(100, 0.2).to_directed()
        e = g.add_edge(1, 1)

        q_cls = {1 : qt.ResourceQueue, 2 : qt.ResourceQueue}
        q_arg = {1 : {'nServers' : 50}, 2 : {'nServers' : 500}}

        qn  = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg)
        qn.max_agents = 400000
        qn.initialize(queues=range(qn.g.num_edges()))

        ct  = np.random.randint(25, 52)
        ans = np.zeros(ct+4, bool)
        qn.animate(out='test', n=ct, output_size=(200,200))

        for k in range(ct+1) :
            ans[k] = os.path.isfile('test%s.png' % (k))
            if ans[k] :
                os.remove('test%s.png' % (k))

        for k in range(4) :
            ans[ct+k] = not os.path.isfile('test%s.png' % (ct+k))

        self.assertTrue( ans.all() )


    def test_InfoQueue_network(self) :

        g = nx.random_geometric_graph(100, 0.2).to_directed()
        q_cls = {1 : qt.InfoQueue}
        q_arg = {1 : {'net_size' : g.number_of_edges()} }

        qn  = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg, seed=17)
        qn.max_agents = 40000
        qn.initialize(queues=range(qn.g.num_edges()))
        qn.simulate(n=2000)

        # Finish this
        self.assertTrue( True )


    def test_Agent_compare(self) :

        a0 = qt.Agent()
        a1 = qt.Agent()
        self.assertTrue( a0 == a1 )

        a1._time = 10
        self.assertTrue( a0 <= a1 )
        self.assertTrue( a0 <  a1 )

        a0._time = 20
        self.assertTrue( a0 >= a1 )
        self.assertTrue( a0 >  a1 )



if __name__ == '__main__':
    unittest.main()
