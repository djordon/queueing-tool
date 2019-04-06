import os
import unittest
try:
    import unittest.mock as mock
except ImportError:
    import mock

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import networkx as nx
import numpy as np

import queueing_tool as qt


TRAVIS_TEST = os.environ.get('TRAVIS_TEST', False)


class TestQueueNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g = qt.generate_pagerank_graph(200)
        cls.qn = qt.QueueNetwork(cls.g)
        cls.qn.g.draw_graph = mock.MagicMock()
        cls.qn.max_agents = 2000
        cls.qn.initialize(50)

    def tearDown(self):
        self.qn.clear()
        self.qn.initialize(50)

    def test_QueueNetwork_accounting(self):

        num_events = 2500
        ans = np.zeros(num_events, bool)
        na = np.zeros(self.qn.nE, int)
        for q in self.qn.edge2queue:
            na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

        for k in range(num_events):
            ans[k] = (self.qn.num_agents == na).all()
            self.qn.simulate(n=1)
            for q in self.qn.edge2queue:
                na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

        self.assertTrue(ans.all())

    def test_QueueNetwork_add_arrival(self):

        adj = {0: [1], 1: [2, 3]}
        g = qt.adjacency2graph(adj)
        qn = qt.QueueNetwork(g)
        mat = qt.generate_transition_matrix(g)
        qn.set_transitions(mat)

        qn.initialize(edges=(0, 1))
        qn.start_collecting_data(edge=[(1, 2), (1, 3)])

        qn.simulate(150000)

        data = qn.get_queue_data(edge=[(1, 2), (1, 3)])
        e0, e1 = qn.out_edges[1]

        p0 = np.sum(data[:, 5] == e0, dtype=float) / data.shape[0]
        p1 = np.sum(data[:, 5] == e1, dtype=float) / data.shape[0]

        trans = qn.transitions(False)

        self.assertAlmostEqual(trans[1][2], p0, 2)
        self.assertAlmostEqual(trans[1][3], p1, 2)

    def test_QueueNetwork_animate(self):
        if not HAS_MATPLOTLIB:
            with mock.patch('queueing_tool.network.queue_network.plt.show'):
                self.qn.animate(frames=5)
        else:
            plt.switch_backend('Agg')
            self.qn.animate(frames=5)

    def test_QueueNetwork_blocking(self):

        g = nx.random_geometric_graph(100, 0.2).to_directed()
        g = qt.set_types_random(g, proportions={k: 1.0 / 6 for k in range(1, 7)})
        q_cls = {
            1: qt.LossQueue,
            2: qt.QueueServer,
            3: qt.InfoQueue,
            4: qt.ResourceQueue,
            5: qt.ResourceQueue,
            6: qt.QueueServer
        }

        q_arg = {
            3: {'net_size': g.number_of_edges()},
            4: {'num_servers': 500},
            6: {'AgentFactory': qt.GreedyAgent}
        }

        qn = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg, seed=17)
        qn.blocking = 'RS'
        self.assertEqual(qn.blocking, 'RS')
        self.assertEqual(qn._blocking, False)

        qn.clear()
        self.assertEqual(qn._initialized, False)

    def test_QueueNetwork_blocking_setter_error(self):
        self.qn.blocking = 'RS'
        with self.assertRaises(TypeError):
            self.qn.blocking = 2

    def test_QueueNetwork_closedness(self):

        num_events = 2500
        ans = np.zeros(num_events, bool)
        na = np.zeros(self.qn.nE, int)
        for q in self.qn.edge2queue:
            na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

        for k in range(num_events):
            ans[k] = np.sum(self.qn.num_agents) >= np.sum(na)
            for q in self.qn.edge2queue:
                na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

            self.qn.simulate(n=1)

        self.assertTrue(ans.all())

    def test_QueueNetwork_copy(self):

        g = nx.random_geometric_graph(100, 0.2).to_directed()
        g = qt.set_types_random(g, proportions={k: 0.2 for k in range(1, 6)})
        q_cls = {
            1: qt.LossQueue,
            2: qt.QueueServer,
            3: qt.InfoQueue,
            4: qt.ResourceQueue,
            5: qt.ResourceQueue
        }

        q_arg = {3: {'net_size': g.number_of_edges()},
                 4: {'num_servers': 500}}

        qn = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg, seed=17)
        qn.max_agents = np.infty
        qn.initialize(queues=range(g.number_of_edges()))

        qn.simulate(n=50000)
        qn2 = qn.copy()

        stamp = [(q.num_arrivals, q.time) for q in qn2.edge2queue]
        qn2.simulate(n=25000)

        self.assertFalse(qn.current_time == qn2.current_time)
        self.assertFalse(qn.time == qn2.time)

        ans = []
        for k, q in enumerate(qn2.edge2queue):
            if stamp[k][1] != q.time:
                ans.append(q.time != qn.edge2queue[k].time)

        self.assertTrue(np.array(ans).all())

    @mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', True)
    def test_QueueNetwork_drawing(self):
        scatter_kwargs = {'c': 'b'}
        kwargs = {'bgcolor': 'green'}
        self.qn.draw(scatter_kwargs=scatter_kwargs, **kwargs)
        self.qn.g.draw_graph.assert_called_with(scatter_kwargs=scatter_kwargs,
                                                line_kwargs=None, **kwargs)

        self.qn.draw(scatter_kwargs=scatter_kwargs)
        bgcolor = self.qn.colors['bgcolor']
        self.qn.g.draw_graph.assert_called_with(scatter_kwargs=scatter_kwargs,
                                                line_kwargs=None, bgcolor=bgcolor)

    @mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', False)
    def test_QueueNetwork_drawing_importerror(self):
        with self.assertRaises(ImportError):
            self.qn.draw()

    def test_QueueNetwork_drawing_animation_error(self):
        self.qn.clear()
        with self.assertRaises(qt.QueueingToolError):
            self.qn.animate()

        self.qn.initialize()
        with mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', False):
            with self.assertRaises(ImportError):
                self.qn.animate()

    def test_QueueNetwork_init_error(self):
        g = qt.generate_pagerank_graph(7)
        with self.assertRaises(TypeError):
            qt.QueueNetwork(g, blocking=2)

    def test_QueueNetwork_get_agent_data(self):

        self.qn.clear()
        self.qn.initialize(queues=1)
        self.qn.start_collecting_data()
        self.qn.simulate(n=20000)

        data = self.qn.get_agent_data()
        dat0 = data[(1, 0)]

        a = dat0[:, 0]
        b = dat0[dat0[:, 1] > 0, 1]
        c = dat0[dat0[:, 2] > 0, 2]

        a.sort()
        b.sort()
        c.sort()

        self.assertTrue((a == dat0[:, 0]).all())
        self.assertTrue((b == dat0[dat0[:, 1] > 0, 1]).all())
        self.assertTrue((c == dat0[dat0[:, 2] > 0, 2]).all())
        self.assertTrue((dat0[1:, 0] == dat0[dat0[:, 2] > 0, 2]).all())

    def test_QueueNetwork_get_queue_data(self):

        g = nx.random_geometric_graph(50, 0.5).to_directed()
        q_cls = {1: qt.QueueServer}

        qn = qt.QueueNetwork(g, q_classes=q_cls, seed=17)
        k = np.random.randint(10000, 20000)

        qn.max_agents = 4000
        qn.initialize(queues=range(qn.nE))
        qn.start_collecting_data()
        qn.simulate(n=k)

        data = qn.get_queue_data()
        self.assertEqual(data.shape, (k, 6))
        qn.stop_collecting_data()
        qn.clear_data()

        ans = np.array([q.data == {} for q in qn.edge2queue])
        self.assertTrue(ans.all())

    def test_QueueNetwork_greedy_routing(self):

        lam = np.random.randint(1, 10) + 0.0
        rho = np.random.uniform(0.75, 1)
        nSe = np.random.randint(1, 10)
        mu = lam / (3 * rho * nSe)

        def arr(t):
            return t + np.random.exponential(1 / lam)

        def ser(t):
            return t + np.random.exponential(1 / mu)

        def ser_id(t):
            return t

        adj = {
            0: {1: {'edge_type': 1}},
            1: {
                2: {'edge_type': 2},
                3: {'edge_type': 2},
                4: {'edge_type': 2}
            }
        }
        g = qt.adjacency2graph(adj)

        qcl = {1: qt.QueueServer, 2: qt.QueueServer}
        arg = {
            1: {
                'arrival_f': arr,
                'service_f': ser_id,
                'AgentFactory': qt.GreedyAgent
            },
            2: {
                'service_f': ser,
                'num_servers': nSe
            }
        }

        qn = qt.QueueNetwork(g, q_classes=qcl, q_args=arg)
        qn.initialize(edges=(0, 1))
        qn.max_agents = 5000

        num_events = 1000
        ans = np.zeros(num_events, bool)
        e01 = qn.g.edge_index[(0, 1)]
        edg = qn.edge2queue[e01].edge
        c = 0

        while c < num_events:
            qn.simulate(n=1)
            if qn.next_event_description() == ('Departure', e01):
                d0 = qn.edge2queue[e01]._departures[0].desired_destination(qn, edg)
                a1 = np.argmin([qn.edge2queue[e].number_queued() for e in qn.out_edges[1]])
                d1 = qn.out_edges[1][a1]
                ans[c] = d0 == d1
                c += 1

        self.assertTrue(ans.all())

    def test_QueueNetwork_initialize_Error(self):
        self.qn.clear()
        with self.assertRaises(ValueError):
            self.qn.initialize(nActive=0)

        with self.assertRaises(TypeError):
            self.qn.initialize(nActive=1.6)

        _get_queues_mock = mock.Mock()
        _get_queues_mock.return_value = []
        mock_location = 'queueing_tool.network.queue_network._get_queues'

        with mock.patch(mock_location, _get_queues_mock):
            with self.assertRaises(qt.QueueingToolError):
                self.qn.initialize(edge_type=1)

    def test_QueueNetwork_initialization(self):
        # Single edge index
        k = np.random.randint(0, self.qn.nE)
        self.qn.clear()
        self.qn.initialize(queues=k)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertEqual(ans, [k])

        # Multiple edge indices
        k = np.unique(np.random.randint(0, self.qn.nE, 5))
        self.qn.clear()
        self.qn.initialize(queues=k)

        ans = np.array([q.edge[2] for q in self.qn.edge2queue if q.active])
        ans.sort()
        self.assertTrue((ans == k).all())

        # Single edge as edge
        k = np.random.randint(0, self.qn.nE)
        e = self.qn.edge2queue[k].edge[:2]
        self.qn.clear()
        self.qn.initialize(edges=e)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertEqual(ans, [k])

        # Single edge as tuple
        k = np.random.randint(0, self.qn.nE)
        e = self.qn.edge2queue[k].edge[:2]
        self.qn.clear()
        self.qn.initialize(edges=e)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertEqual(ans, [k])

        # Multiple edges as tuples
        k = np.unique(np.random.randint(0, self.qn.nE, 5))
        es = [self.qn.edge2queue[i].edge[:2] for i in k]
        self.qn.clear()
        self.qn.initialize(edges=es)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue((ans == k).all())

        # Multple edges as edges
        k = np.unique(np.random.randint(0, self.qn.nE, 5))
        es = [self.qn.edge2queue[i].edge[:2] for i in k]
        self.qn.clear()
        self.qn.initialize(edges=es)

        ans = [q.edge[2] for q in self.qn.edge2queue if q.active]
        self.assertTrue((ans == k).all())

        # Single edge_type
        k = np.random.randint(1, 4)
        self.qn.clear()
        self.qn.initialize(edge_type=k)

        ans = np.array([q.edge[3] == k for q in self.qn.edge2queue if q.active])
        self.assertTrue(ans.all())

        # Multiple edge_types
        k = np.unique(np.random.randint(1, 4, 3))
        self.qn.clear()
        self.qn.initialize(edge_type=k)

        ans = np.array([q.edge[3] in k for q in self.qn.edge2queue if q.active])
        self.assertTrue(ans.all())

        self.qn.clear()
        self.qn.max_agents = 3
        self.qn.initialize(nActive=self.qn.num_edges)
        ans = np.array([q.active for q in self.qn.edge2queue])
        self.assertEqual(ans.sum(), 3)

    def test_QueueNetwork_max_agents(self):

        num_events = 1500
        self.qn.max_agents = 200
        ans = np.zeros(num_events, bool)

        for k in range(num_events // 2):
            ans[k] = np.sum(self.qn.num_agents) <= self.qn.max_agents
            self.qn.simulate(n=1)

        self.qn.simulate(n=20000)

        for k in range(num_events // 2, num_events):
            ans[k] = np.sum(self.qn.num_agents) <= self.qn.max_agents
            self.qn.simulate(n=1)

        self.assertTrue(ans.all())

    def test_QueueNetwork_properties(self):
        self.qn.clear()
        self.assertEqual(self.qn.time, np.infty)
        self.assertEqual(self.qn.num_edges, self.qn.nE)
        self.assertEqual(self.qn.num_vertices, self.qn.nV)
        self.assertEqual(self.qn.num_nodes, self.qn.nV)

    def test_QueueNetwork_set_transitions_Error(self):
        with self.assertRaises(ValueError):
            self.qn.set_transitions({-1: {0: 0.75, 1: 0.25}})

        with self.assertRaises(ValueError):
            self.qn.set_transitions({self.qn.nV: {0: 0.75, 1: 0.25}})

        with self.assertRaises(ValueError):
            self.qn.set_transitions({0: {0: 0.75, 1: -0.25}})

        with self.assertRaises(ValueError):
            self.qn.set_transitions({0: {0: 1.25, 1: -0.25}})

        mat = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            self.qn.set_transitions(mat)

        mat = np.zeros((self.qn.nV, self.qn.nV))
        with self.assertRaises(ValueError):
            self.qn.set_transitions(mat)

        mat[0, 0] = -1
        mat[0, 1] = 2
        with self.assertRaises(ValueError):
            self.qn.set_transitions(mat)

        mat = 1
        with self.assertRaises(TypeError):
            self.qn.set_transitions(mat)

    def test_QueueNetwork_simulate(self):

        g = qt.generate_pagerank_graph(50)
        qn = qt.QueueNetwork(g)
        qn.max_agents = 2000
        qn.initialize(50)
        t0 = np.random.uniform(30, 50)
        qn.max_agents = 2000
        qn.simulate(t=t0)

        self.assertGreater(qn.current_time, t0)

    def test_QueueNetwork_simulate_error(self):
        self.qn.clear()
        with self.assertRaises(qt.QueueingToolError):
            self.qn.simulate()

    def test_QueueNetwork_simulate_slow(self):
        e = self.qn._fancy_heap.array_edges[0]
        edge = self.qn.edge2queue[e].edge

        if edge[0] == edge[1]:
            for q in self.qn.edge2queue:
                if q.edge[0] != q.edge[1]:
                    break
            self.qn._simulate_next_event(slow=True)
        else:
            for q in self.qn.edge2queue:
                if q.edge[0] == q.edge[1]:
                    break
            self.qn._simulate_next_event(slow=True)

        self.qn.clear()
        self.qn.initialize(queues=[q.edge[2]])
        e = self.qn._fancy_heap.array_edges[0]
        edge = self.qn.edge2queue[e].edge

        loop = edge[0] == edge[1]
        self.qn._simulate_next_event(slow=True)

        while True:
            e = self.qn._fancy_heap.array_edges[0]
            edge = self.qn.edge2queue[e].edge

            if (edge[0] != edge[1]) == loop:
                self.qn._simulate_next_event(slow=True)
                break
            else:
                self.qn._simulate_next_event(slow=False)

    @mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', True)
    def test_QueueNetwork_show_type(self):
        args = {'c': 'b', 'bgcolor': 'green'}
        self.qn.show_type(edge_type=2, **args)
        self.qn.g.draw_graph.assert_called_with(scatter_kwargs=None,
                                                line_kwargs=None, **args)

    @mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', True)
    def test_QueueNetwork_show_active(self):
        args = {
            'fname': 'types.png',
            'figsize': (3, 3),
            'bgcolor': 'green'
        }
        self.qn.show_active(**args)
        self.qn.g.draw_graph.assert_called_with(scatter_kwargs=None,
                                                line_kwargs=None, **args)

    def test_QueueNetwork_sorting(self):

        num_events = 2000
        ans = np.zeros(num_events, bool)
        for k in range(num_events // 2):
            queue_times = [q.time for q in self.qn.edge2queue]
            queue_times.sort()
            tmp = queue_times[0]
            self.qn.simulate(n=1)
            ans[k] = (tmp == self.qn._qkey[0])

        self.qn.simulate(n=10000)

        for k in range(num_events // 2, num_events):
            queue_times = [q.time for q in self.qn.edge2queue]
            queue_times.sort()
            tmp = queue_times[0]
            self.qn.simulate(n=1)
            ans[k] = (tmp == self.qn._qkey[0])

        self.assertTrue(ans.all())

    def test_QueueNetwork_transitions(self):

        degree = [len(self.qn.out_edges[k]) for k in range(self.qn.nV)]
        v, deg = np.argmax(degree), max(degree)
        out_edges = sorted(self.qn.g.out_edges(v))

        trans = np.random.uniform(size=deg)
        trans = trans / sum(trans)
        probs = {v: {e[1]: p for e, p in zip(out_edges, trans)}}

        self.qn.set_transitions(probs)
        mat = self.qn.transitions()
        tra = mat[v, [e[1] for e in out_edges]]

        self.assertTrue((tra == trans).all())

        tra = self.qn.transitions(return_matrix=False)
        tra = np.array([tra[v][e[1]] for e in out_edges])
        self.assertTrue((tra == trans).all())

        mat = qt.generate_transition_matrix(self.g)
        self.qn.set_transitions(mat)
        tra = self.qn.transitions()

        self.assertTrue(np.allclose(tra, mat))

        mat = qt.generate_transition_matrix(self.g)
        self.qn.set_transitions({v: {e[1]: mat[e] for e in out_edges}})
        tra = self.qn.transitions()

        self.assertTrue(np.allclose(tra[v], mat[v]))
