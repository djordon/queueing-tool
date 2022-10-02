import os
import unittest.mock as mock

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import networkx as nx
import numpy as np
import pytest

import queueing_tool as qt


CI_TEST = os.environ.get('CI_TEST', False)

@pytest.fixture(scope="module", name="queue_network")
def fixture_queue_network():
    g = qt.generate_pagerank_graph(200)
    qn = qt.QueueNetwork(g)
    qn.g.draw_graph = mock.MagicMock()
    qn.max_agents = 2000
    return qn


@pytest.fixture(name="qn")
def fixture_qn(queue_network):
    queue_network.initialize(50)

    yield queue_network

    queue_network.clear()
    queue_network.initialize(50)


class TestQueueNetwork:
    @staticmethod
    def test_QueueNetwork_accounting(qn):
        num_events = 2500
        ans = np.zeros(num_events, bool)
        na = np.zeros(qn.nE, int)
        for q in qn.edge2queue:
            na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

        for k in range(num_events):
            ans[k] = (qn.num_agents == na).all()
            qn.simulate(n=1)
            for q in qn.edge2queue:
                na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

        assert ans.all()

    @staticmethod
    def test_QueueNetwork_add_arrival(qn):
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

        np.testing.assert_allclose(trans[1][2], p0, atol=10**(-2))
        np.testing.assert_allclose(trans[1][3], p1, atol=10**(-2))

    @staticmethod
    @pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
    @pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything")
    def test_QueueNetwork_animate(qn):
        if HAS_MATPLOTLIB:
            plt.switch_backend('Agg')
            qn.animate(frames=5)
        else:
            with mock.patch('queueing_tool.network.queue_network.plt.show'):
                qn.animate(frames=5)

    @staticmethod
    def test_QueueNetwork_blocking(qn):
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
        assert qn.blocking == 'RS'
        assert qn._blocking == False

        qn.clear()
        assert qn._initialized == False

    @staticmethod
    def test_QueueNetwork_blocking_setter_error(qn):
        qn.blocking = 'RS'
        with pytest.raises(TypeError):
            qn.blocking = 2

    @staticmethod
    def test_QueueNetwork_closedness(qn):
        num_events = 2500
        ans = np.zeros(num_events, bool)
        na = np.zeros(qn.nE, int)
        for q in qn.edge2queue:
            na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

        for k in range(num_events):
            ans[k] = np.sum(qn.num_agents) >= np.sum(na)
            for q in qn.edge2queue:
                na[q.edge[2]] = len(q._arrivals) + len(q._departures) + len(q.queue) - 2

            qn.simulate(n=1)

        assert ans.all()

    @staticmethod
    def test_QueueNetwork_copy(qn):
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

        assert qn.current_time != qn2.current_time
        assert qn.time != qn2.time

        ans = []
        for k, q in enumerate(qn2.edge2queue):
            if stamp[k][1] != q.time:
                ans.append(q.time != qn.edge2queue[k].time)

        assert np.array(ans).all()

    @staticmethod
    def test_QueueNetwork_drawing(qn):
        with mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', True):
            scatter_kwargs = {'c': 'b'}
            kwargs = {'bgcolor': 'green'}
            qn.draw(scatter_kwargs=scatter_kwargs, **kwargs)
            qn.g.draw_graph.assert_called_with(scatter_kwargs=scatter_kwargs,
                                                    line_kwargs=None, **kwargs)

            qn.draw(scatter_kwargs=scatter_kwargs)
            bgcolor = qn.colors['bgcolor']
            qn.g.draw_graph.assert_called_with(scatter_kwargs=scatter_kwargs,
                                                    line_kwargs=None, bgcolor=bgcolor)

    @staticmethod
    def test_QueueNetwork_drawing_importerror(qn):
        with mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', False):
            with pytest.raises(ImportError):
                qn.draw()

    @staticmethod
    def test_QueueNetwork_drawing_animation_error(qn):
        qn.clear()
        with pytest.raises(qt.QueueingToolError):
            qn.animate()

        qn.initialize()
        with mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', False):
            with pytest.raises(ImportError):
                qn.animate()

    @staticmethod
    def test_QueueNetwork_init_error():
        g = qt.generate_pagerank_graph(7)
        with pytest.raises(TypeError):
            qt.QueueNetwork(g, blocking=2)

    @staticmethod
    def test_QueueNetwork_get_agent_data(qn):
        qn.clear()
        qn.initialize(queues=1)
        qn.start_collecting_data()
        qn.simulate(n=20000)

        data = qn.get_agent_data()
        dat0 = data[(1, 0)]

        a = dat0[:, 0]
        b = dat0[dat0[:, 1] > 0, 1]
        c = dat0[dat0[:, 2] > 0, 2]

        a.sort()
        b.sort()
        c.sort()

        assert (a == dat0[:, 0]).all()
        assert (b == dat0[dat0[:, 1] > 0, 1]).all()
        assert (c == dat0[dat0[:, 2] > 0, 2]).all()
        assert (dat0[1:, 0] == dat0[dat0[:, 2] > 0, 2]).all()

    @staticmethod
    def test_QueueNetwork_get_queue_data(qn):
        g = nx.random_geometric_graph(50, 0.5).to_directed()
        q_cls = {1: qt.QueueServer}

        qn = qt.QueueNetwork(g, q_classes=q_cls, seed=17)
        k = np.random.randint(10000, 20000)

        qn.max_agents = 4000
        qn.initialize(queues=range(qn.nE))
        qn.start_collecting_data()
        qn.simulate(n=k)

        data = qn.get_queue_data()
        assert data.shape == (k, 6)
        qn.stop_collecting_data()
        qn.clear_data()

        ans = np.array([q.data == {} for q in qn.edge2queue])
        assert ans.all()

    @staticmethod
    def test_QueueNetwork_greedy_routing(qn):
        lam = float(np.random.randint(1, 10))
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

        assert ans.all()

    @staticmethod
    def test_QueueNetwork_initialize_Error(qn):
        qn.clear()
        with pytest.raises(ValueError):
            qn.initialize(nActive=0)

        with pytest.raises(TypeError):
            qn.initialize(nActive=1.6)

        _get_queues_mock = mock.Mock()
        _get_queues_mock.return_value = []
        mock_location = 'queueing_tool.network.queue_network._get_queues'

        with mock.patch(mock_location, _get_queues_mock):
            with pytest.raises(qt.QueueingToolError):
                qn.initialize(edge_type=1)

    @staticmethod
    def test_QueueNetwork_initialization(qn):
        # Single edge index
        k = np.random.randint(0, qn.nE)
        qn.clear()
        qn.initialize(queues=k)

        ans = [q.edge[2] for q in qn.edge2queue if q.active]
        assert ans == [k]

        # Multiple edge indices
        k = np.unique(np.random.randint(0, qn.nE, 5))
        qn.clear()
        qn.initialize(queues=k)

        ans = np.array([q.edge[2] for q in qn.edge2queue if q.active])
        ans.sort()
        assert (ans == k).all()

        # Single edge as edge
        k = np.random.randint(0, qn.nE)
        e = qn.edge2queue[k].edge[:2]
        qn.clear()
        qn.initialize(edges=e)

        ans = [q.edge[2] for q in qn.edge2queue if q.active]
        assert ans == [k]

        # Single edge as tuple
        k = np.random.randint(0, qn.nE)
        e = qn.edge2queue[k].edge[:2]
        qn.clear()
        qn.initialize(edges=e)

        ans = [q.edge[2] for q in qn.edge2queue if q.active]
        assert ans == [k]

        # Multiple edges as tuples
        k = np.unique(np.random.randint(0, qn.nE, 5))
        es = [qn.edge2queue[i].edge[:2] for i in k]
        qn.clear()
        qn.initialize(edges=es)

        ans = [q.edge[2] for q in qn.edge2queue if q.active]
        assert (ans == k).all()

        # Multple edges as edges
        k = np.unique(np.random.randint(0, qn.nE, 5))
        es = [qn.edge2queue[i].edge[:2] for i in k]
        qn.clear()
        qn.initialize(edges=es)

        ans = [q.edge[2] for q in qn.edge2queue if q.active]
        assert (ans == k).all()

        # Single edge_type
        k = np.random.randint(1, 4)
        qn.clear()
        qn.initialize(edge_type=k)

        ans = np.array([q.edge[3] == k for q in qn.edge2queue if q.active])
        assert ans.all()

        # Multiple edge_types
        k = np.unique(np.random.randint(1, 4, 3))
        qn.clear()
        qn.initialize(edge_type=k)

        ans = np.array([q.edge[3] in k for q in qn.edge2queue if q.active])
        assert ans.all()

        qn.clear()
        qn.max_agents = 3
        qn.initialize(nActive=qn.num_edges)
        ans = np.array([q.active for q in qn.edge2queue])
        assert ans.sum() == 3

    @staticmethod
    def test_QueueNetwork_max_agents(qn):
        num_events = 1500
        qn.max_agents = 200
        ans = np.zeros(num_events, bool)

        for k in range(num_events // 2):
            ans[k] = np.sum(qn.num_agents) <= qn.max_agents
            qn.simulate(n=1)

        qn.simulate(n=20000)

        for k in range(num_events // 2, num_events):
            ans[k] = np.sum(qn.num_agents) <= qn.max_agents
            qn.simulate(n=1)

        assert ans.all()

    @staticmethod
    def test_QueueNetwork_properties(qn):
        qn.clear()
        assert qn.time == np.infty
        assert qn.num_edges == qn.nE
        assert qn.num_vertices == qn.nV
        assert qn.num_nodes == qn.nV

    @staticmethod
    def test_QueueNetwork_set_transitions_Error(qn):
        with pytest.raises(ValueError):
            qn.set_transitions({-1: {0: 0.75, 1: 0.25}})

        with pytest.raises(ValueError):
            qn.set_transitions({qn.nV: {0: 0.75, 1: 0.25}})

        with pytest.raises(ValueError):
            qn.set_transitions({0: {0: 0.75, 1: -0.25}})

        with pytest.raises(ValueError):
            qn.set_transitions({0: {0: 1.25, 1: -0.25}})

        mat = np.zeros((2, 2))
        with pytest.raises(ValueError):
            qn.set_transitions(mat)

        mat = np.zeros((qn.nV, qn.nV))
        with pytest.raises(ValueError):
            qn.set_transitions(mat)

        mat[0, 0] = -1
        mat[0, 1] = 2
        with pytest.raises(ValueError):
            qn.set_transitions(mat)

        mat = 1
        with pytest.raises(TypeError):
            qn.set_transitions(mat)

    @staticmethod
    def test_QueueNetwork_simulate(qn):
        g = qt.generate_pagerank_graph(50)
        qn = qt.QueueNetwork(g)
        qn.max_agents = 2000
        qn.initialize(50)
        t0 = np.random.uniform(30, 50)
        qn.max_agents = 2000
        qn.simulate(t=t0)

        assert qn.current_time > t0

    @staticmethod
    def test_QueueNetwork_simulate_error(qn):
        qn.clear()
        with pytest.raises(qt.QueueingToolError):
            qn.simulate()

    @staticmethod
    def test_QueueNetwork_simulate_slow(qn):
        e = qn._fancy_heap.array_edges[0]
        edge = qn.edge2queue[e].edge

        if edge[0] == edge[1]:
            for q in qn.edge2queue:
                if q.edge[0] != q.edge[1]:
                    break
            qn._simulate_next_event(slow=True)
        else:
            for q in qn.edge2queue:
                if q.edge[0] == q.edge[1]:
                    break
            qn._simulate_next_event(slow=True)

        qn.clear()
        qn.initialize(queues=[q.edge[2]])
        e = qn._fancy_heap.array_edges[0]
        edge = qn.edge2queue[e].edge

        loop = edge[0] == edge[1]
        qn._simulate_next_event(slow=True)

        while True:
            e = qn._fancy_heap.array_edges[0]
            edge = qn.edge2queue[e].edge

            if (edge[0] != edge[1]) == loop:
                qn._simulate_next_event(slow=True)
                break
            else:
                qn._simulate_next_event(slow=False)

    @staticmethod
    def test_QueueNetwork_show_type(qn):
        with mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', True):
            args = {'c': 'b', 'bgcolor': 'green'}
            qn.show_type(edge_type=2, **args)
            qn.g.draw_graph.assert_called_with(
                scatter_kwargs=None,
                line_kwargs=None,
                **args
            )

    @staticmethod
    def test_QueueNetwork_show_active(qn):
        with mock.patch('queueing_tool.network.queue_network.HAS_MATPLOTLIB', True):
            args = {
                'fname': 'types.png',
                'figsize': (3, 3),
                'bgcolor': 'green'
            }
            qn.show_active(**args)
            qn.g.draw_graph.assert_called_with(
                scatter_kwargs=None,
                line_kwargs=None,
                **args
            )

    @staticmethod
    def test_QueueNetwork_sorting(qn):
        num_events = 2000
        ans = np.zeros(num_events, bool)
        for k in range(num_events // 2):
            queue_times = [q.time for q in qn.edge2queue]
            queue_times.sort()
            tmp = queue_times[0]
            qn.simulate(n=1)
            ans[k] = (tmp == qn._qkey[0])

        qn.simulate(n=10000)

        for k in range(num_events // 2, num_events):
            queue_times = [q.time for q in qn.edge2queue]
            queue_times.sort()
            tmp = queue_times[0]
            qn.simulate(n=1)
            ans[k] = (tmp == qn._qkey[0])

        assert ans.all()

    @staticmethod
    def test_QueueNetwork_transitions(qn):
        degree = [len(qn.out_edges[k]) for k in range(qn.nV)]
        v, deg = np.argmax(degree), max(degree)
        out_edges = sorted(qn.g.out_edges(v))

        trans = np.random.uniform(size=deg)
        trans = trans / sum(trans)
        probs = {v: {e[1]: p for e, p in zip(out_edges, trans)}}

        qn.set_transitions(probs)
        mat = qn.transitions()
        tra = mat[v, [e[1] for e in out_edges]]

        assert (tra == trans).all()

        tra = qn.transitions(return_matrix=False)
        tra = np.array([tra[v][e[1]] for e in out_edges])
        assert (tra == trans).all()

        mat = qt.generate_transition_matrix(qn.g)
        qn.set_transitions(mat)
        tra = qn.transitions()

        assert np.allclose(tra, mat)

        mat = qt.generate_transition_matrix(qn.g)
        qn.set_transitions({v: {e[1]: mat[e] for e in out_edges}})
        tra = qn.transitions()

        assert np.allclose(tra[v], mat[v])
