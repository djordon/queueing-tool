import functools

import networkx as nx
import numpy as np
import pytest

import queueing_tool as qt


@pytest.fixture(scope="module")
def arrival_rate():
    return float(np.random.randint(1, 10))


@pytest.fixture(scope="module")
def departure_rate():
    return np.random.uniform(0.5, 1)


class TestQueueServers:
    @staticmethod    
    def test_QueueServer_init_errors():
        with pytest.raises(TypeError):
            qt.QueueServer(num_servers=3.0)

        with pytest.raises(ValueError):
            qt.QueueServer(num_servers=0)

    @staticmethod
    def test_QueueServer_set_num_servers():
        nSe = np.random.randint(1, 10)
        q = qt.QueueServer(num_servers=nSe)

        Se1 = q.num_servers
        q.set_num_servers(2 * nSe)

        Se2 = q.num_servers
        q.set_num_servers(np.infty)

        assert Se1 == nSe
        assert Se2 == 2 * nSe
        assert q.num_servers is np.inf

    @staticmethod
    def test_QueueServer_set_num_servers_errors():
        q = qt.QueueServer(num_servers=3)

        with pytest.raises(TypeError):
            q.set_num_servers(3.0)

        with pytest.raises(ValueError):
            q.set_num_servers(0)

    @staticmethod
    def test_QueueServer_set_inactive():
        q = qt.QueueServer()
        q.set_active()

        was_active = q.active
        q.set_inactive()

        assert was_active
        assert not q.active

    @staticmethod
    def test_QueueServer_copy():
        q1 = qt.QueueServer(seed=15)
        q1.set_active()
        q1.simulate(t=100)

        q2 = q1.copy()
        t = q1.time
        q2.simulate(t=20)

        assert t < q2.time

    @staticmethod
    def test_QueueServer_active_cap():
        def r(t):
            return 2 + np.sin(t)

        arr = functools.partial(qt.poisson_random_measure, rate=r, rate_max=3)

        q = qt.QueueServer(active_cap=1000, arrival_f=arr, seed=12)
        q.set_active()
        q.simulate(n=3000)

        assert q.num_departures == 1000
        assert q.num_arrivals == [1000, 1000]

    @staticmethod
    def test_QueueServer_accounting(arrival_rate, departure_rate):

        nSe = np.random.randint(1, 10)
        mu = arrival_rate / (departure_rate * nSe)

        def arr(t):
            return t + np.random.exponential(1 / arrival_rate)

        def ser(t):
            return t + np.random.exponential(1 / mu)

        q = qt.QueueServer(num_servers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        num_events = 15000

        ans = np.zeros((num_events, 3), bool)

        for k in range(num_events):
            nt = len(q._departures) + len(q.queue) + len(q._arrivals) - 2
            nS = len(q._departures) + len(q.queue) - 1
            ans[k, 0] = nt == q._num_total
            ans[k, 1] = nS == q.num_system
            ans[k, 2] = len(q._departures) - 1 <= q.num_servers
            q.simulate(n=1)

        assert ans.all()

    @staticmethod
    def test_QueueServer_deactivate():
        q = qt.QueueServer(num_servers=3, deactive_t=10)
        q.set_active()

        assert q.active

        q.simulate(t=10)
        assert not q.active

    @staticmethod
    def test_QueueServer_simulation(arrival_rate, departure_rate):
        nSe = np.random.randint(1, 10)
        mu = arrival_rate / (departure_rate * nSe)

        def arr(t):
            return t + np.random.exponential(1 / arrival_rate)

        def ser(t):
            return t + np.random.exponential(1 / mu)

        q = qt.QueueServer(num_servers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        num_events = 5000

        ans = np.zeros(4, bool)

        k = np.random.randint(num_events * 0.75, num_events * 1.25)
        nA0 = q._oArrivals
        nD0 = q.num_departures
        q.simulate(n=k)
        ans[0] = q.num_departures + q._oArrivals - nA0 - nD0 == k

        k = np.random.randint(num_events * 0.75, num_events * 1.25)
        nA0 = q._oArrivals
        q.simulate(nA=k)
        ans[1] = q._oArrivals - nA0 == k

        k = np.random.randint(num_events * 0.75, num_events * 1.25)
        nD0 = q.num_departures
        q.simulate(nD=k)
        ans[2] = q.num_departures - nD0 == k

        t = 100 * np.random.uniform(0.5, 1)
        t0 = q.current_time
        q.simulate(t=t)
        ans[3] = q.current_time - t0 >= t

        assert ans.all()

    @staticmethod
    def test_LossQueue_accounting(arrival_rate, departure_rate):
        nSe = np.random.randint(1, 10)
        mu = arrival_rate / (departure_rate * nSe)

        def arr(t):
            return t + np.random.exponential(1 / arrival_rate)

        def ser(t):
            return t + np.random.exponential(1 / mu)

        q = qt.LossQueue(num_servers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        num_events = 15000

        ans = np.zeros((num_events, 3), bool)

        for k in range(num_events):
            nt = len(q._departures) + len(q.queue) + len(q._arrivals) - 2
            nS = len(q._departures) + len(q.queue) - 1
            ans[k, 0] = nt == q._num_total
            ans[k, 1] = nS == q.num_system
            ans[k, 2] = len(q._departures) - 1 <= q.num_servers
            q.simulate(n=1)

        assert ans.all()

    @staticmethod
    def test_LossQueue_blocking(arrival_rate, departure_rate):
        nSe = np.random.randint(1, 10)
        mu = arrival_rate / (departure_rate * nSe)
        k = np.random.randint(5, 15)
        scl = 1 / (mu * k)

        def arr(t):
            return t + np.random.exponential(1 / arrival_rate)

        def ser(t):
            return t + np.random.gamma(k, scl)

        q = qt.LossQueue(num_servers=nSe, arrival_f=arr, service_f=ser)
        q.set_active()
        nE = 500
        c = 0
        ans = np.zeros(nE, bool)

        while c < nE:
            if q.next_event_description() == 1 and q.at_capacity():
                nB0 = q.num_blocked
                q.simulate(n=1)
                ans[c] = nB0 + 1 == q.num_blocked
                c += 1
            else:
                q.simulate(n=1)

        assert ans.all()

    @staticmethod
    def test_NullQueue_data_collection():
        adj = {
            0: {1: {'edge_type': 1}},
            1: {2: {'edge_type': 2},
                3: {'edge_type': 2},
                4: {'edge_type': 2}}
        }
        g = qt.adjacency2graph(adj)

        qcl = {1: qt.QueueServer, 2: qt.NullQueue}

        qn = qt.QueueNetwork(g, q_classes=qcl)
        qn.initialize(edges=(0, 1))
        qn.start_collecting_data(edge_type=2)
        qn.max_agents = 5000
        qn.simulate(n=10000)
        data = qn.get_queue_data()

        # Data collected by NullQueues do not have departure and
        # service start times in the data

        assert not data[:, (1, 2)].any()

    @staticmethod
    def test_ResourceQueue_network():
        g = nx.random_geometric_graph(100, 0.2).to_directed()
        q_cls = {1: qt.ResourceQueue, 2: qt.ResourceQueue}
        q_arg = {1: {'num_servers': 50}, 2: {'num_servers': 500}}

        qn = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg)
        qn.max_agents = 400000
        qn.initialize(queues=range(qn.g.number_of_edges()))
        qn.simulate(n=50000)

        nServ = {1: 50, 2: 500}
        ans = np.array([q.num_servers != nServ[q.edge[3]] for q in qn.edge2queue])
        assert ans.any()

    @staticmethod
    def test_ResourceQueue_network_data_collection():
        g = qt.generate_random_graph(100)
        q_cls = {1: qt.ResourceQueue, 2: qt.ResourceQueue}
        q_arg = {1: {'num_servers': 500},
                 2: {'num_servers': 500,
                     'AgentFactory': qt.Agent}}

        qn = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg)
        qn.max_agents = 40000
        qn.initialize(queues=range(qn.g.number_of_edges()))
        qn.start_collecting_data()
        qn.simulate(n=5000)

        data = qn.get_queue_data()
        assert len(data) > 0

    @staticmethod
    def test_ResourceQueue_network_current_color():
        q = qt.ResourceQueue(num_servers=50)
        ans = q._current_color(0)
        col = q.colors['vertex_fill_color']
        col = [i * (0.9 - 1. / 6) / 0.9 for i in col]
        col[3] = 1.0
        assert ans == col

        ans = q._current_color(1)
        col = q.colors['edge_loop_color']
        col = [i * (0.9 - 1. / 6) / 0.9 for i in col]
        col[3] = 0
        assert ans == col

        ans = q._current_color(2)
        col = q.colors['vertex_pen_color']
        assert ans == col

    @staticmethod
    def test_InfoQueue_network():

        g = nx.random_geometric_graph(100, 0.2).to_directed()
        q_cls = {1: qt.InfoQueue}
        q_arg = {1: {'net_size': g.number_of_edges()}}

        qn = qt.QueueNetwork(g, q_classes=q_cls, q_args=q_arg, seed=17)
        qn.max_agents = 40000
        qn.initialize(queues=range(qn.g.number_of_edges()))
        qn.simulate(n=2000)

        # TODO: Finish this test

    @staticmethod
    def test_Agent_compare():

        a0 = qt.Agent()
        a1 = qt.Agent()
        assert a0 == a1

        a1._time = 10
        assert a0 <= a1
        assert a0 < a1

        a0._time = 20
        assert a0 >= a1
        assert a0 > a1

    @staticmethod
    def test_increasing_num_server_fix_64():
        def arr(t):
            return t + 1.0

        def ser(t):
            return t + 15.0

        queue = qt.QueueServer(num_servers=3, arrival_f=arr, service_f=ser)
        queue.set_active()
        queue.simulate(nA=5)

        assert queue.num_system == 5
        # Should be max(queue.num_system - queue.num_servers, 0) == 2
        assert queue.number_queued() == 2

        queue.set_num_servers(8)

        # Should be max(queue.num_system - queue.num_servers, 0) == 0
        assert queue.number_queued() == 0
        assert queue.num_system == 5

        # Decreasing the number of servers doesn't change the number queued
        queue.set_num_servers(3)

        assert queue.number_queued() == 0
        assert queue.num_system == 5

    @staticmethod
    def test_decreasing_num_server_fix_64():
        def arr(t):
            return t + 1.0

        def ser(t):
            return t + 15.0

        queue = qt.QueueServer(num_servers=5, arrival_f=arr, service_f=ser)
        queue.collect_data = True
        queue.set_active()

        # We want 7 agents in the system. After the 
        queue.simulate(nA=6)
        queue.set_inactive()
        queue.next_event()

        assert queue.num_system == 7
        # Should be max(queue.num_system - queue.num_servers, 0) == 2
        assert queue.number_queued() == 2

        queue.set_num_servers(3)
        # Decreasing the number of queues doesn't immediately eject any
        # agents that are currently being serviced.
        assert queue.number_queued() == 2
        assert queue.num_system == 7
        assert queue.num_departures == 0

        # The next events are all departure since the queue is inactive
        assert queue.next_event_description() == 2

        # There are 5 agents being serviced but only 3 servers since we
        # just changed the number of servers. So the next two departures
        # won't result in us drawing agents from the queue to be serviced
        for k in range(1, 3):
            queue.next_event()
            assert queue.number_queued() == 2
            assert queue.num_system == 7 - k
            assert queue.num_departures == k
            assert queue.next_event_description() == 2

        # Now there are 3 servers and 3 agents being serviced. The next
        # departures will result in an agent being drawn from the queue to
        # receive service.
        queue.next_event()
        assert queue.number_queued() == 1
        assert queue.num_system == 4
        assert queue.next_event_description() == 2

        queue.next_event()
        assert queue.number_queued() == 0
        assert queue.num_system == 3
