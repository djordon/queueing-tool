import collections
import copy
from heapq import heappush, heappop

import numpy as np
from numpy import infty

from queueing_tool.network.queue_network import QueueNetwork
from queueing_tool.queues.agents import Agent
from queueing_tool.queues.choice import _choice
from queueing_tool.queues.queue_servers import QueueServer


class MultiClassQueueNetwork(QueueNetwork):

    def __init__(self, *args, **kwargs):
        super(MultiClassQueueNetwork, self).__init__(*args, **kwargs)

        def default_factory():
            return copy.deepcopy(self._route_probs)

        self._routing_transitions = collections.defaultdict(default_factory)

    def set_transitions(self, mat, category=None):
        for key, value in mat.items():
            probs = list(value.values())

            if key not in self.g.node:
                msg = "One of the keys don't correspond to a vertex."
                raise ValueError(msg)
            elif len(self.out_edges[key]) > 0 and not np.isclose(sum(probs), 1):
                msg = "Sum of transition probabilities at a vertex was not 1."
                raise ValueError(msg)
            elif (np.array(probs) < 0).any():
                msg = "Some transition probabilities were negative."
                raise ValueError(msg)

            for k, e in enumerate(self.g.out_edges(key)):
                self._routing_transitions[category][key][k] = value.get(e[1], 0)

    def set_categorical_transitions(self, adjacency_list):
        for category, mat in adjacency_list.items():
            self.set_transitions(mat, category)

    def transitions(self):
        mat = {
            category: {
                k: {e[1]: p for e, p in zip(self.g.out_edges(k), value)}
                for k, value in enumerate(routing_probs)
            }
            for category, routing_probs in self._routing_transitions.items()
        }
        return mat

    def routing_transition(self, destination, category=None):
        if category not in self._routing_transitions:
            category = None

        return self._routing_transitions[category][destination]

    def copy(self):
        network = super(MultiClassQueueNetwork, self).copy()
        network._routing_transitions = copy.deepcopy(self._routing_transitions)


class MultiClassQueueServer(QueueServer):

    def next_event(self):
        """Simulates the queue forward one event.

        Use :meth:`.simulate` instead.

        Returns
        -------
        out : :class:`.Agent` (sometimes)
            If the next event is a departure then the departing agent
            is returned, otherwise nothing is returned.

        See Also
        --------
        :meth:`.simulate` : Simulates the queue forward.
        """
        if self._departures[0]._time < self._arrivals[0]._time:
            new_depart = heappop(self._departures)
            self._current_t = new_depart._time
            self._num_total -= 1
            self.num_system -= 1
            self.num_departures += 1

            if self.collect_data and new_depart.agent_id in self.data:
                self.data[new_depart.agent_id][-1][2] = self._current_t

            if len(self.queue) > 0:
                agent = self.queue.popleft()
                if self.collect_data and agent.agent_id in self.data:
                    self.data[agent.agent_id][-1][1] = self._current_t

                agent._time = self.service_f(self._current_t, agent)
                agent.queue_action(self, 1)
                heappush(self._departures, agent)

            new_depart.queue_action(self, 2)
            self._update_time()
            return new_depart

        elif self._arrivals[0]._time < infty:
            arrival = heappop(self._arrivals)
            self._current_t = arrival._time

            if self._active:
                self._add_arrival()

            self.num_system += 1
            self._num_arrivals += 1

            if self.collect_data:
                b = 0 if self.num_system <= self.num_servers else 1
                if arrival.agent_id not in self.data:
                    self.data[arrival.agent_id] = \
                        [[arrival._time, 0, 0, len(self.queue) + b, self.num_system]]
                else:
                    self.data[arrival.agent_id]\
                        .append([arrival._time, 0, 0, len(self.queue) + b, self.num_system])  # noqa: E501

            arrival.queue_action(self, 0)

            if self.num_system <= self.num_servers:
                if self.collect_data:
                    self.data[arrival.agent_id][-1][1] = arrival._time

                arrival._time = self.service_f(arrival._time, arrival)
                arrival.queue_action(self, 1)
                heappush(self._departures, arrival)
            else:
                self.queue.append(arrival)

            self._update_time()


ClassAgentID = collections.namedtuple(
    typename='AgentID',
    field_names=['edge_index', 'agent_qid', 'category']
)


class ClassedAgent(Agent):
    def __init__(self, agent_id=(0, 0), category=None, **kwargs):
        self._category = category
        super(ClassedAgent, self).__init__(agent_id=agent_id, **kwargs)
        self.agent_id = ClassAgentID(agent_id[0], agent_id[1], self.category)

    @property
    def category(self):
        return self._category or self.__class__.__name__

    def desired_destination(self, network, edge):

        n = len(network.out_edges[edge[1]])
        if n <= 1:
            return network.out_edges[edge[1]][0]

        pr = network.routing_transition(edge[1], self.category)
        u = np.random.uniform()
        k = _choice(pr, u, n)
        return network.out_edges[edge[1]][k]
