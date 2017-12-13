import collections
import copy

import numpy as np

from queueing_tool.network.queue_network import QueueNetwork
from queueing_tool.queues.agents import Agent
from queueing_tool.queues.choice import _choice


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

    def routing_transitions(self, destination, category=None):
        if category not in self._routing_transitions:
            category = None

        return self._routing_transitions[category][destination]

    def copy(self):
        network = super(MultiClassQueueNetwork, self).copy()
        network._routing_transitions = copy.deepcopy(self._routing_transitions)


class ClassedAgent(Agent):
    def __init__(self, agent_id=(0, 0), category=None, **kwargs):
        super(ClassedAgent, self).__init__(agent_id=agent_id, **kwargs)
        self._category = category

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
