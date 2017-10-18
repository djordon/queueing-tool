from heapq import heappush, heappop

from numpy import logical_or, infty
import numpy as np

from queueing_tool.queues.agents import Agent
from queueing_tool.queues.queue_servers import LossQueue


class ResourceAgent(Agent):
    """An agent designed to interact with the :class:`.ResourceQueue`
    class.

    When an ``ResourceAgent`` departs from a :class:`.ResourceQueue`,
    they take a *resource* from the queue if the agent does not have
    a resource yet. It does this by reducing the number of servers at
    that queue by one. When a ``ResourceAgent`` with a resource arrives
    at a :class:`.ResourceQueue`, the ``ResourceAgent`` adds
    a resource to that queue by increasing the number of
    servers there by one; the ``ResourceAgent`` is then deleted.
    """
    def __init__(self, agent_id=(0, 0)):
        super(ResourceAgent, self).__init__(agent_id)
        self._has_resource = False
        self._had_resource = False

    def __repr__(self):
        return "ResourceAgent; agent_id:{0}. Time: {1}".format(self.agent_id, round(self._time, 3))

    def queue_action(self, queue, *args, **kwargs):
        """Function that specifies the interaction with a
        :class:`.ResourceQueue` upon departure.

        When departuring from a :class:`.ResourceQueue` (or a
        :class:`.QueueServer`), this method is called. If the agent
        does not already have a resource then it decrements the number
        of servers at :class:`.ResourceQueue` by one. Note that this
        only applies to :class:`ResourceQueue's<.ResourceQueue>`.

        Parameters
        ----------
        queue : :class:`.QueueServer`
            The instance of the queue that the ``ResourceAgent`` will
            interact with.
        """
        if isinstance(queue, ResourceQueue):
            if self._has_resource:
                self._has_resource = False
                self._had_resource = True
            else:
                if queue.num_servers > 0:
                    queue.set_num_servers(queue.num_servers - 1)
                    self._has_resource = True
                    self._had_resource = False


class ResourceQueue(LossQueue):
    """An queue designed to interact with the :class:`.ResourceAgent`
    class.

    If a :class:`.ResourceAgent` does not have a resource already it
    will take a *resource* from this queue when it departs. It does
    this by reducing the number of servers here by one. When a
    :class:`.ResourceAgent` arrives to this queue with a resource, it
    adds one to the number of servers here. The :class:`.ResourceAgent`
    is then deleted.

    Attributes
    ----------
    max_servers : int
        The maximum number of servers that can be here. This is a soft
        max, and it is only used to keep track of how often the queue
        will be overflowing with resources.
    over_max : int
        The number of times an agent has deposited a resource here when
        the number of servers was at ``max_servers``.
    kwargs
        Any arguments to pass to :class:`.LossQueue`.
    """

    _default_colors = {
        'edge_loop_color': [0.7, 0.7, 0.7, 0.50],
        'edge_color': [0.7, 0.7, 0.7, 0.50],
        'vertex_fill_color': [1.0, 1.0, 1.0, 1.0],
        'vertex_pen_color': [0.0, 0.235, 0.718, 1.0]
    }

    def __init__(self, num_servers=10, AgentFactory=ResourceAgent, qbuffer=0, **kwargs):
        super(ResourceQueue, self).__init__(
            num_servers=num_servers,
            AgentFactory=AgentFactory,
            qbuffer=qbuffer,
            **kwargs
        )

        self.max_servers = 2 * num_servers
        self.over_max = 0

    def __repr__(self):
        my_str = ("ResourceQueue:{0}. Servers: {1}, max servers: {2}, "
                  "arrivals: {3}, departures: {4}, next time: {5}")
        arg = (self.edge[2], self.num_servers, self.max_servers,
               self.num_arrivals, self.num_departures, round(self._time, 3))
        return my_str.format(*arg)

    def set_num_servers(self, n):
        self.num_servers = n
        if n > self.max_servers:
            self.over_max += 1

    def next_event(self):
        """Simulates the queue forward one event.

        This method behaves identically to a :class:`.LossQueue` if the
        arriving/departing agent is anything other than a
        :class:`.ResourceAgent`. The differences are;

        Arriving:

        * If the :class:`.ResourceAgent` has a resource then it deletes
          the agent upon arrival and adds one to ``num_servers``.
        * If the :class:`.ResourceAgent` is arriving without a resource
          then nothing special happens.

        Departing:

        * If the :class:`.ResourceAgent` does not have a resource, then
          ``num_servers`` decreases by one and the agent then *has a
          resource*.

        Use :meth:`~QueueServer.simulate` for simulating instead.
        """
        if isinstance(self._arrivals[0], ResourceAgent):
            if self._departures[0]._time < self._arrivals[0]._time:
                return super(ResourceQueue, self).next_event()
            elif self._arrivals[0]._time < infty:
                if self._arrivals[0]._has_resource:
                    arrival = heappop(self._arrivals)
                    self._current_t = arrival._time
                    self._num_total -= 1
                    self.set_num_servers(self.num_servers + 1)

                    if self.collect_data:
                        t = arrival._time
                        if arrival.agent_id not in self.data:
                            self.data[arrival.agent_id] = [[t, t, t, len(self.queue), self.num_system]]
                        else:
                            self.data[arrival.agent_id].append([t, t, t, len(self.queue), self.num_system])

                    if self._arrivals[0]._time < self._departures[0]._time:
                        self._time = self._arrivals[0]._time
                    else:
                        self._time = self._departures[0]._time

                elif self.num_system < self.num_servers:
                    super(ResourceQueue, self).next_event()

                else:
                    self.num_blocked += 1
                    self._num_arrivals += 1
                    self._num_total -= 1
                    arrival = heappop(self._arrivals)
                    self._current_t = arrival._time

                    if self.collect_data:
                        if arrival.agent_id not in self.data:
                            self.data[arrival.agent_id] = [[arrival._time, 0, 0, len(self.queue), self.num_system]]
                        else:
                            self.data[arrival.agent_id].append([arrival._time, 0, 0, len(self.queue), self.num_system])

                    if self._arrivals[0]._time < self._departures[0]._time:
                        self._time = self._arrivals[0]._time
                    else:
                        self._time = self._departures[0]._time
        else:
            return super(ResourceQueue, self).next_event()

    def _current_color(self, which=0):
        if which == 1:
            nSy = self.num_servers
            cap = self.max_servers
            div = 5. if cap <= 1 else (3. * cap)
            tmp = 0.9 - min(nSy / div, 0.9)

            color = [i * tmp / 0.9 for i in self.colors['edge_loop_color']]
            color[3] = 0.0

        elif which == 2:
            color = self.colors['vertex_pen_color']
        else:
            nSy = self.num_servers
            cap = self.max_servers
            div = 5. if cap <= 1 else (3. * cap)
            tmp = 0.9 - min(nSy / div, 0.9)

            if self.edge[0] == self.edge[1]:
                color = [i * tmp / 0.9 for i in self.colors['vertex_fill_color']]
                color[3] = 1.0
            else:
                color = [i * tmp / 0.9 for i in self.colors['edge_color']]
                color[3] = 0.5

        return color

    def clear(self):
        super(ResourceQueue, self).clear()
        self.num_blocked = 0
        self.over_max = 0


class InfoAgent(Agent):
    """\
    An agent that carries information about the :class:`.QueueNetwork`
    around.

    This agent is designed to work with the :class:`.InfoQueue`. It
    collects load data (utilization rate, and the number of agents
    waiting to be served) from each queue that it visits.

    Parameters
    ----------
    agent_id : tuple (optional, default: ``(0, 0)``)
        A unique identifier for an agent. Is set automatically by the
        queue that instantiates the agent. The first slot is queues
        edge index and the second slot is specifies the instantiation
        number for that queue.
    net_size : int (optional, default: 1)
        The number of edges in the network.
    **kwargs :
        Any arguments to pass to :class:`.Agent`.
    """
    def __init__(self, agent_id=(0, 0), net_size=1, **kwargs):
        super(InfoAgent, self).__init__(agent_id, **kwargs)

        self.stats = np.zeros((net_size, 3), np.int32)
        self.net_data = np.ones((net_size, 3)) * -1

    def __repr__(self):
        return "InfoAgent; agent_id:{0}. Time: {1}".format(self.agent_id, round(self._time, 3))

    def add_loss(self, qedge, *args, **kwargs):  # qedge[2] is the edge_index of the queue
        self.stats[qedge[2], 2] += 1

    def get_beliefs(self):
        return self.net_data[:, 2]

    def queue_action(self, queue, *args, **kwargs):
        if isinstance(queue, InfoQueue):
            # update information
            a = logical_or(self.net_data[:, 0] < queue.net_data[:, 0], self.net_data[:, 0] == -1)
            self.net_data[a, :] = queue.net_data[a, :]

            # stamp this information
            n = queue.edge[2]    # This is the edge_index of the queue
            if self.agent_id in queue.data:
                tmp = queue.data[self.agent_id][-1][1] - queue.data[self.agent_id][-1][0]
                self.stats[n, 0] = self.stats[n, 0] + tmp
                self.stats[n, 1] += 1 if tmp > 0 else 0
            self.net_data[n, :] = queue._current_t, queue.num_servers, queue.num_system / queue.num_servers


class InfoQueue(LossQueue):
    """A queue that stores information about the network.

    This queue gets information about the state of the network (number
    of :class:`Agent's<.Agent>` at other queues) from arriving
    :class:`InfoAgent's<.InfoAgent>`. When an :class:`.InfoAgent`
    arrives, the queue extracts all the information the agent has and
    replaces it's out-of-date network information with the agents more
    up-to-date information (if the agent has any). When an
    :class:`.InfoAgent` departs this queue, the queue gives the
    departing agent all the information it has about the state of the
    network.

    Parameters
    ----------
    net_size : int (optional, default: ``1``)
        The total number of edges in the network.
    AgentFactory : class (optional, default: :class:`.InfoAgent`)
        The class of agents that arrive from outside the network.
    qbuffer : int (optional, default: ``infty``)
        The maximum number of agents that can be waiting to be
        serviced.
    **kwargs :
        Extra parameters to pass to :class:`.LossQueue`.
    """
    def __init__(self, net_size=1, AgentFactory=InfoAgent, qbuffer=np.infty, **kwargs):
        super(InfoQueue, self).__init__(AgentFactory=AgentFactory, qbuffer=qbuffer, **kwargs)

        self.networking(net_size)

    def __repr__(self):
        my_str = ("InfoQueue:{0}. Servers: {1}, queued: {2}, "
                  "arrivals: {3}, departures: {4}, next time: {5}")
        arg = (self.edge[2], self.num_servers, len(self.queue),
               self.num_arrivals, self.num_departures, round(self._time, 3))
        return my_str.format(*arg)

    def networking(self, network_size):
        self.net_data = -1 * np.ones((network_size, 3))

    def extract_information(self, agent):
        if isinstance(agent, InfoAgent):
            a = self.net_data[:, 0] < agent.net_data[:, 0]
            self.net_data[a, :] = agent.net_data[a, :]

    def _add_arrival(self, agent=None):
        if agent is not None:
            self._num_total += 1
            heappush(self._arrivals, agent)
        else:
            if self._current_t >= self._next_ct:
                self._next_ct = self.arrival_f(self._current_t)

                if self._next_ct >= self.deactive_t:
                    self.active = False
                    return

                self._num_total += 1
                new_agent = self.AgentFactory((self.edge[2], self._oArrivals), len(self.net_data))
                new_agent._time = self._next_ct
                heappush(self._arrivals, new_agent)

                self._oArrivals += 1

                if self._oArrivals >= self.active_cap:
                    self._active = False

        if self._arrivals[0]._time < self._departures[0]._time:
            self._time = self._arrivals[0]._time

    def next_event(self):
        if self._arrivals[0]._time < self._departures[0]._time:
            self.extract_information(self._arrivals[0])

        return super(InfoQueue, self).next_event()

    def clear(self):
        super(InfoQueue, self).clear()
        self.networking(len(self.net_data))
