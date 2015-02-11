from .queue_servers   import QueueServer, LossQueue
from .agents          import Agent
from numpy.random     import randint, exponential
from heapq            import heappush, heappop

import numpy as np
import copy


class ResourceAgent(Agent) :
    """An agent designed to interact with the :class:`~ResourceQueue` class.

    When an ``ResourceAgent`` departs from a :class:`~ResourceQueue`, they take
    a *resource* from the queue if it does not have a resource yet. It does this
    by reducing the number of servers at that queue by one. If a ``ResourceAgent``
    with a resource arrives at the next :class:`~ResourceQueue` (this could be the
    same queue) the :class:`~ResourceQueue` adds a resource to that queue by adding
    increasing the number of servers there by one; the ``ResourceAgent`` is then deleted.
    """
    def __init__(self, issn) :
        Agent.__init__(self, issn)
        self._has_resource = False
        self._had_resource = False

    def __repr__(self) :
        return "ResourceAgent. issn: %s, time: %s" % (self.issn, self._time)


    def queue_action(self, queue, *args, **kwargs) :
        """Function that specifies the interaction with a :class:`~ResourceQueue` 
        upon departure.

        Upon departure from a :class:`~ResourceQueue` (or a :class:`~QueueServer`), this
        method is called where the ``queue`` is the :class:`~ResourceQueue` that the agent
        is departing from. If the agent does not already have a resource then it decrements
        the number of servers at :class:`~ResourceQueue` by one.

        Parameters
        ----------
        queue : :class:`~QueueServer`
            The instance of the queue that the ``ResourceAgent`` will interact with. If
            It's not a :class:`~ResourceQueue` then it does nothing.
        """
        if isinstance(queue, ResourceQueue) :
            if self._has_resource :
                self._has_resource = False
                self._had_resource = True
            else :
                if queue.nServers > 0 :
                    queue.set_nServers(queue.nServers - 1)
                    self._has_resource = True
                    self._had_resource = False


    def __deepcopy__(self, memo) :
        new_agent               = Agent.__deepcopy__(self, memo)
        new_agent._has_resource = copy.deepcopy(self._has_resource)
        new_agent._had_resource = copy.deepcopy(self._had_resource)
        return new_agent



class ResourceQueue(LossQueue) :
    """An queue designed to interact with the :class:`~ResourceAgent` class.

    If a ``ResourceAgent`` does not have a resource already it will take a
    *resource* from this queue when it departs. It does this by reducing the
    number of servers here by one. When that agent arrives to this queue with a
    resource it adds one to the number of servers here upon arrival.

    Attributes
    ----------
    max_servers : int
        The maximum number of servers that be here. This is a soft max, and it
        is used to keep track of how often the queue will be overflowing with
        resources.
    over_max : int
        The number of times an agent has deposited a resource here when the
        number of servers was at ``max_servers``\.
    **kwargs :
        Any arguments to pass to :class:`~queueing_tool.queues.LossQueue`.
    """
    def __init__(self, nServers=10, AgentClass=ResourceAgent, qbuffer=0, **kwargs) :

        default_colors  = { 'edge_loop_color'   : [0.7, 0.7, 0.7, 0.50],
                            'edge_color'        : [0.7, 0.7, 0.7, 0.50],
                            'vertex_fill_color' : [1.0, 1.0, 1.0, 1.0],
                            'vertex_pen_color'  : [0.0, 0.235, 0.718, 1.0] }

        if 'colors' in kwargs :
            for col in set(default_colors.keys()) - set(kwargs['colors'].keys()) :
                kwargs['colors'][col] = default_colors[col]
        else :
            kwargs['colors'] = default_colors

        LossQueue.__init__(self, nServers, AgentClass, qbuffer, **kwargs)

        self.max_servers  = 2 * nServers
        self.over_max     = 0


    def __repr__(self) :
        tmp = "ResourceQueue: %s. servers: %s, max servers: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, self.max_servers, self.nArrivals, self.nDepartures, np.round(self._time, 3))
        return tmp


    def set_nServers(self, n) :
        self.nServers = n
        if n > self.max_servers :
            self.over_max += 1


    def next_event(self) :
        """Simulates the queue forward one event.

        This method behaves identically to a :class:`~LossQueue` if the
        arriving/departing agent anything other than a :class:`~ResourceAgent`.
        The differences are:
        * If the agent is a ``ResourceAgent`` and they have a resource
          then it deletes it upon arrival and adds one to ``nServers``.
        * If the :class:`~ResourceAgent` is arriving without a resource then
          nothing special happens.
        """
        if isinstance(self._arrivals[0], ResourceAgent) :
            if self._arrivals[0]._time < self._departures[0]._time :
                if self._arrivals[0]._has_resource :
                    new_arrival   = heappop(self._arrivals)
                    self._current_t = new_arrival._time
                    self.nTotal  -= 1
                    self.set_nServers(self.nServers+1)

                    if self._arrivals[0]._time < self._departures[0]._time :
                        self._time = self._arrivals[0]._time
                    else :
                        self._time = self._departures[0]._time

                elif self.nSystem < self.nServers :
                    QueueServer.next_event(self)

                else :
                    self.nBlocked      += 1
                    self.nArrivals[0]  += 1
                    self.nTotal        -= 1
                    new_arrival         = heappop(self._arrivals)
                    self._current_t       = new_arrival._time
                    if self._arrivals[0]._time < self._departures[0]._time :
                        self._time = self._arrivals[0]._time
                    else :
                        self._time = self._departures[0]._time

            elif self._departures[0]._time < self._arrivals[0]._time :
                return QueueServer.next_event(self)
        else :
            return LossQueue.next_event(self)


    def current_color(self, which=0) :
        if which == 1 :
            nSy = self.nServers
            cap = self.max_servers
            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            color    = [ i * tmp / 0.9 for i in self.colors['edge_loop_color'] ]
            color[3] = 0.0
  
        elif which == 2 :
            color = self.colors['vertex_pen_color']
        else :
            nSy = self.nServers
            cap = self.max_servers
            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            if self.edge[0] == self.edge[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_fill_color'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_color'] ]
                color[3] = 0.5

        return color


    def clear(self) :
        """Resets all class attributes, and clears out all agents."""
        LossQueue.clear(self)
        self.nBlocked  = 0
        self.over_max  = 0


    def __deepcopy__(self, memo) :
        new_server              = LossQueue.__deepcopy__(self, memo)
        new_server.max_servers  = copy.copy(self.max_servers)
        new_server.over_max     = copy.copy(self.over_max)
        return new_server



class InfoAgent(Agent) :
    """An agent that carries information about the queue around.

    This agent is designed to work with the :class:`~InfoQueue`. It
    collects load data from each queue that it visits.

    Parameters
    ----------
    issn : tuple (optional, the default is (0,0))
        A unique identifier for an agent. Is set automatically by the
        :class:`~QueueServer` that instantiates the agent. The first slot is
        the ``QueueServer``'s edge index and the second slot is specifies the
        ``InfoAgent``'s instantiation number for that queue.
    net_size : int (optional, the default is 1)
        The size of the network.
    **kwargs :
        Any arguments to pass to :class:`~queueing_tool.queues.Agent`.        
    """
    def __init__(self, issn=(0,0), net_size=1, **kwargs) :
        Agent.__init__(self, issn, **kwargs)

        self.stats    = np.zeros((net_size, 3), np.int32 )
        self.net_data = np.ones((net_size, 3)) * -1

    def __repr__(self) :
        return "InfoAgent. issn: %s, time: %s" % (self.issn, self._time)


    def add_loss(self, qedge, *args, **kwargs) : # qedge[2] is the edge_index of the queue
        self.stats[qedge[2], 2] += 1


    def get_beliefs(self) :
        return self.net_data[:, 2]


    def _set_dest(self, net=None, dest=None) :
        if dest != None :
            self.dest = int(dest)
        else :
            nodes   = net.g.gp['node_index']['dest_road']
            dLen    = net.dest_count
            rLen    = net.nV - dLen - net.fcq_count
            probs   = [0.3 / dLen for k in range(dLen)]
            probs.extend([0.7/rLen for k in range(rLen)])
            dest    = int(choice(nodes, size=1, p=probs))

            if self.old_dest != None :
                while dest == int(self.old_dest) :
                    dest = int(choice(nodes, size=1, p=probs))
            self.dest = dest


    def desired_destination(self, network, edge, **kwargs) :
        if self.dest != None and edge[1] == self.dest :
            self.old_dest   = self.dest
            self.dest       = None
            self.trip_t[1] += network.t - self.trip_t[0] 
            self.trips     += 1
            self._set_dest(net = network)

        elif self.dest == None :
            self.trip_t[0]  = network.t
            self._set_dest(net = network)
            while self.dest == edge[1] :
                self._set_dest(net = network)
        
        z   = network.shortest_path[edge[1], self.dest]
        z   = network.g.edge(edge[1], z)
        return z


    def queue_action(self, queue, *args, **kwargs) :
        if isinstance(queue, InfoQueue) :
            ### update information
            a = logical_or(self.net_data[:, 0] < queue.net_data[:, 0], self.net_data[:, 0] == -1)
            self.net_data[a, :] = queue.net_data[a, :]

            ### stamp this information
            n   = queue.edge[2]    # This is the edge_index of the queue
            self.stats[n, 0]    = self.stats[n, 0] + (queue.data[self.issn][-1][1] - queue.data[self.issn][-1][0])
            self.stats[n, 1]   += 1 if (queue.data[self.issn][-1][1] - queue.data[self.issn][-1][0]) > 0 else 0
            self.net_data[n, :] = queue._current_t, queue.nServers, queue.nSystem / queue.nServers


    def __deepcopy__(self, memo) :
        new_agent          = Agent.__deepcopy__(self, memo)
        new_agent.stats    = copy.deepcopy(self.stats)
        new_agent.net_data = copy.deepcopy(self.net_data)
        return new_agent



class InfoQueue(LossQueue) :
    """A queue that stores information about the network.

    This queue gets information about the state of the network (number of
    ``Agent``'s at other queues and loads) from arriving :class:`~InfoAgent`\'s.
    When an ``InfoAgent`` arrives, the queue extracts all the information the 
    agent has and replaces it's own network out network information with the
    agents more up-to-date information (if the agent has any). When an
    ``InfoAgent`` departs this queue, the queue gives the departing agent all
    the information it has about the state of the network.

    Parameters
    ----------
    net_size : int (optional, the default is 1)
        The total number of queues/edges in the network.
    AgentClass : class (optional, the default is ``InfoAgent``\)
        The class of agents that arrive from outside the network.
    qbuffer : int (optional, the default is infinity)
        The maximum length of the queue/line.
    **kwargs :
        Extra parameters to pass to :class:`~queueing_tool.queues.LossQueue`.
    """
    def __init__(self, net_size=1, AgentClass=InfoAgent, qbuffer=np.infty, **kwargs) :
        LossQueue.__init__(self, AgentClass, qbuffer, **kwargs)

        self.networking(net_size)

    def __repr__(self) :
        tmp = "InfoQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, len(self._queue), self.nArrivals, self.nDepartures, np.round(self._time, 3))
        return tmp

    def __repr__(self) :
        tmp = "InfoQueue: %s. servers: %s, max servers: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, self.max_servers, self.nArrivals, self.nDepartures, np.round(self._time, 3))
        return tmp


    def networking(self, network_size) :
        self.net_data = -1 * np.ones((network_size, 3))


    def extract_information(self, agent) :
        if isinstance(agent, InfoAgent) :
            a = self.net_data[:, 0] < agent.net_data[:, 0]
            self.net_data[a, :] = agent.net_data[a, :]


    def _add_arrival(self, *args) :
        if len(args) > 0 :
            self.nTotal += 1
            heappush(self._arrivals, args[0])
        else : 
            if self._current_t >= self._next_ct :
                self.nTotal  += 1
                self._next_ct = self.arrival_f(self._current_t)

                if self._next_ct >= self.deactive_t :
                    self.active = False
                    return

                new_arrival   = self.AgentClass(self.edge, len(self.net_data))
                new_arrival.set_arrival(self._next_ct)
                heappush(self._arrivals, new_arrival)

                self.nArrivals[1] += 1
                if self.nArrivals[1] >= self.active_cap :
                    self.active = False

        if self._arrivals[0]._time < self._departures[0]._time :
            self._time = self._arrivals[0]._time
        else :
            self._time = self._departures[0]._time


    def append_departure(self, agent, t) :
        self.extract_information(agent)
        LossQueue.append_departure(self, agent, t)


    def next_event(self) :
        if self._arrivals[0]._time < self._departures[0]._time :
            self.extract_informaInfoAgention(self._arrivals[0])

        LossQueue.next_event(self)


    def clear(self) :
        LossQueue.clear(self)
        self.networking( len(self.net_data) )


    def __deepcopy__(self, memo) :
        new_server          = LossQueue.__deepcopy__(self, memo)
        new_server.net_data = copy.copy(self.net_data)
        return new_server

