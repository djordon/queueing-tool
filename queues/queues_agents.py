from numpy.random           import uniform, exponential, randint
from numpy                  import infty, log
from heapq                  import heappush, heappop

import numpy       as np
import collections
import copy


def arrival(rate, rate_max, t) :
    t   = t + exponential(rate_max)
    while rate_max * uniform() > rate(t) :
        t   = t + exponential(rate_max)
    return t


def departure(rate, rate_max, t) :
    t   = t + exponential(rate_max)
    while rate_max * uniform() > rate(t) :
        t   = t + exponential(rate_max)
    return t



class Agent :

    def __init__(self, issn=(0,0), *args) :
        self.issn     = issn
        self.time     = 0         # agents arrival or departure time
        self.dest     = None
        self.old_dest = None
        self.trips    = 0
        self.type     = 0
        self.trip_t   = [0, 0]
        self.od       = [0, 0]
        self.blocked  = 0

    def __repr__(self) :
        return "Agent. edge: %s, time: %s" % (self.issn, self.time)

    def __lt__(a, b) :
        return a.time < b.time

    def __gt__(a, b) :
        return a.time > b.time

    def __eq__(a, b) :
        return a.time == b.time

    def __le__(a, b) :
        return a.time <= b.time

    def __ge__(a, b) :
        return a.time >= b.time


    def set_arrival(self, t) :
        self.time = t


    def set_departure(self, t) :
        self.time = t


    def set_type(self, n) :
        self.type = n


    def add_loss(self, *args, **kwargs) :
        self.blocked   += 1 


    def desired_destination(self, *info) :
        network, qedge = info[:2]
        n   = len( network.adjacency[qedge[1]] )
        d   = randint(0, n)
        z   = network.adjacency[qedge[1]][d]
        return z


    def get_beliefs(self) :
        pass


    def queue_action(self, queue, *args, **kwargs) :
        pass


    def __deepcopy__(self, memo) :
        new_agent           = self.__class__()
        new_agent.issn      = copy.copy(self.issn)
        new_agent.time      = copy.copy(self.time)
        new_agent.dest      = copy.copy(self.dest)
        new_agent.old_dest  = copy.copy(self.old_dest)
        new_agent.trips     = copy.copy(self.trips)
        new_agent.type      = copy.copy(self.type)
        new_agent.blocked   = copy.copy(self.blocked)
        new_agent.trip_t    = copy.deepcopy(self.trip_t)
        new_agent.od        = copy.deepcopy(self.od)
        return new_agent



class InftyAgent :

    def __init__(self) :
        self.time = infty

    def __repr__(self) :
        return "InftyAgent"

    def __lt__(a, b) :
        return a.time < b.time

    def __gt__(a, b) :
        return a.time > b.time

    def __eq__(a, b) :
        return a.time == b.time

    def __le__(a, b) :
        return a.time <= b.time

    def __ge__(a, b) :
        return a.time >= b.time

    def __deepcopy__(self, memo) :
        return self.__class__()



class QueueServer :
    """
    The QueueServer class is designed to operate within the QueueNetwork
    class. Becuase of this, there are some variables that used by the
    QueueNetwork class:
        The edge variable is a tuple of integers with the source vertex,
        target vertex, and edge index.
        The colors variable is a dictionary that indicates the colors used
        to draw the edge (or vertex if the edge points to itself).
        The active variable dictates whether the Queue 'generates' arrivals
        from the outside world.
        This base class has arrival and departure functions that depend on
        the current time and nothing else.
        AgentClass dictates what kinds of Agents are generated when there
        is an arrival.
    """
    def __init__(self, nServers=1, edge=(0,0,0), fArrival=lambda x : x + exponential(1),
            fDepart =lambda x : x + exponential(0.95), AgentClass=Agent) :

        self.edge       = edge
        self.nServers   = nServers
        self.AgentClass = AgentClass
        self.nDeparts   = 0
        self.nSystem    = 0
        self.nTotal     = 0
        self.nArrivals  = [0, 0]            #  First slot: the total number of arrivals, 
                                            # Second slot: number of arrivals from the 'outside world'.
        self.local_t    = 0
        self.time       = infty
        self.active     = False
        self.cap        = infty
        self.next_ct    = 0

        self.keep_data  = False
        self.data       = {}                # agent issn : [arrival t, service start t, departure t]

        self.colors     = {'edge_normal'   : [0.9, 0.9, 0.9, 0.5],
                           'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                           'vertex_pen'    : [0.0, 0.5, 1.0, 1.0]}

        self.queue      = collections.deque()
        inftyAgent      = InftyAgent()
        self.arrivals   = [inftyAgent]
        self.departures = [inftyAgent]

        self.fArrival   = fArrival
        self.fDepart    = fDepart

    def __repr__(self) :
        tmp = "QueueServer: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, len(self.queue), self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp

    def __lt__(a, b) :
        return a.time < b.time

    def __gt__(a, b) :
        return a.time > b.time

    def __eq__(a, b) :
        return a.time == b.time

    def __le__(a, b) :
        return a.time <= b.time

    def __ge__(a, b) :
        return a.time >= b.time


    def blocked(self) :
        return 0


    def at_capacity(self) :
        return False


    def initialize(self, add_arrival=True) :
        self.active = True
        if add_arrival :
            self._add_arrival()


    def set_nServers(self, n) :
        if n > 0 :
            self.nServers = n
        else :
            raise Exception("nServers must be positive, tried to set to %s.\n%s" % (n, str(self)) )


    def nQueued(self) :
        n = 0 if self.nServers == infty else self.nSystem - self.nServers
        return max([n, 0])


    def _add_arrival(self, *args) :
        if len(args) > 0 :
            self.nTotal += 1
            heappush(self.arrivals, args[0])
        else : 
            if self.local_t >= self.next_ct :
                self.nTotal    += 1
                self.next_ct    = self.fArrival(self.local_t)
                new_arrival     = self.AgentClass( (self.edge[2], self.nArrivals[1]) )
                new_arrival.set_arrival(self.next_ct)

                heappush(self.arrivals, new_arrival)

                self.nArrivals[1] += 1
                if self.nArrivals[1] >= self.cap :
                    self.active = False

        if self.arrivals[0].time < self.departures[0].time :
            self.time = self.arrivals[0].time


    def append_departure(self, agent, t) :
        self.nSystem       += 1
        self.nArrivals[0]  += 1

        if self.nSystem <= self.nServers :
            agent.set_departure(self.fDepart(t))
            heappush(self.departures, agent)
        else :
            self.queue.append(agent)

        if self.arrivals[0].time >= self.departures[0].time :
            self.time = self.departures[0].time


    def delay_service(self) :
        agent = heappop(self.departures)
        agent.set_departure(self.fDepart(agent.time))
        heappush(self.departures, agent)

        if self.arrivals[0].time < self.departures[0].time :
            self.time = self.arrivals[0].time
        else :
            self.time = self.departures[0].time


    def next_event_type(self) :
        if self.arrivals[0].time < self.departures[0].time :
            return 1
        elif self.departures[0].time < self.arrivals[0].time :
            return 2
        elif self.departures[0].time < infty :
            return 2
        else :
            return 0


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            arrival       = heappop(self.arrivals)
            self.local_t  = arrival.time

            if self.active :
                self._add_arrival()

            self.nSystem       += 1
            self.nArrivals[0]  += 1

            if self.keep_data :
                if arrival.issn not in self.data :
                    self.data[arrival.issn] = [[arrival.time, 0, 0]]
                else :
                    self.data[arrival.issn].append([arrival.time, 0, 0])

            if self.nSystem <= self.nServers :
                if self.keep_data :
                    self.data[arrival.issn][-1][1] = arrival.time

                arrival.set_departure(self.fDepart(arrival.time))
                heappush(self.departures, arrival)
            else :
                self.queue.append(arrival)

            if self.arrivals[0].time < self.departures[0].time :
                self.time = self.arrivals[0].time
            else :
                self.time = self.departures[0].time
                
        elif self.departures[0].time < infty :
            new_depart      = heappop(self.departures)
            self.local_t    = new_depart.time
            self.nDeparts  += 1
            self.nTotal    -= 1
            self.nSystem   -= 1

            if self.keep_data and new_depart.issn in self.data :
                self.data[new_depart.issn][-1][2] = self.local_t

            if len(self.queue) > 0 :
                agent = self.queue.popleft()
                if self.keep_data and agent.issn in self.data :
                    self.data[agent.issn][-1][1] = self.local_t

                agent.set_departure(self.fDepart(self.local_t))
                heappush(self.departures, agent)

            new_depart.queue_action(self, 'departure')

            if self.arrivals[0].time < self.departures[0].time :
                self.time = self.arrivals[0].time
            else :
                self.time = self.departures[0].time

            return new_depart


    def current_color(self, which='') :
        if which == 'edge' :
            color = [0, 0, 0, 0]
  
        elif which == 'pen' :
            color = self.colors['vertex_pen']

        else :
            nSy = self.nSystem
            cap = self.nServers
            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            if self.edge[0] == self.edge[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
                color[3] = 0.7 - tmp / 1.8

        return color


    def clear(self) :
        self.nArrivals  = [0, 0]
        self.nDeparts   = 0
        self.nSystem    = 0
        self.nTotal     = 0
        self.local_t    = 0
        self.time       = infty
        self.next_ct    = 0
        self.keep_data  = False
        self.data       = {}
        self.queue      = collections.deque()
        inftyAgent      = InftyAgent()
        self.arrivals   = [inftyAgent]
        self.departures = [inftyAgent]


    def __deepcopy__(self, memo) :
        new_server            = self.__class__()
        new_server.edge       = copy.copy(self.edge)
        new_server.nServers   = copy.copy(self.nServers)
        new_server.cap        = copy.copy(self.cap)
        new_server.nDeparts   = copy.copy(self.nDeparts)
        new_server.nSystem    = copy.copy(self.nSystem)
        new_server.nTotal     = copy.copy(self.nTotal)
        new_server.local_t    = copy.copy(self.local_t)
        new_server.time       = copy.copy(self.time)
        new_server.active     = copy.copy(self.active)
        new_server.next_ct    = copy.copy(self.next_ct)
        new_server.keep_data  = copy.copy(self.keep_data)
        new_server.data       = copy.deepcopy(self.data)
        new_server.nArrivals  = copy.deepcopy(self.nArrivals)
        new_server.colors     = copy.deepcopy(self.colors)
        new_server.queue      = copy.deepcopy(self.queue, memo)
        new_server.arrivals   = copy.deepcopy(self.arrivals, memo)
        new_server.departures = copy.deepcopy(self.departures, memo)
        new_server.fArrival   = self.fArrival
        new_server.fDepart    = self.fDepart
        new_server.AgentClass = self.AgentClass
        return new_server



class LossQueue(QueueServer) :

    def __init__(self, nServers=1, edge=(0,0,0), fArrival=lambda x : x + exponential(1),
            fDepart =lambda x : x + exponential(0.95), AgentClass=Agent, qbuffer=0) :

        QueueServer.__init__(self, nServers, edge, fArrival, fDepart, AgentClass)

        self.colors   = { 'edge_normal'   : [0.7, 0.7, 0.7, 0.50],
                          'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                          'vertex_pen'    : [0.133, 0.545, 0.133, 1.0] }
        self.nBlocked = 0
        self.buffer   = qbuffer


    def __repr__(self) :
        tmp = "LossQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, len(self.queue), self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp


    def blocked(self) :
        return (self.nBlocked / self.nArrivals[0]) if self.nArrivals[0] > 0 else 0


    def at_capacity(self) :
        return self.nSystem >= self.nServers + self.buffer


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            if self.nSystem < self.nServers + self.buffer :
                QueueServer.next_event(self)
            else :
                self.nBlocked      += 1
                self.nArrivals[0]  += 1
                self.nSystem       += 1

                new_arrival   = heappop(self.arrivals)
                new_arrival.add_loss(self.edge)

                self.local_t  = new_arrival.time
                if self.active :
                    self._add_arrival()

                heappush(self.departures, new_arrival)

                if self.arrivals[0].time < self.departures[0].time :
                    self.time = self.arrivals[0].time
                else :
                    self.time = self.departures[0].time

        elif self.departures[0].time < self.arrivals[0].time :
            return QueueServer.next_event(self)


    def current_color(self, which='') :
        if which == 'edge' :
            color = [0, 0, 0, 0]

        elif which == 'pen' :
            color = self.colors['vertex_pen']

        else :
            nSy = self.nSystem
            cap = self.nServers
            tmp = 0.9 - min(nSy, 0.9) if cap <= 1 else 0.9 - min(nSy / cap, 0.9)

            if self.edge[0] == self.edge[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
                color[3] = 0.7 - tmp / 1.8

        return color


    def clear(self) :
        QueueServer.clear(self)
        self.nBlocked  = 0


    def __deepcopy__(self, memo) :
        new_server          = QueueServer.__deepcopy__(self, memo)
        new_server.nBlocked = copy.copy(self.nBlocked)
        new_server.buffer   = copy.copy(self.buffer)
        return new_server



class NullQueue(QueueServer) :

    def __init__(self, servers=1, edge=(0,0,0), *args, **kwargs) :
        QueueServer.__init__(self, 1, edge)

    def __repr__(self) :
        return "NullQueue: %s." %  (self.edge[2])

    def initialize(self, *args, **kwargs) :
        pass

    def set_nServers(self, *args, **kwargs) :
        pass

    def nQueued(self) :
        return 0

    def _add_arrival(self, *args, **kwargs) :
        pass

    def append_departure(self, *args, **kwargs) :
        pass

    def delay_service(self) :
        pass

    def next_event_type(self) :
        return 0

    def next_event(self) :
        pass

    def current_color(self, which='') :
        if which == 'edge' :
            color = [0, 0, 0, 0]
        elif which == 'pen' :
            color = [0, 0, 0, 1]
        else :
            color = [1, 1, 1, 1]
        return color

    def clear(self) :
        pass

    def __deepcopy__(self, memo) :
        return self.__class__()
