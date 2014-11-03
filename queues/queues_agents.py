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

    def __init__(self, issn, *args) :
        self.issn     = issn
        self.time     = 0                                     # agents arrival or departure time
        self.dest     = None
        self.old_dest = None
        self.resting  = False
        self.trips    = 0
        self.type     = 0
        self.rest_t   = [0, 0]
        self.trip_t   = [0, 0]
        self.arr_ser  = [0, 0]
        self.od       = [0, 0]
        self.blocked  = 0

    def __repr__(self) :
        return "Agent. issn: %s, time: %s" % (self.issn, self.time)

    def __lt__(a,b) :
        return a.time < b.time

    def __gt__(a,b) :
        return a.time > b.time

    def __eq__(a,b) :
        return a.time == b.time

    def __le__(a,b) :
        return a.time <= b.time

    def __ge__(a,b) :
        return a.time >= b.time


    def set_arrival(self, t) :
        self.time = t


    def set_departure(self, t) :
        self.time = t


    def set_type(self, n) :
        self.type = n


    def set_rest(self) :
        self.resting    = False
        self.rest_t[1] += self.time - self.rest_t[0]


    def add_loss(self, *args, **kwargs) :
        self.blocked   += 1 


    def desired_destination(self, *info) :
        network, qissn = info[:2]
        n   = len( network.adjacency[qissn[1]] )
        d   = randint(0, n)
        z   = network.adjacency[qissn[1]][d]
        return z


    def get_beliefs(self) :
        pass


    def queue_action(self, queue, *args, **kwargs) :
        pass


    def __deepcopy__(self, memo) :
        new_agent           = self.__class__(self.issn)
        new_agent.__dict__  = copy.deepcopy(self.__dict__, memo)
        return new_agent



class QueueServer :

    def __init__(self, nServers=1, issn=(0,0,0), active=False, fArrival=lambda x : x + exponential(1), 
            fDepart =lambda x : x + exponential(0.95), AgentClass=Agent) :

        self.issn       = issn
        self.nServers   = nServers
        self.AgentClass = AgentClass
        self.nArrivals  = 0
        self.nDeparts   = 0
        self.nSystem    = 0
        self.nTotal     = 0

        self.local_t    = 0
        self.time       = infty
        self.active     = active
        self.next_ct    = 0

        self.colors     = {'edge_normal'   : [0.9, 0.9, 0.9, 0.5],
                           'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                           'vertex_pen'    : [0.0, 0.5, 1.0, 1.0]}

        self.queue      = collections.deque()
        self.arrivals   = []
        self.departures = []
        inftyAgent      = AgentClass(0)
        inftyAgent.time = infty

        heappush(self.arrivals, inftyAgent)
        heappush(self.departures, inftyAgent)

        self.fArrival   = fArrival
        self.fDepart    = fDepart

    def __repr__(self) :
        tmp = "QueueServer: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.issn[2], self.nServers, len(self.queue), self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp

    def __lt__(a,b) :
        return a.time < b.time

    def __gt__(a,b) :
        return a.time > b.time

    def __eq__(a,b) :
        return a.time == b.time

    def __le__(a,b) :
        return a.time <= b.time

    def __ge__(a,b) :
        return a.time >= b.time


    def blocked(self) :
        return 0


    def initialize(self, add_arrival=True) :
        self.active = True
        if add_arrival :
            self._add_arrival()


    def set_nServers(self, n) :
        if n > 0 :
            self.nServers = n
        else :
            print("nServers must be positive, tried to set to %s.\n%s" % (n, str(self)) )


    def nQueued(self) :
        n = 0 if self.nServers == infty else self.nSystem - self.nServers
        return max([n, 0])


    def _add_arrival(self, *args) :
        if len(args) > 0 :
            self.nTotal += 1
            heappush(self.arrivals, args[0])
        else : 
            if self.local_t >= self.next_ct :
                self.nTotal  += 1
                self.next_ct  = self.fArrival(self.local_t)
                new_arrival   = self.AgentClass(self.nArrivals+1)
                new_arrival.set_arrival( self.next_ct )
                heappush(self.arrivals, new_arrival)

        if self.arrivals[0].time < self.departures[0].time :
            self.time = self.arrivals[0].time
        else :
            self.time = self.departures[0].time


    def next_event_type(self) :
        if self.arrivals[0].time < self.departures[0].time :
            return 1
        elif self.arrivals[0].time > self.departures[0].time :
            return 2
        else :
            return 0


    def append_departure(self, agent, t) :
        self.nSystem       += 1
        self.nArrivals     += 1
        agent.arr_ser[0]    = t

        if self.nSystem <= self.nServers :
            agent.arr_ser[1]    = t
            agent.set_departure(self.fDepart(t))
            heappush(self.departures, agent)
        else :
            self.queue.append(agent)

        if self.arrivals[0].time < self.departures[0].time :
            self.time = self.arrivals[0].time
        else :
            self.time = self.departures[0].time


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            arrival       = heappop(self.arrivals)
            self.local_t  = arrival.time

            if self.active :
                self._add_arrival()

            self.nSystem       += 1
            self.nArrivals     += 1
            arrival.arr_ser[0]  = arrival.time

            if self.nSystem <= self.nServers :
                arrival.arr_ser[1]    = arrival.time
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

            if len(self.queue) > 0 :
                agent             = self.queue.popleft()
                agent.arr_ser[1]  = self.local_t
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

            if self.issn[0] == self.issn[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
                color[3] = 0.7 - tmp / 1.8

        return color


    def clear(self) :
        self.nArrivals  = 0
        self.nSystem    = 0
        self.nTotal     = 0
        self.local_t    = 0
        self.time       = infty
        self.next_ct    = 0
        self.queue      = collections.deque()
        self.arrivals   = []
        self.departures = []
        inftyAgent      = self.AgentClass(0)
        inftyAgent.time = infty

        heappush(self.arrivals, inftyAgent)
        heappush(self.departures, inftyAgent)


    def __deepcopy__(self, memo) :
        new_server          = self.__class__()
        new_server.__dict__ = copy.deepcopy(self.__dict__, memo)
        return new_server



class LossQueue(QueueServer) :

    def __init__(self, nServers=1, issn=0, active=False, fArrival=lambda x : x + exponential(1), 
            fDepart =lambda x : x + exponential(0.95), AgentClass=Agent, queue_cap=0) :

        QueueServer.__init__(self, nServers, issn, active, fArrival, fDepart, AgentClass)

        self.colors     = { 'edge_normal'   : [0.7, 0.7, 0.7, 0.50],
                            'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                            'vertex_pen'    : [0.133, 0.545, 0.133, 1.0] }
        self.nBlocked   = 0
        self.queue_cap  = queue_cap


    def __repr__(self) :
        tmp = "LossQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.issn[2], self.nServers, len(self.queue), self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp


    def blocked(self) :
        return (self.nBlocked / self.nArrivals) if self.nArrivals > 0 else 0


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            if self.nSystem < self.nServers + self.queue_cap :
                self.arrivals[0].set_rest()

                QueueServer.next_event(self)
            else :
                self.nBlocked  += 1
                self.nArrivals += 1
                self.nSystem   += 1
                new_arrival     = heappop(self.arrivals)
                new_arrival.add_loss(self.issn)

                self.local_t    = new_arrival.time
                if self.active :
                    self._add_arrival()

                new_arrival.arr_ser[0]  = self.local_t
                new_arrival.arr_ser[1]  = self.local_t

                heappush(self.departures, new_arrival)

                if self.arrivals[0].time < self.departures[0].time :
                    self.time = self.arrivals[0].time
                else :
                    self.time = self.departures[0].time

        elif self.departures[0].time < self.arrivals[0].time :
            return QueueServer.next_event(self)


    def clear(self) :
        QueueServer.clear(self)
        self.nBlocked  = 0



class MarkovianQueue(QueueServer) :

    def __init__(self, nServers=1, issn=(0,0,0), active=False, aRate=1, dRate=1.1, AgentClass=Agent) :
        aMean = 1 / aRate
        dMean = 1 / dRate
        QueueServer.__init__(self, nServers, issn, active, lambda x : x + exponential(aMean),
            lambda x : x + exponential(dMean), AgentClass)

        self.rates  = [aRate, dRate]

    def __repr__(self) :
        tmp = "MarkovianQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s, rates: %s" \
            %  (self.issn[2], self.nServers, len(self.queue), self.nArrivals, 
                self.nDeparts, np.round(self.time, 3), self.rates)
        return tmp


    def change_rates(self, aRate=None, dRate=None) :
        if aRate != None :
            aMean = 1 / aRate
            self.rates[0] = aRate
            self.fArrival = lambda x : x + exponential(aMean)
        if dRate != None :
            dMean = 1 / dRate
            self.rates[1] = dRate
            self.fDepart  = lambda x : x + exponential(dMean)

