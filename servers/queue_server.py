from numpy.random           import uniform
from numpy                  import infty, log
from heapq                  import heappush, heappop
from .. agents.queue_agents import Agent, SmartAgent, LearningAgent, ResourceAgent

import numpy       as np
import collections
import copy


def arrival(rate, rate_max, t) :
    t   = t - log(uniform()) / rate_max
    while rate_max * uniform() > rate(t) :
        t   = t - log(uniform()) / rate_max
    return t


def departure(rate, rate_max, t) :
    t   = t - log(uniform()) / rate_max
    while rate_max * uniform() > rate(t) :
        t   = t - log(uniform()) / rate_max
    return t



class QueueServer :

    def __init__(self, nServers=1, issn=(0,0,0), active=False, net_size=1,
            fArrival=lambda x : x - log(uniform()) / 1, 
            fDepart =lambda x : x - log(uniform()) / 1.1,
            fDepart_mu=lambda x : 1/1.1, AgentClass=Agent) :

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
                           'vertex_pen'    : [0.0, 0.5, 1.0, 1.0] }

        self.queue      = collections.deque()
        self.arrivals   = []
        self.departures = []
        inftyAgent      = Agent(0, 1)
        inftyAgent.time = infty

        heappush(self.arrivals, inftyAgent)
        heappush(self.departures, inftyAgent)

        self.fArrival   = fArrival
        self.fDepart    = fDepart
        self.fDepart_mu = fDepart_mu # returns the mean of the departure distribution at time t

        self.networking(net_size)

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
        pass


    def networking(self, network_size) :
        self.net_data = -1 * np.ones((network_size, 3))


    def initialize(self, add_arrival=True) :
        self.active = True
        if add_arrival :
            self._add_arrival()

    ## Needs updating
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
                new_arrival   = self.AgentClass(self.nArrivals+1, len(self.net_data) )
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


    def extract_information(self, agent) :
        if isinstance(agent, SmartAgent) :
            a = self.net_data[:, 0] < agent.net_data[:, 0]
            self.net_data[a, :] = agent.net_data[a, :]


    def append_departure(self, agent, t) :
        self.nSystem       += 1
        self.nArrivals     += 1
        agent.arr_ser[0]    = t

        self.extract_information(agent)

        if self.nSystem <= self.nServers :
            agent.arr_ser[1]    = t
            agent.set_departure(self.fDepart(t))
            heappush(self.departures, agent)
        else :
            self.queue.append(agent)


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            new_arrival   = heappop(self.arrivals)
            self.local_t  = new_arrival.time

            if self.active :
                self._add_arrival()

            self.append_departure(new_arrival, self.local_t)
            new_depart = None
                
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

            if self.nSystem == 0 : 
                self.networking( len(self.net_data) )

        if self.arrivals[0].time < self.departures[0].time :
            self.time = self.arrivals[0].time
        else :
            self.time = self.departures[0].time

        return new_depart


    def current_color(self, which='') :
        if which == 'edge' :
            nSy = self.nSystem
            cap = self.nServers
            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
            color[3] = 0.0
  
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
                color[3] = 0.5

        return color


    def reset(self) :
        self.nArrivals  = 0
        self.nSystem    = 0
        self.nTotal     = 0
        self.local_t    = 0
        self.time  = infty
        self.next_ct    = 0
        self.queue      = collections.deque()
        self.arrivals   = []
        self.departures = []
        inftyAgent      = Agent(0, 1)
        inftyAgent.time = infty

        heappush(self.arrivals, inftyAgent)
        heappush(self.departures, inftyAgent)

        self.networking( len(self.net_data) )


    def __deepcopy__(self, memo) :
        new_server              = QueueServer()
        new_server.issn         = copy.deepcopy(self.issn)
        new_server.nArrivals    = copy.deepcopy(self.nArrivals)
        new_server.nDeparts     = copy.deepcopy(self.nDeparts)
        new_server.nSystem      = copy.deepcopy(self.nSystem)
        new_server.nTotal       = copy.deepcopy(self.nTotal)
        new_server.nServers     = copy.deepcopy(self.nServers)
        new_server.local_t      = copy.deepcopy(self.local_t)
        new_server.time         = copy.deepcopy(self.time)
        new_server.next_ct      = copy.deepcopy(self.next_ct)
        new_server.queue        = copy.deepcopy(self.queue)
        new_server.arrivals     = copy.deepcopy(self.arrivals)
        new_server.departures   = copy.deepcopy(self.departures)
        new_server.net_data     = copy.deepcopy(self.net_data)
        new_server.AgentClass   = copy.deepcopy(self.AgentClass)
        return new_server



class LossQueue(QueueServer) :

    def __init__(self, nServers=1, issn=0, active=False, net_size=1, 
            fArrival=lambda x : x - log(uniform()) / 1, 
            fDepart =lambda x : x - log(uniform()) / 1.1, fDepart_mu=lambda x : 1/1.1,
            AgentClass=Agent, queue_cap=0) :

        QueueServer.__init__(self, nServers, issn, active, net_size, fArrival, fDepart, AgentClass=AgentClass)

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
                self.extract_information(new_arrival)

                heappush(self.departures, new_arrival)

                if self.arrivals[0].time < self.departures[0].time :
                    self.time = self.arrivals[0].time
                else :
                    self.time = self.departures[0].time

        elif self.departures[0].time < self.arrivals[0].time :
            return QueueServer.next_event(self)


    def reset(self) :
        QueueServer.reset(self)
        self.nBlocked  = 0


    def __deepcopy__(self, memo) :
        new_server              = LossQueue()
        new_server.issn         = copy.deepcopy(self.issn)
        new_server.nArrivals    = copy.deepcopy(self.nArrivals)
        new_server.nDeparts     = copy.deepcopy(self.nDeparts)
        new_server.nSystem      = copy.deepcopy(self.nSystem)
        new_server.nTotal       = copy.deepcopy(self.nTotal)
        new_server.nBlocked     = copy.deepcopy(self.nBlocked)
        new_server.nServers     = copy.deepcopy(self.nServers)
        new_server.local_t      = copy.deepcopy(self.local_t)
        new_server.time         = copy.deepcopy(self.time)
        new_server.next_ct      = copy.deepcopy(self.next_ct)
        new_server.queue        = copy.deepcopy(self.queue)
        new_server.arrivals     = copy.deepcopy(self.arrivals)
        new_server.departures   = copy.deepcopy(self.departures)
        new_server.net_data     = copy.deepcopy(self.net_data)
        new_server.AgentClass   = copy.deepcopy(self.AgentClass)
        return new_server



class MarkovianQueue(QueueServer) :

    def __init__(self, nServers=1, issn=0, active=False, net_size=1, aRate=1, dRate=1.1) :
        QueueServer.__init__(self, nServers, issn, active, net_size,
            lambda t : t - log(uniform()) / aRate,
            lambda t : t - log(uniform()) / dRate, lambda t : 1 / dRate, Agent) 

        self.rates  = [aRate, dRate]

    def __repr__(self) :
        tmp = "MarkovianQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s, rates: %s" \
            %  (self.issn[2], self.nServers, len(self.queue), self.nArrivals, 
                self.nDeparts, np.round(self.time, 3), self.rates)
        return tmp


    def change_rates(aRate=None, dRate=None) :
        if aRate != None :
            self.fArrival   = lambda t : t - log(uniform()) / aRate
        if dRate != None :    
            self.fDepart    = lambda t : t - log(uniform()) / dRate
            self.fDepart_mu = lambda t : 1 / aRate



class ResourceQueue(LossQueue) :

    def __init__(self, nServers=1, issn=0, active=False, net_size=1, max_servers=10) :
        LossQueue.__init__(self, nServers, issn, active, net_size, 
            fArrival=lambda t : t - log(uniform()), fDepart=lambda t : t, 
            fDepart_mu=lambda t : 0, AgentClass=ResourceAgent, queue_cap=0)

        self.colors = { 'edge_normal'   : [0.7, 0.7, 0.7, 0.50],
                        'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                        'vertex_pen'    : [0.0, 0.235, 0.718, 1.0] }
        self.max_servers  = max_servers
        self.over_max     = 0
        self.nBlocked     = 0


    def __repr__(self) :
        tmp = "ResourceQueue: %s. servers: %s, max servers: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.issn[2], self.nServers, self.max_servers, self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp

    def __str__(self) :
        return "ResourceQueue"


    def set_nServers(self, n) :
        self.nServers = n
        if n > self.max_servers :
            self.over_max += 1


    def next_event(self) :
        if isinstance(self.arrivals[0], ResourceAgent) :
            if self.arrivals[0].time < self.departures[0].time :
                if self.arrivals[0].has_resource :
                    new_arrival  = heappop(self.arrivals)
                    self.local_t = new_arrival.time
                    self.nTotal -= 1
                    self.set_nServers(self.nServers+1)

                    if self.arrivals[0].time < self.departures[0].time :
                        self.time = self.arrivals[0].time
                    else :
                        self.time = self.departures[0].time

                elif self.nSystem < self.nServers :
                    QueueServer.next_event(self)

                else :
                    self.nBlocked  += 1
                    self.nArrivals += 1
                    self.nTotal    -= 1
                    new_arrival     = heappop(self.arrivals)
                    self.local_t    = new_arrival.time
                    if self.arrivals[0].time < self.departures[0].time :
                        self.time = self.arrivals[0].time
                    else :
                        self.time = self.departures[0].time

            elif self.departures[0].time < self.arrivals[0].time :
                return QueueServer.next_event(self)
        else :
            return LossQueue.next_event(self)


    def current_color(self, which='') :
        if which == 'edge' :
            nSy = self.nServers
            cap = self.max_servers
            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
            color[3] = 0.0
  
        elif which == 'pen' :
            color = self.colors['vertex_pen']
        else :
            nSy = self.nServers
            cap = self.max_servers
            tmp = 0.9 - min(nSy / 5, 0.9) if cap <= 1 else 0.9 - min(nSy / (3 * cap), 0.9)

            if self.issn[0] == self.issn[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
                color[3] = 0.5

        return color


    def reset(self) :
        QueueServer.reset(self)
        self.nBlocked  = 0
        self.over_max = 0


