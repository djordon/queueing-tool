from numpy.random           import uniform
from numpy                  import ones, zeros, infty, log
from heapq                  import heappush, heappop
from collections            import deque
from .. agents.queue_agents import Agent, Smart_Agent, Learning_Agent, Random_Agent

import numpy                as np
import copy



def arrival( rate, rate_max, t ) :
    t   = t - log( uniform() ) / rate_max
    while rate_max * uniform() > rate(t) :
        t   = t - log( uniform() ) / rate_max
    return t


def departure( rate, rate_max, t ) :
    t   = t - log( uniform() ) / rate_max
    while rate_max * uniform() > rate(t) :
        t   = t - log( uniform() ) / rate_max
    return t


def exponential_rv( rate, t ) :
    return t - log( uniform() ) / rate
     


class Queue_server :

    def __init__(self, nServers=1, issn=0, create=True, net_size=1,
            xArrival=lambda x: exponential_rv(1, x), 
            xDepart =lambda x: exponential_rv(1.1, x),
            xDepart_mu = lambda x : 1/1.1) :

        self.issn       = issn
        self.nServers   = nServers
        self.nArrivals  = 0
        self.nDeparts   = 0
        self.nSystem    = 0
        self.nTotal     = 0

        self.t          = 0
        self.CREATE     = create
        self.smart_p    = 0
        self.next_ct    = 0

        self.queue      = deque()
        self.arrivals   = []
        self.departures = []

        inftyAgent      = Agent(0, 1)
        inftyAgent.time = infty

        heappush(self.arrivals, inftyAgent)
        heappush(self.departures, inftyAgent )

        self.xArrival   = xArrival
        self.xDepart    = xDepart
        self.xDepart_mu = xDepart_mu # returns the mean of the departure distribution at time t

        self.networking(net_size)

    def __repr__(self) :
        tmp = "Queue_server. # servers: %s, queued: %s, arrivals: %s, departures: %s, local_time: %s" \
            %  (self.nServers, len(self.queue), self.nArrivals, \
                self.nArrivals - self.nSystem, self.t)
        return tmp

    def __lt__(a,b) :
        min_a   = min(a.arrivals[0].time, a.departures[0].time)
        min_b   = min(b.arrivals[0].time, b.departures[0].time)
        return min_a < min_b
    def __gt__(a,b) :
        min_a   = min(a.arrivals[0].time, a.departures[0].time)
        min_b   = min(b.arrivals[0].time, b.departures[0].time)
        return min_a > min_b
    def __eq__(a,b) :
        return (not a < b) and (not b < a)
    def __le__(a,b) :
        return not b < a
    def __ge__(a,b) :
        return not a < b


    def networking(self, network_size) :
        self.net_data   = -1 * ones( (network_size, 3) )


    ## Needs updating
    def set_nServers(self, n) :
        self.nServers   = n


    def nQueued(self) :
        if self.nServers == infty : n = 0
        else :    n = self.nSystem - self.nServers

        return max([n,0])


    def travel_stats(self) :
        ans = zeros(4)
        for agent in self.arrivals :
            if isinstance(agent, Smart_Agent) : ans[3]  += 1
            if agent != infty :
                ans[0] += agent.park_t[1]
                ans[1] += agent.trip_t[1]
                ans[2] += agent.trips
        for agent,j in self.departures :
            if isinstance(agent, Smart_Agent) : ans[3]  += 1
            if agent != infty :
                ans[0] += agent.park_t[1]
                ans[1] += agent.trip_t[1]
                ans[2] += agent.trips
        for agent in self.queue :
            if isinstance(agent, Smart_Agent) : ans[3]  += 1
            ans[0] += agent.park_t[1]
            ans[1] += agent.trip_t[1]
            ans[2] += agent.trips
        return ans


    def add_arrival(self, *args, **kwargs) :
        if len(args) != 0 :
            for a in args :
                self.nTotal    += 1
                heappush(self.arrivals, a)
        else : 
            if self.CREATE :
                if self.t >= self.next_ct :
                    self.nTotal    += 1
                    self.next_ct    = self.xArrival(self.t)
                    if uniform() < self.smart_p :
                        new_arrival = Smart_Agent( self.nArrivals+1, self.net_data.shape[0])
                    else :
                        new_arrival = Random_Agent(self.nArrivals+1, self.net_data.shape[0])
                    new_arrival.set_arrival( self.next_ct )
                    heappush(self.arrivals, new_arrival)


    def next_time(self) :
        return min(self.arrivals[0].time, self.departures[0].time)


    def next_event_type(self) :
        if self.arrivals[0].time < self.departures[0].time :
            return "arrival"
        elif self.arrivals[0].time > self.departures[0].time :
            return "departure"
        else :
            return "nothing"


    def extract_information(self, agent) :
        if isinstance(agent, Smart_Agent) :
            a   = self.net_data[:, 0] < agent.net_data[:, 0]
            self.net_data[a, :]    = agent.net_data[a, :]


    def append_departure(self, agent, t) :
        self.nSystem       += 1
        self.nArrivals     += 1
        agent.arr_ser[0]    = t

        self.extract_information(agent)

        if self.nSystem <= self.nServers :
            agent.arr_ser[1]    = t
            agent.set_departure( self.xDepart(t) )
            heappush(self.departures, agent)
        else :
            self.queue.append(agent)


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            new_arrival     = heappop(self.arrivals)
            self.t          = new_arrival.time

            self.add_arrival()
            self.append_departure(new_arrival, self.t)
                
        elif self.departures[0].time < infty :
            new_depart      = heappop(self.departures)
            self.t          = new_depart[1]
            self.nDeparts  += 1
            self.nTotal    -= 1
            self.nSystem   -= 1

            if len(self.queue) > 0 :
                agent               = self.queue.popleft()
                agent.arr_ser[1]    = self.t
                agent.set_departure( self.xDepart(self.t) )
                heappush(self.departures, agent)

            new_depart.update_information(self.net_data)
            new_depart.stamp(self.issn, self.nSystem, self.nServers, self.t)

            if self.nSystem == 0 : 
                self.networking( self.net_data.shape[0] )

            return new_depart


    def reset(self) :
        self.nArrivals  = 0
        self.nSystem    = 0
        self.nTotal     = 0
        self.t          = 0
        self.next_ct    = 0

        self.queue      = deque()
        self.arrivals   = []
        self.departures = []

        inftyAgent      = Agent(0, 1)
        inftyAgent.time = infty

        heappush(self.arrivals, inftyAgent)
        heappush(self.departures, inftyAgent )

        self.networking(self.net_data.shape[0])


    def __deepcopy__(self, memo) :
        new_server              = Queue_server()
        new_server.issn         = copy.deepcopy(self.issn)
        new_server.nArrivals    = copy.deepcopy(self.nArrivals)
        new_server.nDeparts     = copy.deepcopy(self.nDeparts)
        new_server.nSystem      = copy.deepcopy(self.nSystem)
        new_server.nTotal       = copy.deepcopy(self.nTotal)
        new_server.nServers     = copy.deepcopy(self.nServers)

        new_server.t            = copy.deepcopy(self.t)
        new_server.next_ct      = copy.deepcopy(self.next_ct)

        new_server.queue        = copy.deepcopy(self.queue)
        new_server.arrivals     = copy.deepcopy(self.arrivals)
        new_server.departures   = copy.deepcopy(self.departures)
        new_server.net_data     = copy.deepcopy(self.net_data)
        return new_server


class Loss_Queue( Queue_server ) :

    def __init__(self, nServers=1, issn=0, create=True, net_size=1, 
            xArrival=lambda x: exponential_rv(1,x), 
            xDepart =lambda x: exponential_rv(1,x), queue_cap=0) :

        if nServers == infty :
            print("Loss_Queue must have finite number of servers. Setting to 1")
            nServers = 1

        Queue_server.__init__(self, nServers+1, issn, create, net_size, xArrival, xDepart) 
        self.nServers   = nServers
        self.nLossed    = 0
        self.queue_cap  = queue_cap

    def __repr__(self) :
        tmp = "Loss_queue. # servers: %s, queued: %s, arrivals: %s, departures: %s, local_time: %s" \
            %  (self.nServers, len(self.queue), self.nArrivals, \
                self.nArrivals - self.nSystem, self.t)
        return tmp

    def set_nServers(self, n) :
        if n == infty :
            print("Loss_Queue must have finite number of servers. Setting to 1")
            n = 1

        Queue_server.set_nServers(self, n+1)
        self.nServers   = n


    def lossed(self) :
        return (self.nLossed / self.nArrivals) if self.nArrivals > 0 else 0


    def next_event(self) :
        event    = self.next_event_type()
        if event == "arrival" :
            if self.nSystem < self.nServers + self.queue_cap :
                self.arrivals[0].set_parked()

                Queue_server.next_event(self)
            else :
                self.nLossed   += 1
                self.nArrivals += 1
                self.nSystem   += 1
                new_arrival     = heappop(self.arrivals)
                new_arrival.add_loss(self.issn)

                self.t  = new_arrival.time
                self.add_arrival()
                new_arrival.arr_ser[0]  = self.t
                new_arrival.arr_ser[1]  = self.t
                self.extract_information(new_arrival)

                heappush(self.departures, new_arrival )

        elif event == "departure" :
            return Queue_server.next_event(self)


    def reset(self) :
        Queue_server.reset(self)
        self.nLossed    = 0


    def __deepcopy__(self, memo) :
        new_server              = Loss_Queue()
        new_server.issn         = copy.deepcopy(self.issn)
        new_server.nArrivals    = copy.deepcopy(self.nArrivals)
        new_server.nDeparts     = copy.deepcopy(self.nDeparts)
        new_server.nSystem      = copy.deepcopy(self.nSystem)
        new_server.nTotal       = copy.deepcopy(self.nTotal)
        new_server.nLossed      = copy.deepcopy(self.nTotal)
        new_server.nServers     = copy.deepcopy(self.nServers)

        new_server.t            = copy.deepcopy(self.t)
        new_server.next_ct      = copy.deepcopy(self.next_ct)

        new_server.queue        = copy.deepcopy(self.queue)
        new_server.arrivals     = copy.deepcopy(self.arrivals)
        new_server.departures   = copy.deepcopy(self.departures)
        new_server.net_data     = copy.deepcopy(self.net_data)
        return new_server

