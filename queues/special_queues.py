from .queues_agents import Agent, QueueServer, LossQueue
from numpy.random   import randint, exponential
from heapq          import heappush, heappop

import numpy as np
import copy


class ResourceAgent(Agent) :
    def __init__(self, issn) :
        Agent.__init__(self, issn)
        self.has_resource = False
        self.had_resource = False

    def __repr__(self) :
        return "ResourceAgent. issn: %s, time: %s" % (self.issn, self.time)


    def desired_destination(self, *info) :
        network, qedge = info[:2]

        if self.had_resource :
            z = qedge[2]
        else :
            n = len( network.adjacency[qedge[1]] )
            d = randint(0, n)
            z = network.adjacency[qedge[1]][d]
        return z


    def queue_action(self, queue, *args, **kwargs) :
        nServers = queue.nServers
        if isinstance(queue, ResourceQueue) :
            if self.has_resource :
                self.has_resource = False
                self.had_resource = True
            else :
                if queue.nServers > 0 :
                    queue.set_nServers(nServers - 1)
                    self.has_resource = True


    def __deepcopy__(self, memo) :
        new_agent               = Agent.__deepcopy__(self, memo)
        new_agent.has_resource  = copy.deepcopy(self.has_resource)
        new_agent.had_resource  = copy.deepcopy(self.had_resource)
        return new_agent



class ResourceQueue(LossQueue) :

    def __init__(self, nServers=10, edge=(0,0,0), fArrival=lambda x : x + exponential(1),
                    fDepart=lambda x : x, AgentClass=ResourceAgent, qbuffer=0) :
        LossQueue.__init__(self, nServers, edge, fArrival, fDepart, AgentClass, qbuffer)

        self.colors = { 'edge_normal'   : [0.7, 0.7, 0.7, 0.50],
                        'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                        'vertex_pen'    : [0.0, 0.235, 0.718, 1.0] }

        self.max_servers  = nServers
        self.over_max     = 0


    def __repr__(self) :
        tmp = "ResourceQueue: %s. servers: %s, max servers: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, self.max_servers, self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp


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
                    self.nBlocked      += 1
                    self.nArrivals[0]  += 1
                    self.nTotal        -= 1
                    new_arrival         = heappop(self.arrivals)
                    self.local_t        = new_arrival.time
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

            if self.edge[0] == self.edge[1] :
                color    = [ i * tmp / 0.9 for i in self.colors['vertex_normal'] ]
                color[3] = 1.0
            else :
                color    = [ i * tmp / 0.9 for i in self.colors['edge_normal'] ]
                color[3] = 0.5

        return color


    def clear(self) :
        QueueServer.clear(self)
        self.nBlocked  = 0
        self.over_max  = 0


    def __deepcopy__(self, memo) :
        new_server              = LossQueue.__deepcopy__(self, memo)
        new_server.max_servers  = copy.copy(self.max_servers)
        new_server.over_max     = copy.copy(self.over_max)
        return new_server



class InfoAgent(Agent) :

    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)

        self.stats    = np.zeros((net_size, 3), np.int32 )
        self.net_data = np.ones((net_size, 3)) * -1

    def __repr__(self) :
        return "InfoAgent. issn: %s, time: %s" % (self.issn, self.time)


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


    def desired_destination(self, *info) :
        network, qedge = info[:2]
        if self.dest != None and qedge[1] == self.dest :
            self.old_dest   = self.dest
            self.dest       = None
            self.trip_t[1] += network.t - self.trip_t[0] 
            self.trips     += 1
            self._set_dest(net = network)

        elif self.dest == None :
            self.trip_t[0]  = network.t
            self._set_dest(net = network)
            while self.dest == qedge[1] :
                self._set_dest(net = network)
        
        z   = network.shortest_path[qedge[1], self.dest]
        z   = network.g.edge(qedge[1], z)
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
            self.net_data[n, :] = queue.local_t, queue.nServers, queue.nSystem / queue.nServers


    def __deepcopy__(self, memo) :
        new_agent          = Agent.__deepcopy__(self, memo)
        new_agent.stats    = copy.deepcopy(self.stats)
        new_agent.net_data = copy.deepcopy(self.net_data)
        return new_agent



class InfoQueue(LossQueue) :

    def __init__(self, nServers=1, edge=(0,0,0), net_size=1,
            fArrival=lambda x : x + exponential(1), fDepart =lambda x : x + exponential(0.95),
            AgentClass=InfoAgent, qbuffer=np.infty) :
        LossQueue.__init__(self, nServers, edge, fArrival, fDepart, fDepart_mu, AgentClass, qbuffer)

        self.colors = {'edge_normal'   : [0.9, 0.9, 0.9, 0.5],
                       'vertex_normal' : [1.0, 1.0, 1.0, 1.0],
                       'vertex_pen'    : [0.0, 0.5, 1.0, 1.0]}

        self.networking(net_size)

    def __repr__(self) :
        tmp = "InfoQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, len(self.queue), self.nArrivals, self.nDeparts, np.round(self.time, 3))
        return tmp

    def __repr__(self) :
        tmp = "InfoQueue: %s. servers: %s, max servers: %s, arrivals: %s, departures: %s, next time: %s" \
            %  (self.edge[2], self.nServers, self.max_servers, self.nArrivals, self.nDeparts, np.round(self.time, 3))
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
            heappush(self.arrivals, args[0])
        else : 
            if self.local_t >= self.next_ct :
                self.nTotal  += 1
                self.next_ct  = self.fArrival(self.local_t)
                new_arrival   = self.AgentClass(self.edge, len(self.net_data) )
                new_arrival.set_arrival( self.next_ct )
                heappush(self.arrivals, new_arrival)

                self.nArrivals[1] += 1
                if self.nArrivals[1] >= self.cap :
                    self.active = False

        if self.arrivals[0].time < self.departures[0].time :
            self.time = self.arrivals[0].time
        else :
            self.time = self.departures[0].time


    def append_departure(self, agent, t) :
        self.extract_information(agent)
        LossQueue.append_departure(self, agent, t)


    def next_event(self) :
        if self.arrivals[0].time < self.departures[0].time :
            self.extract_information(self.arrivals[0])

        LossQueue.next_event(self)


    def clear(self) :
        LossQueue.clear(self)
        self.networking( len(self.net_data) )


    def __deepcopy__(self, memo) :
        new_server          = LossQueue.__deepcopy__(self, memo)
        new_server.net_data = copy.copy(self.net_data)
        return new_server



class MarkovianQueue(QueueServer) :

    def __init__(self, nServers=1, edge=(0,0,0), aRate=1, dRate=1.1, AgentClass=Agent) :
        aMean = 1 / aRate
        dMean = 1 / dRate
        QueueServer.__init__(self, nServers, edge, lambda x : x + exponential(aMean),
            lambda x : x + exponential(dMean), AgentClass)

        self.rates  = [aRate, dRate]

    def __repr__(self) :
        tmp = "MarkovianQueue: %s. servers: %s, queued: %s, arrivals: %s, departures: %s, next time: %s, rates: %s" \
            %  (self.edge[2], self.nServers, len(self.queue), self.nArrivals,
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


    def __deepcopy__(self, memo) :
        new_server        = QueueServer.__deepcopy__(self, memo)
        new_server.rates  = copy.copy(self.rates)
        return new_server
