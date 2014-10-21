from numpy.random   import uniform, randint, choice
from numpy          import logical_or

import numpy as np
import copy


np.set_printoptions(precision=3, suppress=True, threshold=2000)

class Agent :

    def __init__(self, issn, net_size) :
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

        self.stats    = np.zeros((net_size, 3), np.int32 )
        self.net_data = np.ones((net_size, 3)) * -1

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


    def add_loss(self, qissn, *args, **kwargs) :
        # This qissn[2] is the edge_index of the server
        self.stats[qissn[2], 2] += 1 


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
        new_agent             = self.__class__(self.issn, self.net_data.shape[0])
        new_agent.issn        = copy.deepcopy(self.issn)
        new_agent.time        = copy.deepcopy(self.time)
        new_agent.stats       = copy.deepcopy(self.stats)
        new_agent.dest        = copy.deepcopy(self.dest)
        new_agent.old_dest    = copy.deepcopy(self.old_dest)
        new_agent.resting     = copy.deepcopy(self.resting)
        new_agent.trips       = copy.deepcopy(self.trips)
        new_agent.rest_t      = copy.deepcopy(self.rest_t)
        new_agent.trip_t      = copy.deepcopy(self.trip_t)
        new_agent.type        = copy.deepcopy(self.type)
        new_agent.net_data    = copy.deepcopy(self.net_data)
        new_agent.arr_ser     = copy.deepcopy(self.arr_ser)
        new_agent.od          = copy.deepcopy(self.od)
        return new_agent



class SmartAgent(Agent) :

    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)

    def __repr__(self) :
        return "SmartAgent. issn: %s, time: %s" % (self.issn, self.time)


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
        network, qissn = info[:2]
        if self.dest != None and qissn[1] == self.dest :
            self.old_dest   = self.dest
            self.dest       = None
            self.rest_t[0]  = network.t
            self.trip_t[1] += network.t - self.trip_t[0] 
            self.resting    = True
            self.trips     += 1
            self._set_dest(net = network)

        elif self.dest == None :
            self.trip_t[0]  = network.t
            self._set_dest(net = network)
            while self.dest == qissn[1] : #int(e.target()) :
                self._set_dest(net = network)
        
        z   = network.shortest_path[qissn[1], self.dest]
        z   = network.g.edge(qissn[1], z)
        return z


    def queue_action(self, queue, *args, **kwargs) :
        ### update information
        a = logical_or(self.net_data[:, 0] < queue.net_data[:, 0], self.net_data[:, 0] == -1)
        self.net_data[a, :] = queue.net_data[a, :]

        ### stamp this information
        n   = queue.issn[2]    # This is the edge_index of the server
        self.stats[n, 0]    = self.stats[n, 0] + (self.arr_ser[1] - self.arr_ser[0])
        self.stats[n, 1]   += 1 if (self.arr_ser[1] - self.arr_ser[0]) > 0 else 0
        self.net_data[n, :] = queue.local_t, queue.nServers, queue.nSystem / queue.nServers



class LearningAgent(SmartAgent) :

    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)

    def __repr__(self) :
        return "LearningAgent. issn: %s, time: %s" % (self.issn, self.time)


    def get_beliefs(self) :
        return self.net_data[:, 2]


    def desired_destination(self, *info) :
        network, queue = info[:2]
        return network.g.edge(queue[1], self.dest)


    def set_dest(self, net=None, dest=None) :
        self.dest = int(dest)


class ResourceAgent(Agent) :
    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)
        self.has_resource = False
        self.had_resource = False

    def __repr__(self) :
        return "ResourceAgent. issn: %s, time: %s" % (self.issn, self.time)


    def desired_destination(self, *info) :
        network, qissn = info[:2]
        v = network.g.vertex(qissn[1])

        if self.had_resource :
            z = network.g.edge(qissn[0], qissn[1])
        else :
            #n = v.out_degree()
            v = network.g.vertex(qissn[1])
            d = randint(0, v.out_degree())
            z = list(v.out_edges())[d]

        return z


    def set_arrival(self, t) :
        self.time = t


    def set_departure(self, t) :
        self.time = t


    def queue_action(self, queue, *args, **kwargs) :
        nServers = queue.nServers
        if str(queue) == "ResourceQueue" :
            if self.has_resource :
                self.has_resource = False
                self.had_resource = True
            else :
                if queue.nServers > 0 :
                    queue.set_nServers(nServers - 1)
                    self.has_resource = True


