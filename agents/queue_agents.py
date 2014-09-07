from numpy.random   import uniform, randint, choice
from numpy          import ones, zeros, logical_or, argmax

import numpy        as np
import copy


class Agent :

    def __init__(self, issn, net_size) :
        self.issn       = issn
        self.time       = 0                                     # agents arrival or departure time
        self.dest       = None
        self.old_dest   = None
        self.parking    = False
        self.parked     = False
        self.trips      = 0
        self.park_t     = [0, 0]
        self.trip_t     = [0, 0]
        self.type       = 0
        self.garage_tmp = None
        self.arr_ser    = [0, 0]
        self.od         = [0, 0]

        self.stats      = zeros( (net_size, 3), np.int32 )
        self.net_data   = zeros( (net_size, 3), np.int32 )


    def __lt__(a,b) :
        return a.time < b.time
    def __gt__(a,b) :
        return a.time > b.time
    def __eq__(a,b) :
        return (not a < b) and (not b < a)
    def __le__(a,b) :
        return not b < a
    def __ge__(a,b) :
        return not a < b

    def __repr__(self) :
        return "Agent. issn: %s, arr. or dep. time: %s" % (self.issn, self.time)

    def set_arrival(self, t) :
        self.time    = t

    def set_departure(self, t) :
        self.time    = t

    def __getitem__(self, index) :
        return self.time


    def add_loss(self, server_name) :
        if not isinstance( server_name, int ) :
            n   = server_name[2]    # This is the edge_index of the server
        else :
            n   = server_name
        self.stats[n, 2] += 1 


    def set_type(self, n) :
        self.type   = n


    def stamp(self, server_name, nSystem, cap, depart_t) :
        if not isinstance( server_name, int ) :
            n   = server_name[2]    # This is the edge_index of the server
        else :
            n   = server_name
        self.stats[n, 0]    = self.stats[n, 0] + (self.arr_ser[1] - self.arr_ser[0])
        self.stats[n, 1]   += 1 if (self.arr_ser[1] - self.arr_ser[0]) > 0 else 0
        if self.parking and nSystem == cap and self.garage_tmp != None:
            i   = argmax( self.garage_tmp[0,:] == n )
            self.garage_tmp[1,i] = 0


    def set_dest(self, net=None, dest=None) :
        if dest != None :
            self.dest = int(dest)
        else :
            nodes   = net.g.gp['node_index']['dest_road']
            dLen    = net.g.gp['dest_count']
            rLen    = net.nV - dLen - net.g.gp['garage_count']
            probs   = [0.3 / dLen for k in range(dLen)]
            probs.extend( [0.7/rLen for k in range(rLen)] )
            dest    = int( choice(nodes, size=1, p=probs) )

            if self.old_dest != None :
                while dest == int( self.old_dest ) :
                    dest    =   int( choice(nodes, size=1, p=probs) )
            self.dest   = dest


    def set_parked(self) :
        self.parked     = True
        self.parking    = False
        self.park_t[1] += self.time - self.park_t[0]
        self.garage_tmp = None


    def desired_destination(self, *info) :
        network = info[0]
        e       = info[1]
        if self.dest != None and int(e.target()) == self.dest :
            self.old_dest   = self.dest
            self.dest       = None
            self.park_t[0]  = network.t
            self.trip_t[1] += network.t - self.trip_t[0] 
            self.parking    = True
            self.trips     += 1
            self.set_dest(net = network)

        elif self.dest == None :
            self.trip_t[0]  = network.t
            self.set_dest(net = network)
            while self.dest == int(e.target()) :
                self.set_dest(net = network)
        
        z   = network.g.vp['shortest_path'][e.target()][self.dest]
        z   = network.g.edge(e.target(), z)
        return z


    def update_information(self, data) :
        pass


    def get_beliefs(self) :
        pass


    def __deepcopy__(self, memo) :
        new_agent               = self.__class__(self.issn, self.net_data.shape[0])
        new_agent.issn          = copy.deepcopy(self.issn)
        new_agent.time          = copy.deepcopy(self.time)
        new_agent.stats         = copy.deepcopy(self.stats)
        new_agent.dest          = copy.deepcopy(self.dest)
        new_agent.old_dest      = copy.deepcopy(self.old_dest)
        new_agent.parking       = copy.deepcopy(self.parking)
        new_agent.parked        = copy.deepcopy(self.parked)
        new_agent.trips         = copy.deepcopy(self.trips)
        new_agent.park_t        = copy.deepcopy(self.park_t)
        new_agent.trip_t        = copy.deepcopy(self.trip_t)
        new_agent.type          = copy.deepcopy(self.type)
        new_agent.garage_tmp    = copy.deepcopy(self.garage_tmp)
        new_agent.net_data      = copy.deepcopy(self.net_data)
        new_agent.arr_ser       = copy.deepcopy(self.arr_ser)
        new_agent.od            = copy.deepcopy(self.od)
        return new_agent


class Learning_Agent( Agent ) :

    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)
        self.net_data   = -1 * ones( (net_size, 3) )

    def __repr__(self) :
        return "Learning_Agent. issn: %s, arr. or dep. time: %s" % (self.issn, self.time)

    def stamp(self, server_name, nSystem, cap, depart_t) :
        if not isinstance( server_name, int ) :
            n   = server_name[2]    # This is the edge_index of the server
        else :
            n   = server_name
        self.stats[n, 0]    = self.stats[n, 0] + (self.arr_ser[1] - self.arr_ser[0])
        self.stats[n, 1]   += 1 if (self.arr_ser[1] - self.arr_ser[0]) > 0 else 0
        self.net_data[n, :] = depart_t, cap, nSystem/cap


    def update_information(self, net_data) :
        a = logical_or(self.net_data[:, 0] < net_data[:, 0], self.net_data[:, 0] == -1 )
        self.net_data[a, :]    = net_data[a, :]


    def get_beliefs(self) :
        return self.net_data[:, 2]


    def desired_destination(self, *info) :
        network = info[0]
        e       = info[1]
        return network.g.edge(e.target(), self.dest)




class Smart_Agent( Agent ) :

    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)
        self.net_data = -1 * ones( (net_size, 3) )

    def __repr__(self) :
        return "Smart_Agent. issn: %s, arr. or dep. time: %s" % (self.issn, self.time)

    def stamp(self, server_name, nSystem, cap, depart_t) :
        if not isinstance( server_name, int ) :
            n   = server_name[2]    # This is the edge_index of the server
        else :
            n   = server_name
        self.stats[n, 0]    = self.stats[n, 0] + (self.arr_ser[1] - self.arr_ser[0])
        self.stats[n, 1]   += 1 if (self.arr_ser[1] - self.arr_ser[0]) > 0 else 0
        self.net_data[n, :] = depart_t, cap, nSystem/cap


    def update_information(self, net_data) :
        a = logical_or(self.net_data[:, 0] < net_data[:, 0], self.net_data[:, 0] == -1 )
        self.net_data[a, :]    = net_data[a, :]


    def get_beliefs(self) :
        return self.net_data[:, 2]




class Random_Agent( Agent ) :

    def __init__(self, issn, net_size) :
        Agent.__init__(self, issn, net_size)
        self.net_data = -1 * ones( (net_size, 3) )

    def __repr__(self) :
        return "Random_Agent. issn: %s, arr. or dep. time: %s" % (self.issn, self.time)

    def stamp(self, server_name, nSystem, cap, depart_t) :
        if not isinstance( server_name, int ) :
            n   = server_name[2]    # This is the edge_index of the server
        else :
            n   = server_name
        self.stats[n, 0]    = self.stats[n, 0] + (self.arr_ser[1] - self.arr_ser[0])
        self.stats[n, 1]   += 1 if (self.arr_ser[1] - self.arr_ser[0]) > 0 else 0
        self.net_data[n, :] = depart_t, cap, nSystem/cap


    def update_information(self, net_data) :
        a = logical_or(self.net_data[:, 0] < net_data[:, 0], self.net_data[:, 0] == -1 )
        self.net_data[a, :]    = net_data[a, :]


    def get_beliefs(self) :
        return self.net_data[:, 2]


    def desired_destination(self, *info) :
        network = info[0]
        e       = info[1]
        n       = e.target().out_degree()
        d       = randint(0, n)
        ct      = 0
        for eo in e.target().out_edges() :
            if ct == d :
                z   = eo
                break
            ct += 1

        return z

