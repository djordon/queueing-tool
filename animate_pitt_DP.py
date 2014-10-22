from numpy          import log
from numpy.random   import uniform
import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import adp_v2           as adp
import copy
import datetime
import cProfile
import pickle

np.set_printoptions(precision=3,suppress=True,threshold=2000)
g   = gt.load_graph('pitt_network.xml', fmt='xml')
pit = qt.QueueNetwork( g )
a   = adp.approximate_dynamic_program(pit.g)
vs  = [a.Qn.g.vertex(202), a.Qn.g.vertex(453), a.Qn.g.vertex(255),
       a.Qn.g.vertex(449), a.Qn.g.vertex(218), a.Qn.g.vertex(72),
       a.Qn.g.vertex(126), a.Qn.g.vertex(326)]

vs  = [202, 453, 255, 449, 218, 72, 126, 326]


#a.Qn.initialize(queues=vs)
a.Qn.agent_cap  = 1000
a.agent_cap     = 1000


node_dict = {'fcq' : [], 'des' : [], 'arc' : []}
for v in a.Qn.g.vertices() :
    if a.Qn.g.vp['vType'][v] == 1 :
        node_dict['fcq'].append(int(v))
    elif a.Qn.g.vp['vType'][v] == 2 :
        node_dict['des'].append(int(v))
    else :
        node_dict['arc'].append(int(v))

nLearners   = 1
dest        = list( np.random.choice(node_dict['des'], nLearners) )
orig        = list( np.random.choice(node_dict['arc'], nLearners) )
print((orig, dest))
a.approximate_policy_iteration(orig, dest, save_frames=True, verbose=True)

"""
orig, dest  = [], []
dist        = copy.deepcopy( a.dist )

for p in range( dist.shape[0] ) :
    pairs = np.argwhere(dist == dist.max()-p )
    for i,j in pairs :
        if a.Qn.g.vp['destination'].a[j] :
            orig.append(int(i))
            dest.append(int(j))
            #dist[int(i),int(j)] = 0
            break
    if len(orig) == nLearners :
        break

print(orig)
#print( (a.Qn.g.gp['node_index']['destination'], a.Qn.g.gp['node_index']['garage']) )

#orig, dest  = [344], [21]
a.approximate_policy_iteration(orig, dest, save_frames=True)
"""

"""

        node_dict = {'fcq' : [], 'des' : [], 'arc' : [], 'dest_arc' : []}
        for v in g.vertices() :
            if g.vp['vType'][v] == 1 :
                node_dict['fcq'].append(int(v))
            elif g.vp['vType'][v] == 2 :
                node_dict['des'].append(int(v))
            else :
                node_dict['arc'].append(int(v))
        node_dict['dest_arc'] = copy.copy(node_dict['des'])
        node_dict['dest_arc'].extend(node_dict['arc'])

### QueueServer code

    def travel_stats(self) :
        ans = np.zeros(4)
        for agent in self.arrivals :
            if isinstance(agent, SmartAgent) : 
                ans[3]  += 1
            if agent.time != infty :
                ans[0] += agent.rest_t[1]
                ans[1] += agent.trip_t[1]
                ans[2] += agent.trips
        for agent in self.departures :
            if isinstance(agent, SmartAgent) : 
                ans[3]  += 1
            if agent.time != infty :
                ans[0] += agent.rest_t[1]
                ans[1] += agent.trip_t[1]
                ans[2] += agent.trips
        for agent in self.queue :
            if isinstance(agent, SmartAgent) : 
                ans[3]  += 1
            ans[0] += agent.rest_t[1]
            ans[1] += agent.trip_t[1]
            ans[2] += agent.trips
        return ans
"""


"""
## QueueNetwork code
    def agent_stats(self) :
        ans     = np.zeros(7)
        rested  = 0
        spaces  = 0
        for e in self.g.edges() :
            q        = self.g.ep['queues'][e]
            ans[:4] += q.travel_stats()
            if isinstance(q, qs.LossQueue) :
                ans[4] += q.lossed()
                rested += q.nSystem
                spaces += q.nServers
        ans[5] = self.fcq_count
        ans[6] = rested/spaces
        return ans


    def information_stats(self) :
        real  = np.zeros(self.nE)
        data  = np.zeros(self.nE)
        a     = np.arange(self.nE)

        for e in self.g.edges() :
            q   = self.g.ep['queues'][e]
            k   = self.g.edge_index[e]
            tmp = q.net_data[:,2]
            ind = np.logical_and(a!=k, tmp != -1)
            real[q.issn[2]] = q.nSystem / q.nServers    # q.issn[2] is the edge_index for that edge
            data[ind]      += (tmp[ind] - real[ind])**2

        g_index = np.array(self.node_index['fcq'])
        d_index = np.array(self.node_index['destination'])
        r_index = np.array(self.node_index['arc'])
        data   /= (self.nE - 1)

        return np.array( ( (np.mean(data[r_index]), np.std(data[r_index])), \
                        (np.mean(data[d_index]), np.std(data[d_index])), \
                        (np.mean(data[g_index]), np.std(data[g_index])), \
                        (np.mean(data),          np.std(data)) ) )


    def _information_stats(self) :
        g   = self.node_index['fcq'][0]
        d   = self.node_index['des'][0]
        r   = self.node_index['arc'][1]

        ans = np.zeros( (3,2) )
        ct  = 0

        for q in g,d,r :
            data  = infty * np.ones(self.nE)
            a     = np.arange(self.nE)
            a     = a[a!=q]
            qnew  = self.g.ep['queues'][self.edges[q]]
            real  = qnew.nSystem / qnew.nServers

            for k in a :
                tmpdata = self.g.ep['queues'][self.edges[k]].net_data[q, 2]
                if tmpdata == -1 : 
                    continue
                data[k] = (tmpdata - real)
         
            data       = data[ data != infty ]
            ans[ct, :] = np.mean( data ), np.std( data )
            ct += 1

        return ans
"""

#loc = "/home/dan/math/code/python/queueing/data/"
#dat = datetime.datetime.today().isoformat().replace('T', '_')
#pickle.dump(a.agent_variables, open(loc+"theta_data_"+dat+".p", "wb") )


"""
import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import cProfile
import os
data_dir  = os.path.expanduser('~') + '/math/data/graphs/'
g   = gt.load_graph(data_dir+'pitt_network.xml', fmt='xml')
pit = qt.QueueNetwork(g, seed=10)
pit.initialize(nActive=5)
pit.agent_cap = 1000
pr  = cProfile.Profile()
pr.enable()
pit.simulate(90)
pr.disable()
pr.print_stats(sort='time')

cProfile.run('a.Qn.simulate()')
"""




import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import cProfile
pit = qt.QueueNetwork(nVertices=50, seed=12)
pit.initialize(nActive=50)
pit.agent_cap = 10000
pit.simulate(150)
pit.draw()


import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import cProfile
pit = qt.QueueNetwork(nVertices=50, seed=12)
pit.initialize(nActive=50)
pit.agent_cap = 10000
pr  = cProfile.Profile()
pr.enable()
pit.simulate(50)
pr.disable()
pr.print_stats(sort='time')



import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import cProfile
pit = qt.QueueNetwork(nVertices=500, seed=13)
pit.initialize(nActive=200)
pit.agent_cap = 2000
pit.simulate(40)
%timeit -n3 pit.simulate(25)
pr  = cProfile.Profile()
pr.enable()
pit.simulate(25)
pr.disable()
pr.print_stats(sort='cumtime')
