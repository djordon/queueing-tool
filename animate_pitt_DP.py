import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import adp
import copy
import datetime
import cProfile
import pickle

np.set_printoptions(precision=3,suppress=True,threshold=2000)
directory   = './'
g   = gt.load_graph('pitt_network.xml', fmt='xml')
pit = qt.QueueNetwork( g )
a   = adp.approximate_dynamic_program(pit.g)
vs  = [a.Qn.g.vertex(202), a.Qn.g.vertex(453), a.Qn.g.vertex(255),
       a.Qn.g.vertex(449), a.Qn.g.vertex(218), a.Qn.g.vertex(72),
       a.Qn.g.vertex(126), a.Qn.g.vertex(326)]

for v in vs :
    for e in v.in_edges() :
        a.Qn.g.ep['queues'][e].active       = True
        a.Qn.g.ep['queues'][e].active_p     = 0
        a.Qn.g.ep['queues'][e].xArrival     = lambda x : qt.exponential_rv( 2, x )
        a.Qn.g.ep['queues'][e].xDepart      = lambda x : qt.exponential_rv( 3, x )
        a.Qn.g.ep['queues'][e].xDepart_mu   = lambda x : 1/3
        a.Qn.g.ep['queues'][e].add_arrival()
        break

a.Qn.agent_cap  = 1000
a.agent_cap     = 1000

nLearners   = 3
dest        = list( np.random.choice(a.Qn.g.gp['node_index']['destination'], nLearners) )
orig        = list( np.random.choice(a.Qn.g.gp['node_index']['road'], nLearners) )
print((orig, dest))
a.approximate_policy_iteration(orig, dest, save_frames=False)

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

#loc = "/home/dan/math/code/python/queueing/data/"
#dat = datetime.datetime.today().isoformat().replace('T', '_')
#pickle.dump(a.agent_variables, open(loc+"theta_data_"+dat+".p", "wb") )


"""
import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import cProfile
g   = gt.load_graph('pitt_network.xml', fmt='xml')
pit = qt.QueueNetwork(g, seed=10)
pit.initialize(nActive=5)
pit.agent_cap = 2000
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
g   = gt.load_graph('pitt_network.xml', fmt='xml')
pit = qt.QueueNetwork(nVertices=50, seed=10)
pit.initialize(nActive=5)
pit.agent_cap = 1000
pit.simulate(50)
pit.draw()


import numpy            as np
import graph_tool.all   as gt
import queueing_tool    as qt
import cProfile
g   = gt.load_graph('pitt_network.xml', fmt='xml')
pit = qt.QueueNetwork(nVertices=50, seed=12)
pit.initialize(nActive=5)
pit.agent_cap = 1000
pr  = cProfile.Profile()
pr.enable()
pit.simulate(90)
pr.disable()
pr.print_stats(sort='time')
