import numpy            as np
import graph_tool.all   as gt
import queue_tool       as qt
import queue_server     as qs
import adp
import copy
import datetime
import cProfile
import pickle

np.set_printoptions(precision=2,suppress=True,threshold=2000)

directory   = '/home/dan/math/code/python/queueing/'
pit = pickle.load(open(directory+'pitt_network.p', 'rb') )
#Qn  = qt.Queue_network( pit.g )
#g   = gt.load_graph('pitt_network.xml')
#Qn  = qt.Queue_network( g )
a   = adp.approximate_dynamic_program(pit.g)
vs  = [a.Qn.g.vertex(202), a.Qn.g.vertex(453), a.Qn.g.vertex(255),
       a.Qn.g.vertex(449), a.Qn.g.vertex(218), a.Qn.g.vertex(72),
       a.Qn.g.vertex(126), a.Qn.g.vertex(326)]

for v in vs :
    for e in v.in_edges() :
        a.Qn.g.ep['queues'][e].CREATE       = True
        a.Qn.g.ep['queues'][e].create_p     = 0
        a.Qn.g.ep['queues'][e].xArrival     = lambda x : qs.exponential_rv( 2, x )
        a.Qn.g.ep['queues'][e].xDepart      = lambda x : qs.exponential_rv( 3, x )
        a.Qn.g.ep['queues'][e].xDepart_mu   = lambda x : 1/3
        a.Qn.g.ep['queues'][e].add_arrival()
        break

a.Qn.agent_cap  = 1000
a.agent_cap     = 1000

nLearners   = 3
dest        = list( np.random.choice(a.Qn.g.gp['node_index']['destination'], nLearners) )
orig        = list( np.random.choice(a.Qn.g.gp['node_index']['road'], nLearners) )

a.approximate_policy_iteration(orig, dest, save_frames=True)

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
import queue_tool       as qt
import queue_server     as qs
agent       = qs.Learning_Agent(5, 10)
learning_agents = {k : qs.Learning_Agent(k, 10) for k in range(4)}
learning_agents[agent.issn] = agent

n=10
open_slot  = [-k for k in range(n)]
open_slot  = [open_slot[-k] for k in range(1,n+1)]
b          = [-k for k in range(n-1,-1,-1)]


import approximate_DP   as adp
import numpy            as np
import graph_tool.all   as gt
import queue_server     as qs
import queue_tool       as qt
import cProfile
import pickle
net = pickle.load( open('pitt_network.p', 'rb') )
que = qt.Queue_network( net.g )
#que.draw()
a   = adp.approximate_dynamic_program(seed=10)
pr  = cProfile.Profile()
pr.enable()
a.Qn.simulate(20)
pr.disable()
pr.print_stats(sort='time')

cProfile.run('a.Qn.simulate()')
"""

"""
v1  = Qn.g.vertex(202)
v2  = Qn.g.vertex(453)

for v in v1,v2 :
    for e in v.in_edges() :
        Qn.g.ep['queues'][e].CREATE       = True
        Qn.g.ep['queues'][e].create_p     = 0
        Qn.g.ep['queues'][e].xArrival     = lambda x : qs.exponential_rv( 8, x )
        Qn.g.ep['queues'][e].xDepart      = lambda x : qs.exponential_rv( 3, x )
        Qn.g.ep['queues'][e].xDepart_mu   = lambda x : 1/3
        Qn.g.ep['queues'][e].add_arrival()
        break

"""
