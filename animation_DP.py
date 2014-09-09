import numpy            as np
import graph_tool.all   as gt
import datetime
import cProfile
import pickle
import adp

a = adp.approximate_dynamic_program()

nLearners   = 1
break_now   = False
orig, dest  = [], []

for p in range( int(a.dist.max()-2) ) :
    pairs = np.argwhere(a.dist == a.dist.max()-p)
    for i,j in pairs :
        if a.Qn.g.vp['destination'].a[j] :
            break_now = True
            orig.append(int(i))
            dest.append(int(j))
            break
    if break_now and len(orig) == nLearners :
        break

#print( (a.Qn.g.gp['node_index']['destination'], a.Qn.g.gp['node_index']['garage']) )

#i,j = int(i), int(j)
a.approximate_policy_iteration(orig, dest, save_frames=True)

loc = "./data/"
dat = datetime.datetime.today().isoformat().replace('T', '_')
pickle.dump( (a.theta_history, a.value_history), open(loc+"theta_data_"+dat+".p", "wb") )


"""
import queue_tool    as qn
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
import queue_tool    as qn
import cProfile
import pickle
net = pickle.load( open('pitt_network.p', 'rb') )
que = qn.Queue_network( net.g )
que.draw()
a   = adp.approximate_dynamic_program(seed=10)
pr  = cProfile.Profile()
pr.enable()
a.Qn.simulate(20)
pr.disable()
pr.print_stats(sort='time')

cProfile.run('a.Qn.simulate()')
"""


