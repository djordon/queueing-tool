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
