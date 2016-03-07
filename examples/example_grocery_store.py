import queueing_tool as qt
import numpy as np

# Make an adjacency list
adja_list = [[1], [k for k in range(2, 22)]]

# Make an object that has the same dimensions as your adjacency list that
# specifies the type of queue along each edge.
edge_list = [[1], [2 for k in range(20)]]

# Creates a graph-tool graph using the adjacency list and edge list
g = qt.adjacency2graph(adjacency=adja_list, eType=edge_list)

# Make a mapping between the edge types and the queue classes that sit on each
# edge. Do not use 0 as a key, it's used to map to NullQueues.
q_classes = {0: qt.NullQueue, 1: qt.QueueServer, 2: qt.QueueServer}

# Define the parameters for each of the queues
rate  = lambda t: 25 + 350 * np.sin(np.pi * t / 2)**2
arr_f = lambda t: qt.poisson_random_measure(rate, 375, t)
ser_f = lambda t: t + np.random.exponential(0.2 / 2.5)

# Make a mapping between the edge types and the parameters used to make those
# queues. If a particular parameter is not given then th defaults are used.
q_args    = {1: {'arrival_f': arr_f,
                 'service_f': lambda t: t,
                 'AgentClass': qt.GreedyAgent},
             2: {'nServers': 1,
                 'service_f': ser_f} }
                  
# Put it all together to create the network
qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)

# The default layout is sfdp, which doesn't look good for this network.
# This makes a new one
pos = g.new_vertex_property('pos')
for v in g.vertices():
    if v == 0:
        pos[v] = [0, -0.25]
    elif v == 1:
        pos[v] = [0, -0.125]
    else:
        pos[v] = [-0.5 + (v - 2.0) / 20, 0]

# List the maximum number of agents from the default of 1000 to infinity
qn.max_agents = np.infty

# Before any simulations can take place the network must be initialized to
# allow arrivals from outside the network. This specifies that only type 1
# edges accept arrivals from outside the network.
qn.initialize(eType=1)

# Data is not collected by default. This makes all queues collect data as the
# simulations take place.
qn.start_collecting_data()

# Simulate the network 1.8 simulation time units
qn.simulate(t=1.8)

# Collect data
data = qn.get_queue_data()

# Animate while simulating
qn.animate(output_size=(700,200), pos=pos)
