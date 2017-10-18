import functools

import numpy as np
import queueing_tool as qt


# Make an adjacency list
adja_list = {0: {1: {}}, 1: {k: {} for k in range(2, 22)}}

# Make an object that has the same dimensions as your adjacency list that
# specifies the type of queue along each edge.
edge_list = {0: {1: 1}, 1: {k: 2 for k in range(2, 22)}}

# Creates a networkx directed graph using the adjacency list and edge list
g = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)

# Make a mapping between the edge types and the queue classes that sit on each
# edge. Do not use 0 as a key, it's used to map to NullQueues.
q_classes = {0: qt.NullQueue, 1: qt.QueueServer, 2: qt.QueueServer}


# Define the parameters for each of the queues
def rate(t):
    return 25 + 350 * np.sin(np.pi * t / 2)**2


def ser_f(t):
    return t + np.random.exponential(0.2 / 2.5)


def identity(t):
    return t

arr_f = functools.partial(qt.poisson_random_measure, rate=rate, rate_max=375)


# Make a mapping between the edge types and the parameters used to make those
# queues. If a particular parameter is not given then th defaults are used.
q_args = {
    1: {'arrival_f': arr_f,
        'service_f': identity,
        'AgentFactory': qt.GreedyAgent},
    2: {'num_servers': 1,
        'service_f': ser_f}
}

# Put it all together to create the network
qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args, seed=13)

# The default layout is spring_layout, which doesn't look good for this network.
# This makes a new one
qn.g.new_vertex_property('pos')
pos = {}
for v in qn.g.nodes():
    if v == 0:
        pos[v] = [0, 1]
    elif v == 1:
        pos[v] = [0, 0.5]
    else:
        pos[v] = [-5. + (v - 2.0) / 2, 0]

qn.g.set_pos(pos)
# qn.draw(fname="store1.png", transparent=True, figsize=(12, 3),
#        bgcolor=[0,0,0,0], bbox_inches='tight')

# List the maximum number of agents from the default of 1000 to infinity
qn.max_agents = np.infty

# Before any simulations can take place the network must be initialized to
# allow arrivals from outside the network. This specifies that only type 1
# edges accept arrivals from outside the network.
qn.initialize(edge_type=1)

# Data is not collected by default. This makes all queues collect data as the
# simulations take place.
qn.start_collecting_data()

# Simulate the network 1.8 simulation time units
qn.simulate(t=1.9)
# qn.draw(fname="sim1.png", transparent=True, figsize=(12, 3),
#        bgcolor=[0,0,0,0], bbox_inches='tight')

# Collect data
data = qn.get_queue_data()

# Animate while simulating
qn.animate(figsize=(16, 3.5), pos=pos)
