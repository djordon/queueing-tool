Overview
========

Queueing-tool is an Python simulation package for analysing networks of queues. The simulations are agent based, where events are comprised as arrivals and departures of agents. Arrivals from outside the network are simulated as agents, which then move throughout the network from queue to queue. The network is represented as a graph, which is handled by `graph-tool`. 

Agents and queues are separate classes that can interact with one another. Agents control their one movement, while the queues control service rates. Each queue can have a general time-dependent arrival and service distribution.

The package also contains visualization component, whereby the user can see queueing dynamics in real-time as the simulations take place. The simulations are event-based, with agents and queues interacting with one another.

# Queues and Agents

Queueing-tool has three major components: the `QueueServer` classes, `Agent` classes, and `QueueNetwork` classes. The package includes several different types of each class.

  1. The `QueueServer` is the basic part of the package. They have arrivals enter the queue from the outside world and and these arrivals receive service from the queue before moving on. Each queue can have any arrival and service distribution, and these distributions can depend on time. In Kendall notation, these are $GI(t)/GI(t)/k/c/n$ queues.

  2. `Agents` are the objects that move throughout the network. When an instance of the network is created it starts empty. `Agents` are created by a queue and once serviced the `Agent` moves on to another queue in the network. Each `Agent` *decides* where in the network it wants to arrive at next. An `Agents` can also interact with the queues it visits and change the properties of the queue.
  
  3. The `QueueNetwork` manages the routing of agents from queue to queue. It also manages congestion, and information about the status of the network, such as how many agents are at each queue in the network.

### The `QueueServer` class

The `QueueServer` class is the base queue classes used in `queueing-tool`. Each queue can have a general time-dependent arrival and departure distribution and can cap the number of arrivals from outside the network (by setting a time cap, or a quota). In Kendall's notation, these are $\text{G}_t/\text{G}_t/k$ queues.

This class

### The Agent class

Agents are *semi-autonomous* objects that move throughout the network. They can set their own agenda, or behave in a proscribed random fashion. When an agent departs a queue, they can *act* on the queue from which they are departing.

When an agent departs a queue

QueueServer
"""The generic queue-server class.

Creates an instance of a :math:`\text{GI}_t/\text{GI}_t/n` queue. Each of the 
supplied parameters are set to a corresponding class variable/attribute.

Each queue sits on an edge in a graph. When drawing the graph, the queue colors the edges and 
the edge's target vertex.

