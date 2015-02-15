Overview
========

``Queueing-tool`` is an Python simulation package for analysing networks of queues. The simulations are agent based, where events are comprised as arrivals and departures of agents. Arrivals from outside the network are simulated as agents, which then move throughout the network from queue to queue. The network is represented as a graph, which is handled by `graph-tool`. 

Agents and queues are separate classes that can interact with one another. Agents control their one movement, while the queues control service rates. Each queue can have a general time-dependent arrival and service distribution.

The package also contains visualization component, whereby the user can see queueing dynamics in real-time as the simulations take place. The simulations are event-based, with agents and queues interacting with one another.

There are three major components to ``queueing-tool``: the ``QueueServer`` classes, ``Agent`` classes, and ``QueueNetwork`` classes. The package includes several different types of each class.

  1. The ``QueueServer`` is the basic part of the package. They have arrivals enter the queue from the outside world and and these arrivals receive service from the queue before moving on. Each queue can have any arrival and service distribution, and these distributions can depend on time. In `Kendall's notation`_, these are 
    :math:`\text{GI}_t/\text{GI}_t/c/\infty/N/\text{FIFO}` queues.

  2. ``Agents`` are the objects that move throughout the network. When an instance of the network is created it starts empty. ``Agents`` are created by a queue and once serviced the ``Agent`` moves on to another queue in the network. Each ``Agent`` *decides* where in the network it wants to arrive at next. An ``Agents`` can also interact with the queues it visits and change the properties of the queue.
  
  3. The ``QueueNetwork`` manages the routing of agents from queue to queue. It also manages congestion, and information about the status of the network, such as how many agents are at each queue in the network.

The ``QueueServer`` class
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``QueueServer`` class is the base queue classes used in ``queueing-tool``. Each queue can have a general time-dependent arrival and departure distribution and can cap the number of arrivals from outside the network (by setting a time cap, or a quota).

This class

The ``Agent`` class
^^^^^^^^^^^^^^^^^^^

Agents are *semi-autonomous* objects that move throughout the network. They can set their own agenda, or behave in a proscribed random fashion. When an agent departs a queue, they can *act* on the queue from which they are departing.

When an agent departs a queue

QueueServer
^^^^^^^^^^^

Something
  .. _Kendall's notation: http://en.wikipedia.org/wiki/Kendall%27s_notation
