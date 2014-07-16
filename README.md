Queueing-network:
================

A python module for simulation of queueing networks
---

This is a python class for a network of queues. It is written for python3.* and uses [`graph_tool`](http://graph-tool.skewed.de/) (and [`numpy`](http://www.numpy.org/) of course).


The main goal of this class is to aid in agent based simulations of road traffic networks where the transportation infrastructure is modeled as a network of queues. The physical transportation network is modeled using a graph, and each node in the graph is either a traffic light, a destination, an off-street parking garage, or a set of on-street parking spots. Traffic patterns in the network are modeled using G<sub>t</sub>/G<sub>t</sub>/k FIFO queues with varying departure rates (see [this](http://en.wikipedia.org/wiki/Queueing_theory) for a description of queues and Kendall notation). For example, each intersections can be calibrated to match the timing of traffic light signals. 

The graph for downtown metropolitan areas can be obtained by downloading an `osm` file from [openstreetmaps.org](www.openstreetmaps.org) and converting it to a graph. We've written functions that automate most of this process but they sometimes yield lackluster results. We have written a markdown for using [openstreetmap.org](http://www.openstreetmap.org) to create a the graph of downtown Pittsburgh, you can find it [here](http://nbviewer.ipython.org/gist/danieljordon/975bf898c1ed2f4c8198).

There are a variety of agents that can be used to travel throughout the graph. There are: regular agents, which take the shortest path to their destination; random agents, which don't have destinations and travel randomly in the network; and learning agents, which are used to learn good paths in the network given congestion in the roads. Learning agents are used wtih the approximate dynamic programming functions.

One of the primary functions of this module is to aid in evaluting the use communication technologies in automobiles.
