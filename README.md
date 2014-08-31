Queueing-tool
=============

Queueing-toool is an efficient python module for analysis of graphs. It is written for python3 and uses [`graph-tool`](http://graph-tool.skewed.de/) and [`numpy`](http://www.numpy.org/). The main goal of queueing-toool is to aid in agent based simulations of networks. The original purpose was to analyze transportation networks.

Note: This program is under active development and is distributed in the hope that it will be useful, but WITHOUT ANY GUARANTEES.

## Features

  - It's fast. Since queueing-tool is designed for large scale simulations, it is build with performance in mind. Using graph-tool for the networking component makes most network operations very fast. With a heap based scheduler managing events, things don't get much quicker.
  - It's flexible. You can use general functions (including time dependent functions) for the arrival and departure rates for each queue. You can have queues along the edges (for either directed or undirected graphs) and queues along the nodes too.
  - It's friendly (sometimes). There are functions that make incorporating transportation networks from [openstreetmaps.org](www.openstreetmaps.org) easy.


## Setup

The hardest part of getting graph-tool installed. There are [precompiled packages](http://graph-tool.skewed.de/download#packages) made for Debian & Ubuntu, Gentoo, and Arch linux, and there are macport files [here](http://www.macports.org/ports.php?by=name&substr=graph-tool). There does not seem to be much testing/support for Windows users.

Once you have graph-tool and numpy install you can import queueing-tool with

```python
import queue_tool as qt
```

##### MIT Licensed
