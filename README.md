Queueing-tool
=============

`queueing-tool` is an efficient `python` module for analysis of graphs. It is written for `Python` (2.7 or 3.2+)  and uses [`graph-tool`](http://graph-tool.skewed.de/) and [`numpy`](http://www.numpy.org/). The main goal of `queueing-tool` is to aid in simulations of networks. The original purpose was to analyze transportation networks.

## Features

  - It's fast. `queueing-tool` is designed for medium sized agent-based simulations -- say the downtown road network of a city with 5000 cars -- and it is build with performance in mind. Using `graph-tool` for the networking component makes most network operations very fast. With a heap based scheduler managing events, things don't get much quicker.
  - It's flexible. You can use general functions (including time dependent functions) for the arrival and departure rates for each queue. You can have queues along the edges (for either directed or undirected graphs) and queues along the nodes too.
  - It's friendly (sometimes). There are functions that make incorporating transportation networks from [openstreetmaps.org](www.openstreetmaps.org) easy.


## Installation

To install, you first need to have [`graph-tool`](http://graph-tool.skewed.de/) and [`numpy`](http://www.numpy.org/) installed. There are precompiled [packages](http://graph-tool.skewed.de/download#packages) of `graph-tool` made for Debian & Ubuntu, Gentoo, and Arch linux, and there are macport files [here](http://www.macports.org/ports.php?by=name&substr=graph-tool). You only need the `python 3` version of `graph-tool` installed for `queueing-tool`. Note that there does not seem to be much support for `graph-tool` under Windows users. 

After you have `graph-tool` and `numpy` installed, clone this repository and open a prompt in the package directory. To install, run


```bash
sudo python3 setup.py install
```

#### Local installation

If you only want to test out `queueing-tool`, open a prompt in the package directory and run

```bash
python3 setup.py install --user
```

or 

```bash
python setup.py install --user
```

The above code will install `queueing_tool` at `~/.local/lib/python3.X/`, where `X` is the latest version of python installed on your machine. If `~/.local/lib/python3.X` does not exists it will create it. Once `queueing-tool` is installed you can import it with the following

```python
import queueing_tool as qt
```

## The `QueueServer` Class

The `QueueServer` class is a generic G/G/k [queueing](http://en.wikipedia.org/wiki/Queueing_theory). There are base classes supplied for two service disciplines, one that implements FIFO and the other LIFO. Each `QueueServer` must manage a future `time` variable that tells it's 'time of the next event', and a local time variable that tells the time of the last event (or the event that is occuring now). The service function can depend on time, and number of agents at the `QueueServer`.

Whenever a `QueueServer` instance is created it does not have any arrivals, and does not accept any arrivals from the *outside world* by default. To accept *outside world* arrivals, you must call the `initialize()` function. A `QueueServer` can always accept arrivals from with the `QueueNetwork`, but will not generate new arrivals by default.

## The `QueueNetwork` Class

The `QueueNetwork` class is a manager of the events that take place at the various nodes and edges in the network. It's primary task is sorting of events that will take place in the future. For each edge in the network, there is a corresponding `QueueServer`, which is responsible for maintaining its own schedule of arriving and departing agents.

Whenever an agent departs from a `QueueServer` the agent must decide where to go next. Once this decision is made the agent arrives at the next queue/edge immediately, where he is placed at the end of the line in that queue (if there is one, otherwise he enters into service).

### Creating Pittsburgh's Transportation Network

The [following](http://nbviewer.ipython.org/gist/djordon/975bf898c1ed2f4c8198) is an old demo showing how to set up downtown Pittsburgh's traffic network. It does not use `queueing_tool`, but details how to use [openstreetmaps.org] with `graph_tool` to create a graph for inclusion with `queueing_tool`.


##### Copyright and license

Code and documentation copyright 2014-2015 Daniel Jordon. Code released under the [MIT license](https://github.com/djordon/queueing-tool/blob/master/LICENSE).
