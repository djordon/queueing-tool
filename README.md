
Queueing-tool
=============

[![Build Status](https://travis-ci.org/djordon/queueing-tool.svg?branch=master)](https://travis-ci.org/djordon/queueing-tool) [![Coverage Status](https://coveralls.io/repos/djordon/queueing-tool/badge.svg?branch=master)](https://coveralls.io/r/djordon/queueing-tool?branch=master)


Queueing-tool is a package for simulating and analyzing networks of [queues][] that is written in Python. It also includes a visualization component for seeing network dynamics.

Documentation
-------------

The package documentation can be found at [http://queueing-tool.readthedocs.org/](http://queueing-tool.readthedocs.org/).

Features
--------

- **Fast setup**. The setup process is designed to be quick and painless, while still giving you complete control over the design of your network. This allows you to design your queueing networks quickly, which comes in handy when modeling and analyzing larger networks. You can use general functions (including time dependent functions) for the arrival and departure functions for each queue. Queueing-tool networks allow for probabilistic routing for modeling [Jackson networks][], finite capacity queues and 2 different blocking protocols for studying [loss networks][], as well as specialty queues for modeling more exotic networks.
- **Fast simulation**. Queueing-tool is designed to run very quickly. The core algorithms were written using [cython][], which exports C-extensions of those routines.
- **Visualizations**. There are several tools that allow you to easily view congestion and movement within your network. This includes ready made functions for animating network dynamics, such as congestion, while your simulations take place.
- **Full documentation**. Every function and class is fully documented both [online][] and in the docstrings. There are also worked out examples included in the source.

Installation
------------

**Prerequisites:** Queueing-tool runs on Python 2.7 and 3.3-3.5 and it requires [networkx][] and [numpy][]. If you want to plot, you will need [matplotlib][].

**Platforms**: Queueing tool should work on Windows and any unix-like platform such as Linux or Mac OS X. The developers of [numpy][2] have compiled binary packages for Windows, Mac OS X, and several other unix-like operating systems.

**Manual installation**: Download the latest release from github and run the following commands in a terminal:

```bash
tar xzvf queueing_tool-[VERSION].zip
cd queueing_tool-[VERSION].zip
python setup.py build
sudo python setup.py install
```

To install locally run the following command in a terminal:

```bash
python setup.py develop --user
```


Copyright and license
---------------------

Code and documentation Copyright 2014-2016 Daniel Jordon. Code released under the [MIT license][].

  [queues]: http://en.wikipedia.org/wiki/Queueing_theory
  [Jackson networks]: http://en.wikipedia.org/wiki/Jackson_network
  [loss networks]: http://en.wikipedia.org/wiki/Loss_network
  [cython]: http://cython.org/
  [online]: http://queueing-tool.readthedocs.org/
  [1]: http://graph-tool.skewed.de/download#packages
  [matplotlib]: http://matplotlib.org/
  [numpy]: http://www.numpy.org/
  [networkx]: http://networkx.readthedocs.org/en/stable/
  [2]: http://docs.scipy.org/doc/numpy/user/install.html
  [MIT license]: https://github.com/djordon/queueing-tool/blob/master/LICENSE
