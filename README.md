
Queueing-tool
=============

Queueing-tool is a package for simulating and analyzing networks of [queues][] that is written in Python. It also includes a visualization component for seeing network dynamics.

Features
--------

- **Easy to use**. The setup process is designed to be quick and painless, while still giving you complete control over the design of your network. This allows you to design your queueing networks quickly, which comes in handy when modeling and analyzing larger networks.
- **Large feature set**. You can use general functions (including time dependent functions) for the arrival and departure functions for each queue. Queueing-tool also includes allows for probabilistic routing for modeling [Jackson networks][], finite capacity queues and 2 different blocking protocols for studying [loss networks][], as well as specialty queues for modeling more exotic networks.
- **Fast simulation**. Queueing-tool is designed to run very quickly. The core algorithms were written using [cython][], which exports C-extensions of these routines, and the underlying graph utilizes [graph-tool][], a fast and powerful graph package.
- **Beautiful visualizations**. There are several tools that allow you to easily view congestion and movement within your network. This includes ready made functions for animating network dynamics, such as congestion, while your simulations take place. These features use graph-tool's powerful visualization capabilities and can accommodate many of its visualization features.
- **Full documentation**. Every function and class is fully documented both [online][] and in the docstrings. There are also worked out examples included in the source.

Installation
------------

**Prerequisites:** Queueing-tool runs on Python 2.7, 3.2, 3.3, and 3.4. It requires [graph-tool][] and [numpy][], as well as [scipy][] if you want to run the tests (which you don't need to run). 

**Platforms**: Queueing tool should work on Windows and any unix-like platform such as Linux or Mac OS X. The developers of [numpy][2] and [scipy][3] have compiled binary packages for Windows, Mac OS X, and several other unix-like operating systems, while the developers of [graph-tool][1] have binaries for Mac OS X and several linux distributions but not Windows.

**Manual installation**: Download the latest release from github and do the the following

```bash
tar xzvf queueing_tool-[VERSION].zip
cd queueing_tool-[VERSION].zip
python setup.py build
sudo python setup.py install
```

To install locally use the following

```bash
python setup.py install --user
```


Copyright and license
---------------------

Code and documentation Copyright 2014-2015 Daniel Jordon. Code released under the [MIT license][].

  [queues]: http://en.wikipedia.org/wiki/Queueing_theory
  [Jackson networks]: http://en.wikipedia.org/wiki/Jackson_network
  [loss networks]: http://en.wikipedia.org/wiki/Loss_network
  [cython]: http://cython.org/
  [graph-tool]: http://graph-tool.skewed.de/
  [online]: http://queueing-tool.readthedocs.org/
  [1]: http://graph-tool.skewed.de/download#packages
  [numpy]: http://www.numpy.org/
  [scipy]: http://scipy.org/
  [2]: http://docs.scipy.org/doc/numpy/user/install.html
  [3]: http://scipy.org/install.html
  [MIT license]: https://github.com/djordon/queueing-tool/blob/master/LICENSE
