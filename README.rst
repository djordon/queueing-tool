Queueing-tool
=============

|Build Status| |Coverage Status|

Queueing-tool is a package for simulating and analyzing networks of
`queues <http://en.wikipedia.org/wiki/Queueing_theory>`__ that is
written in Python. It also includes a visualization component for seeing
network dynamics.

Documentation
-------------

The package documentation can be found at
http://queueing-tool.readthedocs.org/.

Features
--------

-  **Fast setup**. The setup process is designed to be quick and
   painless, while still giving you complete control over the design of
   your network. This allows you to design your queueing networks
   quickly, which comes in handy when modeling and analyzing larger
   networks. You can use general functions (including time dependent
   functions) for the arrival and departure functions for each queue.
   Queueing-tool networks allow for probabilistic routing for modeling
   `Jackson networks <http://en.wikipedia.org/wiki/Jackson_network>`__,
   finite capacity queues and 2 different blocking protocols for
   studying `loss
   networks <http://en.wikipedia.org/wiki/Loss_network>`__, as well as
   specialty queues for modeling more exotic networks.
-  **Fast simulation**. Queueing-tool is designed to run very quickly.
   The core algorithms were written using
   `cython <http://cython.org/>`__, which exports C-extensions of those
   routines.
-  **Visualizations**. There are several tools that allow you to easily
   view congestion and movement within your network. This includes ready
   made functions for animating network dynamics, such as congestion,
   while your simulations take place.
-  **Full documentation**. Every function and class is fully documented
   both `online <http://queueing-tool.readthedocs.org/>`__ and in the
   docstrings. There are also worked out examples included in the
   source.

Installation
------------

**Prerequisites:** Queueing-tool runs on Python 2.7 and 3.3-3.5 and it
requires `networkx <http://networkx.readthedocs.org/en/stable/>`__ and
`numpy <http://www.numpy.org/>`__. If you want to plot, you will need
`matplotlib <http://matplotlib.org/>`__.

**Platforms**: Queueing tool should work on Windows and any unix-like
platform such as Linux or Mac OS X. The developers of
`numpy <http://docs.scipy.org/doc/numpy/user/install.html>`__ have
compiled binary packages for Windows, Mac OS X, and several other
unix-like operating systems.

**Manual installation**: Download the latest release from github and run
the following commands in a terminal:

.. code:: bash

    tar xzvf queueing_tool-[VERSION].zip
    cd queueing_tool-[VERSION].zip
    python setup.py build
    sudo python setup.py install

To install locally run the following command in a terminal:

.. code:: bash

    python setup.py develop --user

Copyright and license
---------------------

Code and documentation Copyright 2014-2016 Daniel Jordon. Code released
under the `MIT
license <https://github.com/djordon/queueing-tool/blob/master/LICENSE.txt>`__.

.. |Build Status| image:: https://travis-ci.org/djordon/queueing-tool.svg?branch=master
   :target: https://travis-ci.org/djordon/queueing-tool
.. |Coverage Status| image:: https://coveralls.io/repos/djordon/queueing-tool/badge.svg?branch=master
   :target: https://coveralls.io/r/djordon/queueing-tool?branch=master
