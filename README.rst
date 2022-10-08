Queueing-tool
=============

|Build Status| |Coverage Status| |pyversion| |license|

Queueing-tool is a package for simulating and analyzing networks. It is an
event based simulator that uses
`queues <http://en.wikipedia.org/wiki/Queueing_theory>`__ to simulate congestion
and waiting on the network that includes tools for
visualizing network dynamics.

Documentation
-------------

The package documentation can be found at
http://queueing-tool.readthedocs.org/.

Features
--------

-  **Fast simulation**. Queueing-tool is designed to run very quickly;
   the core algorithms were written in `cython <http://cython.org/>`__.
-  **Visualizations**. There are several tools that allow you to easily
   view congestion and movement within your network. This includes ready
   made functions for animating network dynamics while your simulations
   take place.
-  **Full documentation**. Every function and class is fully documented
   both `online <http://queueing-tool.readthedocs.org/>`__ and in the
   docstrings.
-  **Fast setup**. The network is represented as a
   `networkx graph <http://networkx.readthedocs.org/en/stable/>`__.
   Queueing-tool networks allow for probabilistic routing, finite
   capacity queues, and different blocking protocols for analyzing
   `loss networks <http://en.wikipedia.org/wiki/Loss_network>`__.

Installation
------------

**Prerequisites:** Queueing-tool runs on Python 2.7 and 3.4-3.10, but
changes going forward are only tested against Python 3.6-3.10. Queueing-tool
requires `networkx <http://networkx.readthedocs.org/en/stable/>`__ and
`numpy <http://www.numpy.org/>`__, and depends on
`matplotlib <http://matplotlib.org/>`__ if you want to plot.

**Installation**: To install from
`PyPI <https://pypi.python.org/pypi/queueing-tool>`__ use:

.. code:: bash

    pip install queueing-tool

The above will automatically install networkx and numpy. If you want to plot use:

.. code:: bash

    pip install queueing-tool[plotting]

Note that `queueing-tool` uses `networkx`'s pagerank implementation. As of
networkx 2.8.6, they have several versions of the pagerank algorithm and
`queueing-tool` defaults to using the version that requires `scipy`. If
`scipy` is not installed then it trys the `numpy` based implementation. 

After installation, import queueing-tool with something like:

.. code:: python

    import queueing_tool as qt


Bugs and issues
---------------

The issue tracker is at https://github.com/djordon/queueing-tool/issues. Please report any bugs or issue that you find there. Of course, pull requests are always welcome.


Copyright and license
---------------------

Code and documentation Copyright 2014-2022 Daniel Jordon. Code released
under the `MIT
license <https://github.com/djordon/queueing-tool/blob/master/LICENSE.txt>`__.

.. |Build Status| image:: https://github.com/djordon/queueing-tool/actions/workflows/run-tests.yml/badge.svg
   :target: https://github.com/djordon/queueing-tool/actions/workflows/run-tests.yml/badge.svg

.. |Coverage Status| image:: https://coveralls.io/repos/djordon/queueing-tool/badge.svg?branch=master
   :target: https://coveralls.io/r/djordon/queueing-tool?branch=master

.. |pyversion| image:: https://img.shields.io/pypi/pyversions/queueing-tool.svg
    :alt: Supported Python versions.
    :target: http://pypi.python.org/pypi/queueing-tool/

.. |license| image:: https://img.shields.io/pypi/l/queueing-tool.svg
    :alt: MIT License
    :target: https://opensource.org/licenses/MIT
