Quick Start
===========


Installation
------------

If you want to try out ``queueing-tool``, here's what you do. First you'll need to install ``graph-tool`` and ``numpy``. Installation instructions for ``numpy`` can be found in the `numpy docs <http://docs.scipy.org/doc/numpy/user/install.html>`_, and there are installable binary packages available for Windows, Mac OS X, and many distributions of Lunx. For ``graph-tool`` there are precompiled `packages <http://graph-tool.skewed.de/download#packages>`_ made for Debian & Ubuntu. If you follow the link you'll find ``graph-tool``'s Ubuntu (and Debian) repositories. Once you add them to your sources, open a terminal and run::

    $ sudo apt-get install python3-graph-tool

Next, you'll want to ``clone`` the ``queueing-tool`` repository to your hard drive. To do so, type::

    $ git clone https://github.com/djordon/queueing-tool

in the terminal. If you don't have git installed on your computer (and you don't want to install it) you can go to the `github page <https://github.com/djordon/queueing-tool>`_ and download a zip file of the package. Once you have the package locally on your computer (and unzipped if you downloaded if from github), change directories to the ``queueing-tool`` directory in your terminal and install locally using::

    $ sudo python3 setup.py install

Or use the following code if you only want to install it locally::

    $ python3 setup.py install --user

Note that you **do not** need to use ``sudo`` before this command.

An example
----------

Its probably best to become acquainted with ``queueing-tool`` by way of an example. Suppose you wanted to measure the performance of two queueing systems. The first system, which looks as follows:


has people entering the system from the node on the far left, and choosing between the two nodes one the right. They choose the queue with the shortest line. People wait in line before receiving service and once they receive service they depart the system. Also, to make this system more lifelike, we'll change the expected number of people entering the network as the day goes on, with a majority arriving in the afternoon. 

In ``queueing-tool``, each *person* entering the system is represented as an ``Agent``\. Each ``Agent`` decides how they navigate in the network. In this system an ``Agent`` chooses the shortest queue to enter at whenever they choose which queue to arrive at next. There is already a built-in class of agents that navigate by choosing the shortest queue; this class is the :class:`.GreedyAgent` class.

The network is represented as a ``graph-tool`` :class:`~graph_tool.Graph`. On top of each edge in the graph sits the queues, where each queue is represented as a :class:`.QueueServer`. Each :class:`.QueueServer` handles their arrival from outside the network, as well as the servicing of any arriving :class:`.GreedyAgent`\. On each edge, you can have a different type of :class:`.QueueServer`. They do not handle any routing between queues.

To create an the network you just need to specify an adjacency list (or adjacency matrix):

.. testsetup::

    import graph_tool.all as gt
    import queueing_tool as qt

.. doctest::

    >>> adja_list = [[1], [2, 3], [4], [4], []]

To specify what type of queue sits on each edge, you specify an adjacency list like object. 

.. doctest::

    >>> edge_list = [[1], [2, 2], [0], [0], []]
    >>> graph     = qt.adjacency2graph(adjacency=adja_list, edge_type=edge_list)

Each edge has a type that is used to define which queue sits on that edge. You specify the arguments for each class in another variable. If you want to keep the defaults for a particular type you do not need to specify any arguments.

.. doctest::

    >>> q_classes = { 0 : qt.NullQueue, 1 : qt.QueueServer, 2 : qt.QueueServer}
    >>> q_args    = { 1 : {'arrival_f'  : lambda t: t + 2 + np.sin(t),
    ...                    'service_f'  : lambda t: t,
    ...                    'AgentClass' : qt.GreedyAgent},
    ...               2 : {'nServers'   : 5,
    ...                    'arrival_f'  : lambda t: t + 2 + np.sin(t),
    ...                    'service_f'  : lambda t: t + np.random.exponential(1)} }

Use the following code to create this queueing network

.. doctest::

    >>> QN  = qt.QueueNetwork(g=graph, q_classes=q_classes, q_args=q_args)

By default, each ``QueueServer`` starts with no arrivals from outside the network. This means the some queues needs to be initialized before they the ``QueueNetwork`` can simulate anything. You can specify which queues are initialized with
``QueueNetwork``'s ``initialize()`` function. In this example, we only want agents arriving from the type 1 edge (the edge between nodes 0 and 1) so we run the following code.

.. doctest::

    >>> QN.initialize(types=[1])

To simulate for a specified amount of time run.

.. doctest::

    >>> QN.simulate(t=10)
    >>> Qn.draw()

.. doctest::
    :hide:

    QN.draw(output="two-nodes.png")


.. figure:: two-nodes.png
    :align: center

    A simple directed graph with two vertices and one edge, created by
    the commands above.
