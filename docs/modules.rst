Module documenation
===================

Using this documentation
------------------------

Documentation is also available in docstrings provided with the code.

The docstring examples assume ``queueing_tool`` has been imported as ``qt`` using the following command::

    >>> import queueing_tool as qt

We also use numpy and graph-tool, which were imported with:

    >>> import graph_tool.all as gt
    >>> import numpy as np

Code snippets are indicated by three greater-than signs, such as the above code snippet or the following::

    >>> x = [k for k in range(4)]

.. toctree::

    network
    queues
    generation
