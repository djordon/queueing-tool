from __future__ import absolute_import

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version

from queueing_tool.queues import *
import queueing_tool.queues as queues

from queueing_tool.network import *
import queueing_tool.network as network

from queueing_tool.graph import *
import queueing_tool.graph as graph

__version__ = version(__package__ or __name__)

__all__ = []
__all__.extend(['__version__'])
__all__.extend(queues.__all__)
__all__.extend(network.__all__)
__all__.extend(graph.__all__)

# del queues, network, generation, graph
