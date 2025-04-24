
try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version

from queueing_tool import graph, network, queues
from queueing_tool.graph import *
from queueing_tool.network import *
from queueing_tool.queues import *

__version__ = version(__package__ or __name__)

__all__ = []
__all__.extend(["__version__"])
__all__.extend(queues.__all__)
__all__.extend(network.__all__)
__all__.extend(graph.__all__)

# del queues, network, generation, graph
