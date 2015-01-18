from .queues      import *
from .network     import *
from .generation  import *
from . import queues
from . import network
from . import generation

__all__ = []

__author__    = 'Daniel Jordon <dan@danjordon.com>'
#__copyright__ = 'Copyright 2014 Daniel Jordon'
__license__   = 'MIT'
__URL__       = 'https://github.com/djordon/queueing-tool'
__version__   = '0.1'

__all__.extend(['__author__', '__license__', '__URL__', '__version__'])
__all__.extend(queues.__all__)
__all__.extend(network.__all__)
__all__.extend(generation.__all__)

del queues, network, generation
