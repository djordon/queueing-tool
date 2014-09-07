namespace  = set( dir() )

from .agents  import *
from .servers import *
from .network import *

#__author__    = "Daniel Jordon <dan@danjordon.com>"
#__copyright__ = "Copyright 2014 Daniel Jordon"
#__license__   = "MIT"
#__URL__       = "https://github.com/djordon/queueing-tool"
#__version__   = "0.1"

__all__ = list( set( dir() ) - namespace - {'namespace'} ) 

del namespace, agents, servers, network
