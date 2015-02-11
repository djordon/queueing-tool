from numpy          import infty, argmin
from numpy.random   import randint

class Agent :
    """The base class for an agent.

    ``Agent``\s are the objects that move throughout the network. ``Agents`` are 
    instantiated by a queue, and once serviced the ``Agent`` moves on to another 
    queue in the network. Each ``Agent`` *decides* where in the network it wants 
    to arrive at next but choosing amongst its options (uniformly) at random.

    Parameters
    ----------
    issn : tuple (optional, the default is ``(0,0)``\)
        A unique identifier for an agent. Is set automatically by the
        :class:`~QueueServer` that instantiates the ``Agent``\. The first slot 
        is the ``QueueServer``\'s edge index and the second slot is 
        specifies the ``Agent``\'s instantiation number for that queue.

    Attributes
    ----------
        issn : tuple
            A unique identifier for an agent.
        blocked : int
            Specifies how many times an agent has been blocked by a finite capacity queue.
    """
    def __init__(self, issn=(0,0), **kwargs) :
        self.issn     = issn
        self.blocked  = 0
        self._time    = 0         # agents arrival or departure time

    def __repr__(self) :
        return "Agent. edge: %s, time: %s" % (self.issn, self._time)

    def __lt__(a, b) :
        return a._time < b._time

    def __gt__(a, b) :
        return a._time > b._time

    def __eq__(a, b) :
        return a._time == b._time

    def __le__(a, b) :
        return a._time <= b._time

    def __ge__(a, b) :
        return a._time >= b._time


    def set_arrival(self, t) :
        """Set the agents arrival time to a queue to ``t``."""
        self._time = t


    def set_departure(self, t) :
        """Set the agents departure time from a queue to ``t``."""
        self._time = t


    def add_loss(self, *args, **kwargs) :
        """Adds one to the number of times the agent has been blocked from entering a
        finite capacity queue.
        """
        self.blocked   += 1 


    def desired_destination(self, network, edge) :
        """Returns the agents next destination given their current location on the
        network.

        An ``Agent`` chooses one of the out edges uniformly at random.

        Parameters
        ----------
        network : :class:`~queueing_tool.network.QueueNetwork`
        edge : tuple
            A 3-tuple indicating which edge this agent is located at. The first two 
            slots indicate the current edge's source and target vertices, while the
            third slot indicates this edges ``edge_index``.

        Returns
        -------
        out : int
            Returns an the edge index corresponding to the agents next edge to visit
            in the network. 
        """
        n   = len( network.out_edges[edge[1]] )
        d   = randint(0, n)
        z   = network.out_edges[edge[1]][d]
        return z


    def queue_action(self, queue, *args, **kwargs) :
        """A function that acts on the queue that it is departing from. By default it does
        nothing to the queue.
        """
        pass


    def __deepcopy__(self, memo) :
        new_agent           = self.__class__()
        new_agent.issn      = copy.copy(self.issn)
        new_agent._time     = copy.copy(self._time)
        new_agent.blocked   = copy.copy(self.blocked)
        return new_agent


class GreedyAgent(Agent) :
    """An agent that chooses the queue with the shortest line as his destination."""
    def __init__(self, issn) :
        Agent.__init__(self, issn)

    def __repr__(self) :
        return "GreedyAgent. edge: %s, time: %s" % (self.issn, self._time)

    def desired_destination(self, network, edge) :
        """Returns the agents next destination given their current location on the
        network. 

        ``GreedyAgents`` choose their next destination with-in the network by
        picking the adjacent queue with the fewest number of ``Agents`` in the queue.

        Parameters
        ----------
        network : :class:`~queueing_tool.network.QueueNetwork`
        edge : tuple
            A 3-tuple indicating which edge this agent is located at. The first two 
            slots indicate the current edge's source and target vertices, while the
            third slot indicates this edge's ``edge_index``.

        Returns
        -------
        out : int
            Returns an the edge index corresponding to the agents next edge to visit
            in the network. 
        """
        adjacent_edges = network.out_edges[edge[1]]
        d = argmin([network.edge2queue[d].nQueued() for d in adjacent_edges])
        return adjacent_edges[d]


class InftyAgent :
    """An special agent that only operates within the ``QueueServer`` class.

    This agent never interacts with the ``QueueNetwork``.
    """
    def __init__(self) :
        self._time = infty

    def __repr__(self) :
        return "InftyAgent"

    def __lt__(a, b) :
        return a._time < b._time

    def __gt__(a, b) :
        return a._time > b._time

    def __eq__(a, b) :
        return a._time == b._time

    def __le__(a, b) :
        return a._time <= b._time

    def __ge__(a, b) :
        return a._time >= b._time

    def __deepcopy__(self, memo) :
        return self.__class__()
