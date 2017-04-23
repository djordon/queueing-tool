from numpy import infty
from numpy.random import uniform

from queueing_tool.queues.choice import _choice, _argmin


class Agent(object):
    """The base class for an agent.

    ``Agents`` are the objects that move throughout the network.
    ``Agents`` are instantiated by a queue, and once serviced the
    ``Agent`` moves on to another queue in the network. Each ``Agent``
    *decides* where in the network it wants to arrive at next but
    choosing amongst its options randomly. The probabilities are
    specified in :class:`QueueNetwork's<.QueueNetwork>` transition
    matrix. See :meth:`.set_transitions` for changing the routing
    probabilities.

    Parameters
    ----------
    agent_id : tuple (optional, default: ``(0, 0)``)
        A unique identifier for an agent. Is set automatically by the
        :class:`.QueueServer` that instantiates the ``Agent``. The
        first slot is the :class:`QueueServer's<.QueueServer>` edge
        index and the second slot is the ``Agent's``
        instantiation number for that queue.
    **kwargs :
        Unused.

    Attributes
    ----------
    agent_id : tuple
        A unique identifier for an agent.
    blocked : int
        Specifies how many times an agent has been blocked by a finite
        capacity queue.
    """
    def __init__(self, agent_id=(0, 0), **kwargs):
        self.agent_id = agent_id
        self.blocked = 0
        self._time = 0  # The agents arrival or departure time

    def __repr__(self):
        return "Agent; agent_id:{0}. time: {1}".format(self.agent_id, round(self._time, 3))

    def __lt__(self, b):
        return self._time < b._time

    def __gt__(self, b):
        return self._time > b._time

    def __eq__(self, b):
        return self._time == b._time

    def __le__(self, b):
        return self._time <= b._time

    def __ge__(self, b):
        return self._time >= b._time

    def add_loss(self, *args, **kwargs):
        """Adds one to the number of times the agent has been blocked
        from entering a queue.
        """
        self.blocked += 1

    def desired_destination(self, network, edge):
        """Returns the agents next destination given their current
        location on the network.

        An ``Agent`` chooses one of the out edges at random. The
        probability that the ``Agent`` will travel along a specific
        edge is specified in the :class:`QueueNetwork's<.QueueNetwork>`
        transition matrix.

        Parameters
        ----------
        network : :class:`.QueueNetwork`
            The :class:`.QueueNetwork` where the Agent resides.
        edge : tuple
            A 4-tuple indicating which edge this agent is located at.
            The first two slots indicate the current edge's source and
            target vertices, while the third slot indicates this edges
            ``edge_index``. The last slot indicates the edge type of
            that edge

        Returns
        -------
        out : int
            Returns an the edge index corresponding to the agents next
            edge to visit in the network.

        See Also
        --------
        :meth:`.transitions` : :class:`QueueNetwork's<.QueueNetwork>`
            method that returns the transition probabilities for each
            edge in the graph.
        """
        n = len(network.out_edges[edge[1]])
        if n <= 1:
            return network.out_edges[edge[1]][0]

        u = uniform()
        pr = network._route_probs[edge[1]]
        k = _choice(pr, u, n)

        # _choice returns an integer between 0 and n-1 where the
        # probability of k being selected is equal to pr[k].
        return network.out_edges[edge[1]][k]

    def queue_action(self, queue, *args, **kwargs):
        """A method that acts on the queue the Agent is at. This method
        is called when the Agent arrives at the queue (where
        ``args[0] == 0``), when service starts for the Agent (where
        ``args[0] == 1``), and when the Agent departs from the queue
        (where ``args[0] == 2``). By default, this method does nothing
        to the queue, but is here if the Agent class is extended and
        this method is overwritten.
        """
        pass


class GreedyAgent(Agent):
    """An agent that chooses the queue with the shortest line as their
    next destination.

    Notes
    -----
    If there are any ties, the ``GreedyAgent`` chooses the first queue
    with the shortest line (where the ordering is given by
    :class:`QueueNetwork's<.QueueNetwork>` ``out_edges`` attribute).
    """
    def __init__(self, agent_id=(0, 0)):
        Agent.__init__(self, agent_id)

    def __repr__(self):
        msg = "GreedyAgent; agent_id:{0}. time: {1}"
        return msg.format(self.agent_id, round(self._time, 3))

    def desired_destination(self, network, edge):
        """Returns the agents next destination given their current
        location on the network.

        ``GreedyAgents`` choose their next destination with-in the
        network by picking the adjacent queue with the fewest number of
        :class:`Agents<.Agent>` in the queue.

        Parameters
        ----------
        network : :class:`.QueueNetwork`
            The :class:`.QueueNetwork` where the Agent resides.
        edge : tuple
            A 4-tuple indicating which edge this agent is located at.
            The first two slots indicate the current edge's source and
            target vertices, while the third slot indicates this edges
            ``edge_index``. The last slot indicates the edges edge
            type.

        Returns
        -------
        out : int
            Returns an the edge index corresponding to the agents next
            edge to visit in the network.
        """
        adjacent_edges = network.out_edges[edge[1]]
        d = _argmin([network.edge2queue[d].number_queued() for d in adjacent_edges])
        return adjacent_edges[d]


class InftyAgent(object):
    """An special agent that only operates within the
    :class:`.QueueServer` class.

    This agent never interacts with the :class:`.QueueNetwork`.
    """
    def __init__(self):
        self._time = infty

    def __repr__(self):
        return "InftyAgent"

    def __lt__(self, b):
        return self._time < b._time

    def __gt__(self, b):
        return self._time > b._time

    def __eq__(self, b):
        return self._time == b._time
