import networkx as nx
import numpy as np

from queueing_tool.graph.graph_wrapper import QueueNetworkDiGraph



def _test_graph(g) :
    """A function that makes sure ``g`` is either a :any:`networkx.DiGraph` or
     a string or file object to one.

    Parameters
    ----------
    g : A **str** or a :any:`networkx.DiGraph`.

    Returns
    -------
    :class:`.QueueNetworkDiGraph`
        

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` cannot be turned into a
        :any:`networkx.DiGraph`.
    """
    if not isinstance(g, QueueNetworkDiGraph):
        try:
            g = QueueNetworkDiGraph(g)
        except (nx.NetworkXError, TypeError):
            raise TypeError("Couldn't turn g into a graph.")
    return g


def graph2dict(g) :
    """Takes a graph and returns an adjacency list.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :any:`networkx.Graph`, etc.
        Any object that can be instantiated as a :any:`networkx.DiGraph`.

    Returns
    -------
    adj : :class:`.dict`
        An adjacency representation of graph as a dictionary of dictionaries,
        where a key is the vertex index for a vertex ``v`` and the
        values are :class:`.list`\s of vertex indices where that vertex is
        connected to ``v`` by an edge.
    """
    if not isinstance(g, nx.DiGraph):
        g = QueueNetworkDiGraph(g)

    return nx.to_dict_of_dicts(g)
