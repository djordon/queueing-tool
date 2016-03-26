import networkx as nx
import numpy as np

from queueing_tool.graph.graph_wrapper import QueueNetworkDiGraph



def _test_graph(g) :
    """A function that makes sure ``g`` is either a
    :any:`networkx.DiGraph` or a string or file object to one.

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


def graph2dict(g, return_dict_of_dict=True) :
    """Takes a graph and returns an adjacency list.

    Parameters
    ----------
    g : :any:`networkx.DiGraph`, :any:`networkx.Graph`, etc.
        Any object that networkx can turn into a
        :any:`DiGraph<networkx.DiGraph>`.
    return_dict_of_dict : bool (optional, default: True)
        Specifies whether this function will return a dict of dicts
        or a dict of lists.

    Returns
    -------
    adj : dict
        An adjacency representation of graph as a dictionary of
        dictionaries, where a key is the vertex index for a vertex
        ``v`` and the values are :class:`dicts<.dict>` with keys for
        the vertex index and values as edge properties.
    """
    if not isinstance(g, nx.DiGraph):
        g = QueueNetworkDiGraph(g)

    dict_of_dicts = nx.to_dict_of_dicts(g)
    if return_dict_of_dict:
        return dict_of_dicts
    else:
        return  {k: list(val.keys()) for k, val in dict_of_dicts.items()}
