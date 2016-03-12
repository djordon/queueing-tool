import networkx as nx
import numpy as np

from queueing_tool.graph.graph_wrapper import QueueNetworkDiGraph



def _test_graph(g) :
    """A function that makes sure ``g`` is either a :class:`~graph_tool.Graph` or
     a string or file object to one.

    Parameters
    ----------
    g : A **str** or a :class:`~graph_tool.Graph`.

    Returns
    -------
    :class:`~graph_tool.Graph`
        If ``g`` is a string or a file object then the output given by
        ``graph_tool.load_graph(g, fmt='xml')``, if ``g`` is aready a
        :class:`~graph_tool.Graph` then it is returned unaltered.

    Raises
    ------
    TypeError
        Raises a :exc:`~TypeError` if ``g`` is not a string to a file object,
        or a :class:`~graph_tool.Graph`\.
    """
    if not isinstance(g, QueueNetworkDiGraph):
        if not isinstance(g, nx.DiGraph):
            try:
                import graph_tool.all as gt
            except ImportError:
                msg = ("Graph given was not a networkx DiGraph or graph_tool "
                       "graph.")
                raise ImportError(msg)
            if not isinstance(g, gt.Graph):
                msg = "Need to supply a graph-tool Graph or networkx DiGraph"
                raise TypeError(msg)

        g = QueueNetworkDiGraph(g)
    return g


def graph2dict(g) :
    """Takes a graph and returns an adjacency list.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`

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

def graph_tool_graph2dict(g):
        adj = {}
        vp = g.vp
        for v in g.nodes():
            tmp = {}
            for u in v.out_neighbours():
                tmp[int(u)] = {p: vp[p][u] for p in vp.keys()}

            adj[int(v)] = tmp

        ep = g.ep
        for v in g.nodes():
            tmp = {}
            for e in v.out_edges():
                tmp[int(e.target())] = {p: ep[p][e] for p in g.ep.keys()}

            adj[int(v)] = tmp

        return adj
