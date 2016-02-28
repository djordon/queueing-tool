import networkx as nx


nx2gt_attr = {
    'vertices': 'nodes',
    'num_vertices': 'number_of_nodes',
    'num_edges': 'number_of_edges',
}
gt2nx_attr = {value: key for key, value in nx2gt_attr.items()}


class GraphWrapper(object):
    gt2nx_attr = gt2nx_attr
    nx2gt_attr = nx2gt_attr

    def __init__(self, g):
        if isinstance(g, nx.DiGraph):
            self.is_nx_graph = True
            edge_index = {e: k for k, e in enumerate(g.edges())}
            setattr(g, 'edge_index', edge_index)
            nx.freeze(g)
        else:
            msg = "Must be given a networkx DiGraph or a graph-tool Graph"
            try:
                import graph_tool.all as gt
            except ImportError:
                raise ImportError(msg)

            if not isinstance(g, gt.Graph):
                raise TypeError(msg)

            g.reindex_edges()
            self.is_nx_graph = False
            setattr(g, 'successors', self._successors)
            setattr(g, 'predecessors', self._predecessors)
            setattr(g, 'out_degree', self._out_degree)
            setattr(g, 'out_edges', self._out_edges)
            setattr(g, 'in_edges', self._in_edges)

        self.g = g

    def __getattr__(self, attr):
        if attr in self.gt2nx_attr and not self.is_nx_graph:
            return getattr(self.g, self.gt2nx_attr[attr])
        elif attr in self.nx2gt_attr and self.is_nx_graph:
            return getattr(self.g, self.nx2gt_attr[attr])
        else:
            return getattr(self.g, attr)

    def _successors(self, v):
        v = self.g.vertex(v)
        return [int(e.target()) for e in v.out_edges()]

    def _predecessors(self, v):
        v = self.g.vertex(v)
        return [int(e.source()) for e in v.in_edges()]

    def _out_edges(self, v):
        v = self.g.vertex(v)
        return [(int(e.source()), int(e.target())) for e in v.out_edges()]

    def _in_edges(self, v):
        v = self.g.vertex(v)
        return [(int(e.source()), int(e.target())) for e in v.in_edges()]

    def _out_degree(self, v):
        v = self.g.vertex(v)
        return v.out_degree()

    def vertices(self, *args, **kwargs):
        if self.is_nx_graph:
            return self.g.nodes(*args, **kwargs)
        else:
            return [int(v) for v in self.g.vertices()]

    def graph_draw(self, **kwargs):
        if self.is_nx_graph:
            nx.draw_networkx(self.g, **kwargs)
        else:
            gt.graph_draw(g=self.g, **kwargs)

    def out_neighbours(self, v):
        if self.is_nx_graph:
            return [e[1] for e in self.g.out_edges(v)]
        else:
            v = self.g.vertex(v)
            return [int(u) for u in v.out_neighbours()]

    def edges(self, *args, **kwargs):
        if self.is_nx_graph:
            return self.g.edges(*args, **kwargs)
        else:
            return [(int(e.source()), int(e.target())) for e in self.g.edges()]

    def graph2dict(self):
        """Takes a graph and returns an adjacency list.

        Returns
        -------
        adj : :class:`.dict`
            An adjacency representation of graph as a dictionary of dictionaries,
            where a key is the vertex index for a vertex ``v`` and the
            values are :class:`.list`\s of vertex indices where that vertex is
            connected to ``v`` by an edge.
        """
        if self.is_nx_graph:
            return nx.to_dict_of_dicts(self.g)
        else:
            adj = {}
            if 'eType' not in g.ep:
                for v in self.g.vertices():
                    adj[int(v)] = {int(u): {} for u in v.out_neighbours()}
            else:
                et = self.g.ep['eType']
                for v in self.g.vertices():
                    adj[int(v)] = {int(e.target()): {'eType': et[e]} for e in v.out_edges()}

            return adj

    def ep(self, e, edge_property):
        if self.is_nx_graph:
            return self.g.edge[e[0]][e[1]][edge_property]
        else:
            return self.g.ep[edge_property][e]

    def vp(self, v, vertex_property):
        if self.is_nx_graph:
            return self.g.node[v].get(vertex_property)
        else:
            if vertex_property not in self.g.vp:
                return None
            else:
                return self.g.vp[vertex_property][v]

    @property
    def vertex_properties(self):
        if self.is_nx_graph:
            return self.g.node[0]
        else:
            self.g.vertex_properties

    def new_edge_property(self):
        pass

    def new_edge_property(self):
        pass



def __create_graph():
    vs  = g.add_vertex(nV)

    g.set_directed(is_directed)

    eT = g.new_edge_property("int")

    for u, adj in adjacency.items() :
        if is_directed :
            for j, v in enumerate(adj) :
                e = g.add_edge(u, v)
                eT[e] = eType[u][j]
        else :
            for j, v in enumerate(adj) :
                if len(g.edge(u,v,True)) < adj.count(v) :
                    e = g.add_edge(u, v)
                    eT[e] = eType[u][j]

    g.ep['eType'] = eT
