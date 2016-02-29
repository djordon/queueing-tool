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

            if 'eType' not in g.ep:
                eType = g.new_edge_property('int')
                eType.a = 1
                g.ep['eType'] = eType

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
                ep = self.g.ep
                for v in self.g.vertices():
                    tmp = {}
                    for e in v.out_edges():
                        tmp[int(e.target())] = {p: ep[e] for p in self.g.vp.keys()}

                    adj[int(v)] = tmp

            return adj

    def ep(self, e, edge_property):
        if self.is_nx_graph:
            return self.g.edge[e[0]][e[1]].get(edge_property)
        else:
            if edge_property not in self.g.ep:
                return None
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

    def set_ep(self, e, edge_property, value):
        if self.is_nx_graph:
            self.g.edge[e[0]][e[1]][edge_property] = value
        else:
            self.g.ep[edge_property][e] = value

    def set_vp(self, v, vertex_property, value):
        if self.is_nx_graph:
            self.g.node[v][vertex_property] = value
        else:
            self.g.vp[vertex_property][v] = value

    def graph_draw(self, **kwargs):
        if self.is_nx_graph:
            pass
        else:
            for key in kwargs.keys():
                if key in self.g.ep:
                    kwargs[key] = self.g.ep[key]

            gt.graph_draw(g, **kwargs)

    def get_window(self, **kwargs):
        if self.is_nx_graph:
            return None
        else:
            try:
                from gi.repository import Gtk, GObject
            except ImportError:
                msg = "Need gi.repository module for animating a graph_tool."
                raise ImportError(msg)

            window = gt.GraphWindow(g=self.g, **kwargs)
            return window, Gtk.main_quit, Gtk.main, GObject

    @property
    def vertex_properties(self):
        if self.is_nx_graph:
            props = set()
            for v in self.g.nodes():
                props.update(self.g.node[v].keys())
            return props
        else:
            return set(self.g.vertex_properties.keys())

    @property
    def edge_properties(self):
        if self.is_nx_graph:
            props = set()
            for e in self.g.edges():
                props.update(self.g.edge[e[0]][e[1]].keys())
            return props
        else:
            return set(self.g.edge_properties.keys())

    def new_vertex_property(self, name, property_type):
        if self.is_nx_graph:
            values = {v: None for v in self.g.nodes()}
            nx.set_node_attributes(self.g, name, values)
        else:
            self.g.vp[name] = g.new_vertex_property(property_type)

    def new_edge_property(self, name, property_type):
        if self.is_nx_graph:
            values = {v: None for v in self.g.edges()}
            nx.set_edge_attributes(self.g, name, values)
        else:
            self.g.ep[name] = g.new_edge_property(property_type)

    def set_pos(self):
        if self.is_nx_graph:
            pos = nx.spring_layout(self.g)
            nx.set_node_attributes(self.g, 'pos', pos)
        else:
            pos = gt.sfdp_layout(self.g, epsilon=1e-2, cooling_step=0.95)
            self.g.vp['pos'] = pos

    def is_edge(self, e):
        if self.is_nx_graph:
            return e in self.g.edge_index
        else:
            return self.g.edge(*e) is not None

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
