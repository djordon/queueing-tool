import networkx as nx
import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plt.style.use('ggplot')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

gt2nx_attr = {
    'vertices': 'nodes',
    'num_vertices': 'number_of_nodes',
    'num_edges': 'number_of_edges',
}


class GraphWrapper(object):
    gt2nx_attr = gt2nx_attr

    def __init__(self, g):
        if not isinstance(g, nx.DiGraph):
            msg = "Must be given a networkx DiGraph."
            raise TypeError(msg)

        edge_index = {e: k for k, e in enumerate(g.edges())}
        setattr(g, 'edge_index', edge_index)

        self.g = g
        self.pos = None
        self.edge_color = None
        self.vertex_color = None
        self.vertex_fill_color = None


    def freeze(self):
        edge_index = {e: k for k, e in enumerate(self.g.edges())}
        setattr(self.g, 'edge_index', edge_index)
        nx.freeze(self.g)


    def __getattr__(self, attr):
        if attr in self.gt2nx_attr:
            return getattr(self.g, self.gt2nx_attr[attr])
        else:
            return getattr(self.g, attr)


    def _draw(self, **kwargs):
        nx.draw_networkx(self.g, **kwargs)


    def vertices(self, *args, **kwargs):
        return self.g.nodes(*args, **kwargs)


    def out_neighbours(self, v):
        return [e[1] for e in self.g.out_edges(v)]


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
        return nx.to_dict_of_dicts(self.g)


    def ep(self, e, edge_property):
        return self.g.edge[e[0]][e[1]].get(edge_property)


    def vp(self, v, vertex_property):
        return self.g.node[v].get(vertex_property)


    def set_ep(self, e, edge_property, value):
        self.g.edge[e[0]][e[1]][edge_property] = value
        if hasattr(self, edge_property):
            attr = getattr(self, edge_property)
            attr[self.edge_index[e]] = value


    def set_vp(self, v, vertex_property, value):
        self.g.node[v][vertex_property] = value
        if hasattr(self, vertex_property):
            attr = getattr(self, vertex_property)
            attr[v] = value


    @property
    def vertex_properties(self):
        props = set()
        for v in self.g.nodes():
            props.update(self.g.node[v].keys())
        return props


    @property
    def edge_properties(self):
        props = set()
        for e in self.g.edges():
            props.update(self.g.edge[e[0]][e[1]].keys())
        return props


    def new_vertex_property(self, name):
        values = {v: None for v in self.g.nodes()}
        nx.set_node_attributes(self.g, name, values)
        if name == 'vertex_color':
            self.vertex_color = [0 for v in range(self.number_of_nodes())]
        if name == 'vertex_fill_color':
            self.vertex_fill_color = [0 for v in range(self.number_of_nodes())]


    def new_edge_property(self, name):
        values = {v: None for v in self.g.edges()}
        nx.set_edge_attributes(self.g, name, values)
        if name == 'edge_color':
            self.edge_color = np.zeros((self.number_of_edges(), 4))


    def set_pos(self, pos=None):
        if pos is None:
            pos = nx.spring_layout(self.g)
        nx.set_node_attributes(self.g, 'pos', pos)
        self.pos = np.array([pos[v] for v in self.g.nodes()])


    def draw_graph(self, **kwargs):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required to draw the graph.")

        fig = plt.figure(figsize=kwargs.get('figsize', (7, 7)))
        ax  = fig.gca()

        lines_kwargs, scatter_kwargs = self._lines_scatter_args(ax, **kwargs)

        edge_collection = LineCollection(**lines_kwargs)
        ax.add_collection(edge_collection)
        ax.scatter(**scatter_kwargs)

        ax.set_axis_bgcolor(kwargs.get('bgcolor', [1, 1, 1, 1]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.ion()
        plt.show()
        plt.ioff()


    def _lines_scatter_args(self, ax, **kwargs):

        edge_pos = [(self.pos[e[0]], self.pos[e[1]]) for e in self.g.edges()]
        line_collecton_kwargs = {
            'segments': edge_pos,
            'colors': self.edge_color,
            'linewidths': (1,),
            'antialiaseds': (1,),
            'linestyle': 'solid',
            'transOffset': ax.transData,
            'cmap': plt.cm.ocean_r,
            'pickradius': 5,
            'zorder': 2,
            'facecolors': None,
            'norm': None
        }
        scatter_kwargs = {
            'x': self.pos[:, 0],
            'y': self.pos[:, 1],
            's': 100,
            'c': self.vertex_fill_color,
            'alpha': None,
            'norm': None,
            'vmin': None,
            'vmax': None,
            'marker': 'o',
            'cmap': plt.cm.ocean_r,
            'linewidths': 1,
            'edgecolors': self.vertex_color
        }

        for key, value in kwargs.items():
            if key in line_collecton_kwargs:
                line_collecton_kwargs[key] = value
            if key in scatter_kwargs:
                scatter_kwargs[key] = value

        return line_collecton_kwargs, scatter_kwargs


    def is_edge(self, e):
        return e in self.g.edge_index
