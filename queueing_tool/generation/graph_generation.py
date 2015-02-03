import graph_tool.all as gt
import numpy          as np
import copy

from .graph_preparation import set_types_random, set_types_pagerank

def matrix2list(matrix) :
    n   = len(matrix)
    adj = [ [] for k in range(n)]
    for k in range(n) :
        for j in range(n) :
            if matrix[k, j] :
                adj[k].append(j)

    return adj


def ematrix2list(adjacency, matrix) :
    n   = len(matrix)
    adj = [ [] for k in range(n)]
    for k in range(n) :
        for j in adjacency[k] :
            adj[k].append(int(matrix[k, j]))

    return adj


def adjacency2edgetype(adjacency) :
    n   = len(adjacency)
    ety = [ [] for k in range(n)]
    if not isinstance(adjacency, list) :
        adj = [ [] for k in range(n)]
        for k in range(n) :
            for j in range(n) :
                if adjacency[k, j] :
                    adj[k].append(j)
    else :
        adj = adjacency

    for k in range(n) :
        for j in adj[k] :
            ety[k].append( 1 if len(adj[j]) else 0 )

    return ety


def adjacency2graph(adjacency, edge_types=None, edge_lengths=None) :

    nV  = len(adjacency)
    g   = gt.Graph()
    vs  = g.add_vertex(nV)

    if not isinstance(adjacency, list) :
        adjacency = matrix2list(adjacency)

    for k in range(nV) :
        adj = adjacency[k]
        for j in adj :
            e = g.add_edge(k, j)

    if edge_types is not None and not isinstance(edge_types, list) :
        edge_types  = ematrix2list(adjacency, edge_types)

    if edge_types is None :
        edge_types = copy.deepcopy(adjacency)
        for k, adj in enumerate(adjacency) :
            edge_types[k] = [1 for j in range(len(adj))]


    if edge_lengths is None :
        edge_lengths = copy.deepcopy(adjacency)
        for k, adj in enumerate(adjacency) :
            edge_lengths[k] = [1 for j in range(len(adj))]

    vType   = g.new_vertex_property("int")
    eType   = g.new_edge_property("int")
    elength = g.new_edge_property("double")

    for u, adj in enumerate(adjacency) :
        for j, v in enumerate(adj) :
            e = g.edge(u, v) 
            eType[e]    = edge_types[u][j]
            elength[e]  = edge_lengths[u][j]
            if u == v :
                vType[e.source()] = edge_types[u][j]

    g.vp['vType'] = vType
    g.ep['eType'] = eType
    g.ep['edge_length'] = elength
    return g
    

def generate_random_graph(nVertices=250, **kwargs) :
    g = random_graph(nVertices)
    g = set_types_random(g, **kwargs)
    return g


def generate_pagerank_graph(nVertices=250, **kwargs) :
    g = random_graph(nVertices)
    g = set_types_pagerank(g, **kwargs)
    return g


def random_graph(nVertices, directed=True) :
    """Creates a connected and directed random graph.

    The function creates a minimally connected random graph. To do so, 
    it places vertices randomly on the unit square and each vertex v 
    adds an edge to all other vertices within a certain radius ``r``.
    The radius ``r`` is taken to be as small as possible.
    """
    points  = np.random.random((nVertices, 2)) * 2
    radii   = [(4 + k) / 200 for k in range(560)]
    
    for r in radii :
        g, pos  = gt.geometric_graph(points, r, [(0,2), (0,2)])
        comp, a = gt.label_components(g)
        if max(comp.a) == 0 :
            break

    if directed :
        g.set_directed(True)

    g2  = g.copy()
    for e in g2.edges() :
        e1  = g.add_edge(source=int(e.target()), target=int(e.source()))

    g.reindex_edges()
    pos       = gt.sfdp_layout(g, epsilon=1e-2, cooling_step=0.95)
    pos_array = np.array([pos[v] for v in g.vertices()])
    pos_array = pos_array / (100*(np.max(pos_array,0) - np.min(pos_array,0)))
    
    for v in g.vertices() :
        pos[v]  = pos_array[int(v), :]
    
    g.vp['pos'] = pos
    return g



nVertices = 30
dist_dict = {}
dist_list = []
blobs     = {k : {k} for k in range(nVertices)}
points    = np.random.random((nVertices, 2)) * 10

for k in range(nVertices-1) :
    for j in range(k+1, nVertices) :
        d = np.linalg.norm(points[k] - points[j])
        dist_dict[d] = (k, j)
        dist_list.append(d)

dist_list.sort()
vert = set()
for k, d in enumerate(dist_list) :
    if len(vert) == nVertices - 2:
        break
    j, l = dist_dict[d]
    if j in vert and l in vert :
        continue
    s    = blobs[j].union(blobs[l])
    for j in s :
        blobs[j] = s
    vert.update( (j,l) )

g, pos  = gt.geometric_graph(points, d)
comp, a = gt.label_components(g)
max(comp.a)

g, pos  = gt.geometric_graph(points, d-0.3)
comp, a = gt.label_components(g)
max(comp.a)
(1) Creating object for path '/org/freedesktop/NetworkManager/ActiveConnection/5' failed in libnm-glib.
