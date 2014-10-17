import graph_tool.all as gt
import numpy          as np

def active_graph(self, nVertices=250, pDest=None, pGarage=None) :

    points  = np.random.random((nVertices, 2)) * 2
    radii   = [(4 + k) / 200 for k in range(560)]

    for r in radii :
        g, pos  = gt.geometric_graph(points, r, [(0,2), (0,2)])
        comp, a = gt.label_components(g)
        if max(comp.a) == 0 :
            break

    pos       = gt.sfdp_layout(g, epsilon=1e-2, cooling_step=0.95)
    pos_array = np.array([pos[v] for v in g.vertices()])
    pos_array = pos_array / (100*(np.max(pos_array,0) - np.min(pos_array,0)))

    for v in g.vertices() :
        pos[v]  = pos_array[int(v), :]

    g.vp['pos'] = pos
    g = set_special_nodes(g, pDest, pGarage)
    return g


def set_special_nodes(self, g, pDest=None, pFCQ=None) :
    if pDest == None :
        pDest = 0.1
    if pFCQ == None :
        pFCQ = 1
    pDest = pDest * 100

    def calculate_distance(latlon1, latlon2) :
        lat1, lon1  = latlon1
        lat2, lon2  = latlon2
        R     = 6371          # radius of the earth in kilometers
        dlon  = lon2 - lon1
        dlat  = lat2 - lat1
        a     = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
        c     = 2 * np.pi * R * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) / 180
        return c

    pagerank    = gt.pagerank(g)
    tmp         = np.array(pagerank.a)
    tmp.sort()
    nDests      = int(np.ceil(g.num_vertices()/pDest))
    dests       = np.where(pagerank.a >= tmp[-nDests])[0]

    dest_pos    = np.array([g.vp['pos'][g.vertex(k)] for k in dests])
    nFCQ        = int(pFCQ * np.size(dests))
    min_g_dist  = np.ones(nFCQ) * infty
    ind_g_dist  = np.ones(nFCQ, int)

    r, theta    = np.random.random(nFCQ) / 500, np.random.random(nFCQ) * 360
    xy_pos      = np.array([r * np.cos(theta), r * np.sin(theta)]).transpose()
    g_pos       = xy_pos + dest_pos[ np.array( np.mod(np.arange(nFCQ), nDests), int) ]

    for v in g.vertices() :
        if int(v) not in dests :
            tmp = np.array([calculate_distance(g.vp['pos'][v], g_pos[k, :]) for k in range(nFCQ)])
            min_g_dist = np.min((tmp, min_g_dist), 0)
            ind_g_dist[min_g_dist == tmp] = int(v)

    ind_g_dist  = np.unique(ind_g_dist)
    fcqs        = ind_g_dist[:min( (nFCQ, len(ind_g_dist)) )]

    if not g.is_directed() :
        g.set_directed(True)

        g2  = g.copy()
        for e in g2.edges() :
            e1  = g.add_edge(source=int(e.target()), target=int(e.source()))

    vType   = g.new_vertex_property("int")
    eType   = g.new_edge_property("int")
    elength = g.new_edge_property("double")

    for e in g.edges() :
        latlon1     = g.vp['pos'][e.target()]
        latlon2     = g.vp['pos'][e.source()]
        elength[e]  = np.round(calculate_distance(latlon1, latlon2), 3)

    for v in g.vertices() :
        if int(v) in dests :
            vType[v] = 2
        if int(v) in fcqs :
            vType[v] = 1
        e = g.add_edge(source=v, target=v)
        if vType[v] == 1 :
            eType[e] = 1

    g.vp['vType'] = vType
    g.ep['eType'] = eType
    return g

