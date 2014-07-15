from get_destinations       import get_dests
import networkx             as nx
import numpy                as np
import graph_tool.all       as gt
import matplotlib.pyplot    as plt
import pickle
import osm2nx

l   = ['A Plus Gas Station', 'jewelry', 'turning_circle', 'Z-Best Wings',
       'Herron Station', 'parking_entrance', 'restaurant', "Butch's Baber Shop", 
       'fuel', 'stop', 'hairdresser']

s   = ['1B-C', '70A', '6C (old 10)', '1A', 'A Plus Gas Station', '1C', 'no', 
        '7A-B (old 11A-B)', 'jewelry', 'yes', '1A (old 1)', '1D (old 4)', '70B', 
        '69C', '7A (old 11B)', 'uncontrolled', '71A', 
        'Z-Best Wings', '7B (old 11)', '70C', '1C (old 3)', 
        'US 19, PA 65 North; Ohio River Blvd', '1B', '7B (old 11A)', 
        '71B', '5C (old 7)', 'Herron Station', 'clean', 'North Shore', 'bus_station', 
        '70D', 'parking_entrance', 'restaurant', "Butch's Baber Shop", 'fuel', 
        '6B (old 9)', 'stop', '1B (old 2; older 1)', '2', 'hairdresser', '6A (old 8)']

g.vp['pos'] = pos
G = osm2nx.read_osm('./cities/pittsburgh_downtown.osm')
for k in G.node.keys():
    G.node[k]['destination']    = 0
    G.node[k]['garage']         = 0
    G.node[k]['light']          = 0
    G.node[k]['cfcc']           = 0
    G.node[k]['lanes']          = 1
    G.node[k]['cap']            = 0
    if 'highway' in G.node[k]['data'].tags :
        highway = G.node[k]['data'].tags['highway']
        if highway == 'traffic_signals' :
            G.node[k]['light'] = 1
        if highway == 'crossing' and 'crossing' in G.node[k]['data'].tags :
            if G.node[k]['data'].tags['crossing'] == 'traffic_signals' :
                G.node[k]['light'] = 2


for key1 in G.edge.keys() :
    for key2 in G.edge[key1].keys() :
        try :
            if 'tiger:cfcc' in G.edge[key1][key2]['data'].tags :
                G.node[key2]['cfcc'] = 1
            if 'lanes' in G.edge[key1][key2]['data'].tags :
                G.node[key1]['lanes'] = int(G.edge[key1][key2]['data'].tags['lanes'])
        except TypeError : 
            continue


for key1 in G.edge.keys() :
    for key2 in G.edge[key1].keys() :
        if key2 == 'cfcc' or key2 == 'lanes' :
            continue
        try :
            del G.edge[key1][key2]['data']
        except KeyError or TypeError :
            try :
                del G.edge[key1][key2]['id']
            except KeyError or TypeError :
                continue


for key, value in G.node.items() :
    G.node[key]['lat']  = value['data'].lat
    G.node[key]['lon']  = value['data'].lon
    del G.node[key]['data']

g_dict, d_dict = get_dests()
c   = 0

for k in range(len(g_dict)) :
    G.add_node(c, g_dict[k])
    c  += 1

for k in range(len(d_dict)) :
    G.add_node(c, d_dict[k])
    c  += 1

nx.write_graphml(G, 'downtown_pitt.xml')




## Using graph_tool now
g   = gt.load_graph('downtown_pitt.xml', fmt='xml')

g.set_directed(True)
g2  = g.copy()
for e in g2.edges() :
    target1 = int(e.target())
    source1 = int(e.source())
    e1      = g.add_edge(source=target1, target=source1)


keep    = g.new_vertex_property('bool')
for v in g.vertices() :
    if g.vp['cfcc'][v] :
        keep[v] = True
    else :
        keep[v] = False

g.set_vertex_filter(keep, inverted=False)
g.purge_vertices()


def remove_problem_points() :
    latlong = [[40.442696, -79.9871978], [40.441443, -79.9849928], 
               [40.4469285, -80.0038217], [40.447944, -80.0003869], 
               [40.4326597, -80.0035281], [40.4404269, -79.9912726],
               [40.4405301, -79.9913136], [40.4366812, -80.0009142]]
    to_remove   = True
    while to_remove :
        g2  = g.copy()
        to_remove   = False
        for v in g2.vertices() :
            for latlon in latlong :
                if g.vp['lat'][v] == latlon[0] and g.vp['lon'][v] == latlon[1] :
                    to_remove   = True
                    w   = g.vertex(int(v))
                    g.remove_vertex(w)
                    break
            if to_remove :
                break


def remove_tails() :
    to_remove   = True
    while to_remove :
        g2  = g.copy()
        to_remove   = False
        for v in g2.vertices() :
            a   = not g2.vp['destination'][v] == 1
            b   = not g2.vp['garage'][v] == 1
            if v.out_degree() <= 1 and a and b:
                to_remove   = True
                w   = g.vertex(int(v))
                g.remove_vertex(w)
                break


remove_problem_points()
remove_tails()

comp        = gt.label_components(g)
components  = g.new_vertex_property('bool')

s   = np.argmax([np.sum(comp[0].a == k) for k in range(np.max(comp[0].a))])
for v in g.vertices() :
    if comp[0][v] == s or g.vp['destination'][v] == 1 or g.vp['garage'][v] == 1 :
        components[v] = True

g.set_vertex_filter(components, inverted=False)
g.purge_vertices()


destination = g.new_vertex_property('bool')
garage      = g.new_vertex_property('bool')
light       = g.new_vertex_property('bool')

for v in g.vertices() :
    destination[v]  = True if g.vp['destination'][v] else False
    garage[v]       = True if g.vp['garage'][v] else False
    light[v]        = True if g.vp['light'][v]  else False


pos         = g.new_vertex_property('vector<double>')
col         = g.new_vertex_property('vector<double>')
fill        = g.new_vertex_property('vector<double>')
vsize       = g.new_vertex_property('double')
vpen        = g.new_vertex_property('double')
cap         = g.new_vertex_property('int')


for v in g.vertices() :
    pos[v]          = [g.vp['lat'][v], g.vp['lon'][v]]
    vpen[v]         = 0.6
    vsize[v]        = 6.5
    fill[v]         = [0.0, 0.5, 1.0, 1.0]
    col[v]          = [0.0, 0.5, 1.0, 1.0]
    if light[v] :
        fill[v]     = [1.0, 0.135, 0.0, 1.0]
        col[v]      = [1.0, 0.271, 0.0, 1.0]
    if garage[v] :
        col[v]      = [0.133, 0.545, 0.133, 1.0]
        fill[v]     = [0.133, 0.545, 0.133, 1.0]
    if destination[v] :
        col[v]      = [0.282, 0.239, 0.545, 1.0]
        fill[v]     = [0.282, 0.239, 0.545, 1.0]

g.vp['light']       = light
g.vp['destination'] = destination
g.vp['garage']      = garage

g.vp['pos']         = pos
g.vp['col']         = col
g.vp['fill']        = fill
g.vp['vsize']       = vsize
g.vp['vpen']        = vpen

edge_color          = g.new_edge_property('vector<double>')
control             = g.new_edge_property('vector<double>')
edge_width          = g.new_edge_property('double')
arrow_width         = g.new_edge_property('double')


for e in g.edges() :
    edge_color[e]   = [0.339, 0.3063, 0.3170, 0.390]
    edge_width[e]   = 1.25
    arrow_width[e]  = 6
    control[e]      = [0, 0, 0, 0]


g.ep['edge_color']  = edge_color
g.ep['edge_width']  = edge_width
g.ep['arrow_width'] = arrow_width
g.ep['control']     = control


#g.save('downtown_pitt.xml', fmt='xml')

"""
gt.graph_draw(g, pos=g.vp['pos'],
    vertex_fill_color=g.vp['fill'], 
    vertex_size=g.vp['vsize'], 
    vertex_pen_width=g.vp['vpen'],
    edge_color=g.ep['edge_color'],
    edge_control_points=g.ep['control'],
    edge_marker_size=g.ep['arrow_width'],
    edge_pen_width=g.ep['edge_width'])





from get_destinations       import get_dests
from gistfile1              import *
import approximate_DP       as adp
import numpy                as np
import graph_tool.all       as gt
import matplotlib.pyplot    as plt
import pickle
"""


#g   = gt.load_graph('downtown_pitt.xml', fmt='xml')

def calculate_distance( latlon1, latlon2 ) :
    lat1, lon1  = latlon1
    lat2, lon2  = latlon2
    R           = 6371          # radius of the earth in kilometers
    dlon        = lon2 - lon1
    dlat        = lat2 - lat1
    a           = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c           = 2 * np.pi * R * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) / 180
    return c


edge_length = g.new_edge_property('double')

for e in g.edges() :
    latlon1         = g.vp['pos'][e.target()]
    latlon2         = g.vp['pos'][e.source()]
    edge_length[e]  = np.round( calculate_distance( latlon1, latlon2 ), 3) 

g.ep['edge_length'] = edge_length


dists   = np.ones( (g.num_vertices(), g.num_vertices()) ) * np.infty
for v in g.vertices() :
    if g.vp['garage'][v] or g.vp['destination'][v] :
        for v2 in g.vertices() :
            if not g.vp['garage'][v2] and not g.vp['destination'][v2] :
                latlon1 = g.vp['pos'][v]
                latlon2 = g.vp['pos'][v2]
                d       = np.round( calculate_distance( latlon1, latlon2 ), 3)
                dists[int(v), int(v2)] = d

mnd = np.min(dists)
g2  = g.copy()
for v in g2.vertices() :
    if g.vp['garage'][v] or g.vp['destination'][v] :
        vi      = int(v)
        d       = dists[vi, :]
        n       = np.argmin(d)
        e1      = g.add_edge(source=vi, target=n )
        e2      = g.add_edge(source=n,  target=vi)

        g.ep['edge_length'][e1] = d[n]
        g.ep['edge_length'][e2] = d[n]

        d[n]    = np.infty
        n       = np.argmin(d)
        e1      = g.add_edge(source=vi, target=n )
        e2      = g.add_edge(source=n,  target=vi)

        g.ep['edge_length'][e1] = d[n]
        g.ep['edge_length'][e2] = d[n]

        if g.vp['garage'][v] :
            e1  = g.add_edge(source=vi, target=vi)
            g.ep['edge_length'][e1] = mnd / 10






keep_removing = True
while keep_removing :
    keep_removing = False
    g2  = g.copy()
    for v in g2.vertices() :
        if g2.vp['garage'][v] or g2.vp['destination'][v] :
            for e in v.out_edges() :
                if g.ep['edge_length'][e] > 0.36 :
                    g.remove_vertex(int(v))
                    keep_removing = True
                    break
        if keep_removing :
            break


#edge_length     = g.new_edge_property('double')
edge_t_size     = g.new_edge_property('double')
edge_t_distance = g.new_edge_property('double')
edge_t_parallel = g.new_edge_property('bool')
destination     = g.new_edge_property('bool')
garage          = g.new_edge_property('bool')
#edge_length_t   = g.new_edge_property('string')


s = set()
for e in g.edges() :
    if g.vp['garage'][e.target()] and g.vp['garage'][e.source()]:
        garage[e]       = True
        edge_length[e]  = mnd / 10
    if g.vp['destination'][e.target()]:
        destination[e]  = True
        latlon1         = g.vp['pos'][e.target()]
        latlon2         = g.vp['pos'][e.source()]
        edge_length[e]  = np.round( calculate_distance( latlon1, latlon2 ), 3)
    
    a   = (int(e.source()), int(e.target()))
    b   = (a[1], a[0])
    if b in s :
        continue
    else :
        s   = s.union([a])
    edge_length_t[e] = str(edge_length[e])


g.ep['garage']          = garage
g.ep['destination']     = destination


del g.vp['_graphml_vertex_id'], g.vp['vsize'], g.vp['col']
del g.vp['lat'], g.vp['lon'], g.vp['cfcc'], g.vp['vpen']
del g.vp['id'], g.vp['url']
del g.ep['arrow_width'], g.ep['edge_color']
del g.ep['edge_width'], g.ep['control']
del g.ep['_graphml_edge_id']

a,b = -15,32
rotation_m  = np.zeros( (2,2) )
rotation_m[0,0] =  np.cos( a*np.pi/b )
rotation_m[0,1] = -np.sin( a*np.pi/b )
rotation_m[1,0] =  np.sin( a*np.pi/b )
rotation_m[1,1] =  np.cos( a*np.pi/b )

for v in g.vertices() :
    g.vp['pos'][v] = np.dot( rotation_m, g.vp['pos'][v] )

g.save('downtown_pitt.xml', fmt='xml')






### Initializing this object is very time consuming
import graph_tool.all       as gt
import queue_network        as qn
import queue_server         as qs
myq = qn.Queue_network(ab="downtown_pitt.xml")

for v in myq.g.vertices() :
    myq.g.vp['pos'][v] = np.dot( rotation_m, myq.g.vp['pos'][v] )

### This is needed to save the graph. The lambda functions are not pickle-able
queues              = myq.g.new_edge_property("bool")
myq.g.ep['queues']  = queues
myq.g.save('pitt_network.xml', fmt='xml')
del myq.edges, myq.queue_heap


import pickle
directory   = '/home/dan/math/code/python/queueing/'
pickle.dump( myq, open(directory+'pitt_network.p', 'wb') )






### Load the graph
import graph_tool.all       as gt
import queue_network        as qn
import queue_server         as qs
import pickle
directory   = '/home/dan/math/code/python/queueing/'
myq = pickle.load(open(directory+'pitt_network.p', 'rb') )
que = qn.Queue_network( myq.g )


### This saved the distances after loading an adp
import approximate_DP   as adp
import graph_tool.all   as gt
import queue_network    as qn
import queue_server     as qs
import pickle

directory   = '/home/dan/math/code/python/queueing/'
pit     = pickle.load(open(directory+'pitt_network.p', 'rb') )
a       = adp.approximate_dynamic_program(pit.g)
dist    = a.Qn.g.new_vertex_property("vector<double>")

for v in a.Qn.g.vertices() :
    dist[v] = a.dists[int(v),:]

a.Qn.g.vp['dist']   = dist
queues              = a.Qn.g.new_edge_property("bool")
a.Qn.g.ep['queues'] = queues
del a.Qn.edges, a.Qn.queue_heap

pickle.dump( a.Qn, open(directory+'pitt_network.p', 'wb') )


"""
http://www.itoworld.com/map/7?lon=-79.99940&lat=40.44282&zoom=15
http://metro.teczno.com
https://help.openstreetmap.org/questions/30692/export-only-roads
"""
e_props = set()
for key in pit.g.edge_properties.keys() :
    e_props = e_props.union( [key] )


