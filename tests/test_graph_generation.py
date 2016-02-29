import numpy    as np
import queueing_tool  as qt
#import graph_tool.all as gt
import unittest
import numbers

from numpy.random import randint


def generate_adjacency(a=3, b=25, c=6, n=12) :
    return {k : list(randint(a, b, randint(1,c))) for k in np.unique(randint(a, b, n))}



class TestGraphFunctions(unittest.TestCase) :

    def test_graph2dict(self) :
        adj = generate_adjacency()
        g1  = qt.adjacency2graph(adj,adjust=1)
        aj1 = qt.graph2dict(g1)
        g2  = qt.adjacency2graph(aj1[0],adjust=1)
        self.assertTrue( gt.isomorphism(g1, g2) )


    def test_add_edge_lengths(self) :
        g   = qt.generate_pagerank_graph(10)
        g2  = qt.add_edge_lengths(g)

        edge_props = set()
        for key in g.edge_properties.keys() :
            edge_props.add(key)

        self.assertTrue('edge_length' in edge_props)


    def test_generate_transition(self) :
        g   = qt.generate_random_graph(20)
        mat = qt.generate_transition_matrix(g)

        ans = np.sum(mat, axis=1)
        self.assertTrue( np.allclose(ans, 1) )


    def test_adjacency2graph(self) :

        # Test adjacency argument using ndarray work
        adj = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        eTy = {0 : 5, 1 : [9, 14]}
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=1)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[0] == {0 : [1], 1 : [2, 3], 2 : [], 3 : []})

        # Test adjacency argument using ndarrays work
        adj = {0 : [1], 1 : [2, 3]}
        ety = np.array([[0, 5, 0, 0],
                        [0, 0, 9, 14],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=0)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [9, 14], 2 : [0], 3 : [0]})

        # Test adjacency argument types dict
        adj = {0 : 1, 1 : [2, 3]}
        eTy = [5, [9, 14]]
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=1)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [0, 0], 2 : [], 3 : []})

        # Test adjacency argument types list
        adj = [1, [2, 3]]
        eTy = [[5], [9, 14]]
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=1)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [0, 0], 2 : [], 3 : []})

        # Test edge types adjust 1 and adjacency argument types list
        adj = [[1], [2, 3]]
        eTy = {0 : 5, 1 : [9, 14]}
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=1)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [0, 0], 2 : [], 3 : []})

        # Test edge types adjust 0
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=0)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [9, 14], 2 : [0], 3 : [0]})

        adj = generate_adjacency()
        g1  = qt.adjacency2graph(adj, adjust=1)
        ans = qt.graph2dict(g1)

        vertices = set()
        for key, value in adj.items() :
            vertices.add(key)
            if isinstance(value, numbers.Integral) :
                vertices.add(value)
            else :
                vertices.update(value)

        for v in vertices :
            if v not in adj :
                adj[v] = []

        m   = min(vertices)
        aj2 = {key-m : [v-m for v in value] for key,value in adj.items()}
        g2  = qt.adjacency2graph(aj2, adjust=1)
        self.assertTrue( gt.isomorphism(g1, g2) )

    @unittest.skip('lazy')
    def test_set_types_random(self) :

        nV = 4000
        nT = np.random.randint(5, 10)

        g = nx.random_geometric_graph(nV, 0.1).to_directed()

        eType = np.random.choice(np.arange(5, 100), size=nT, replace=False)
        prob  = np.random.uniform(size=nT)
        prob  = prob / sum(prob)

        pType = {eType[k] : prob[k] for k in range(nT)}
        g = qt.set_types_random(g, pTypes=pType, seed=10)

        props = np.array([np.sum(g.ep['eType'].a == k, dtype=float) for k in eType]) / g.num_edges()
        ps    = np.array([pType[k] for k in eType])

        self.assertTrue( np.allclose(props , ps, atol=0.001) )

    @unittest.skip('Shortest path bad')
    def test_shortest_path(self) :

        nV  = 30
        ps  = np.random.uniform(0, 2, size=(nV, 2))

        g = nx.random_geometric_graph(nV, 0.5).to_directed()
        g.vp['pos'] = pos
        g = qt.add_edge_lengths(g)

        paths, dists = qt.shortest_paths(g)

        # Finish this
        self.assertTrue( True )


if __name__ == '__main__':
    unittest.main()
