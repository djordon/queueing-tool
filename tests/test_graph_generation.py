import numpy    as np
import queueing_tool  as qt
import graph_tool.all as gt
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


    def test_adjacency2graph(self) :

        adj = {0 : 1, 1 : [2, 3]}
        eTy = {0 : 5, 1 : [9, 14]}
        g   = qt.adjacency2graph(adj, eType=eTy, adjust=1)
        ans = qt.graph2dict(g)
        self.assertTrue(ans[1] == {0 : [5], 1 : [0, 0], 2 : [], 3 : []})

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



if __name__ == '__main__':
    unittest.main()
