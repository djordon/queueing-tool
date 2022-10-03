import unittest

import networkx as nx
import numpy as np
import pytest

import queueing_tool as qt


def generate_adjacency(a=3, b=25, c=6, n=12):
    ans = {}
    for k in np.unique(np.random.randint(a, b, n)):
        count = np.random.randint(1, c)
        ans[k] = {j: {} for j in np.random.randint(a, b, count)}
    return ans


@pytest.fixture(name="expected_response0")
def fixture_expected_response0():
    return {
        0: {1: {'edge_type': 5}},
        1: {2: {'edge_type': 9}, 3: {'edge_type': 14}},
        2: {0: {'edge_type': 1}},
        3: {3: {'edge_type': 0}}
    }


@pytest.fixture(name="expected_response1")
def fixture_expected_response1():
    return {
        0: {1: {'edge_type': 5}},
        1: {2: {'edge_type': 9}, 3: {'edge_type': 0}},
        2: {0: {'edge_type': 1}},
        3: {}
    }


class TestGraphFunctions:
    @staticmethod
    def test_graph2dict():
        adj = generate_adjacency()
        g1 = qt.adjacency2graph(adj, adjust=2)
        aj1 = qt.graph2dict(g1)
        g2 = qt.adjacency2graph(aj1, adjust=2)
        assert nx.is_isomorphic(g1, g2)

    @staticmethod
    def test_add_edge_lengths():
        g1 = qt.generate_pagerank_graph(10)
        g2 = qt.add_edge_lengths(g1)

        edge_props = set()
        for key in g2.edge_properties():
            edge_props.add(key)

        assert 'edge_length' in edge_props

    @staticmethod
    def test_generate_transition():
        g = qt.generate_random_graph(20)
        mat = qt.generate_transition_matrix(g)

        ans = np.sum(mat, axis=1)
        assert np.allclose(ans, 1)

        mat = qt.generate_transition_matrix(g, seed=10)
        ans = np.sum(mat, axis=1)
        assert np.allclose(ans, 1)

    @staticmethod
    def test_adjacency2graph_matrix_adjacency(expected_response1):

        # Test adjacency argument using ndarray work
        adj = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0]])
        ety = {0: {1: 5}, 1: {2: 9, 3: 14}}

        g = qt.adjacency2graph(adj, edge_type=ety, adjust=2)
        ans = qt.graph2dict(g)

        assert ans == expected_response1

    @staticmethod
    def test_adjacency2graph_matrix_etype(expected_response0):
        # Test adjacency argument using ndarrays work
        adj = {0: {1: {}}, 1: {2: {}, 3: {}}, 2: {0: {}}, 3: {}}
        ety = np.array([[0, 5, 0, 0],
                        [0, 0, 9, 14],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        g = qt.adjacency2graph(adj, edge_type=ety, adjust=1)
        ans = qt.graph2dict(g)
        assert ans == expected_response0

    @staticmethod
    def test_adjacency2graph_errors():
        with pytest.raises(TypeError):
            qt.adjacency2graph([])

    @staticmethod
    def test_set_types_random():

        nV = 1200
        nT = np.random.randint(5, 10)
        g = nx.random_geometric_graph(nV, 0.1).to_directed()

        eType = np.random.choice(np.arange(5, 100), size=nT, replace=False)
        prob = np.random.uniform(size=nT)
        prob = prob / sum(prob)

        pType = {eType[k]: prob[k] for k in range(nT)}
        g = qt.set_types_random(g, proportions=pType)

        non_loops = [e for e in g.edges() if e[0] != e[1]]
        mat = [[g.ep(e, 'edge_type') == k for e in non_loops] for k in eType]
        props = (np.array(mat).sum(1) + 0.0) / len(non_loops)
        ps = np.array([pType[k] for k in eType])

        assert np.allclose(props, ps, atol=0.01)

        prob[-1] = 2
        pType = {eType[k]: prob[k] for k in range(nT)}
        with pytest.raises(ValueError):
            g = qt.set_types_random(g, proportions=pType, seed=10)

        with pytest.raises(ValueError):
            g = qt.set_types_random(g, loop_proportions=pType, seed=10)

    @staticmethod
    def test_test_graph_importerror():
        with pytest.raises(TypeError):
            qt.generate_transition_matrix(1)
