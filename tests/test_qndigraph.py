import os
import unittest.mock as mock

import matplotlib
import matplotlib.image
import networkx as nx
import numpy as np
import pytest


import queueing_tool as qt


CI_TEST = os.environ.get('CI_TEST', False)

a_mock = mock.Mock()
a_mock.pyplot = mock.Mock()
a_mock.animation = mock.Mock()
a_mock.collections = mock.Mock()

matplotlib_mock = {
    'matplotlib': a_mock,
    'matplotlib.pyplot': a_mock.pyplot,
    'matplotlib.animation': a_mock.animation,
    'matplotlib.collections': a_mock.collections,
}

@pytest.fixture(scope="module", name="graph")
def fixture_queue_network_graph():
    return qt.QueueNetworkDiGraph(nx.krackhardt_kite_graph())


@pytest.fixture(scope="module", autouse=True)
def fixture_set_seed():
    np.random.seed(10)
    return


class TestQueueNetworkDiGraph:
    @staticmethod
    def testlines_scatter_args(graph):
        with mock.patch.dict('sys.modules', matplotlib_mock):
            ax = mock.Mock()
            ax.transData = mock.Mock()
            line_args = {'linewidths': 77, 'vmax': 107}
            scat_args = {'vmax': 107}
            kwargs = {'pos': {v: (910, 10) for v in graph.nodes()}}

            a, b = graph.lines_scatter_args(line_args, scat_args, **kwargs)

            assert a['linewidths'] == 77
            assert b['vmax'] == 107
            assert 'beefy' not in a and 'beefy' not in b

    @pytest.mark.xfail
    def test_draw_graph(graph):
        pos = np.random.uniform(size=(graph.number_of_nodes(), 2))
        kwargs = {
            'fname': 'test1.png',
            'pos': pos
        }
        graph.draw_graph(scatter_kwargs={'s': 100}, **kwargs)

        version = 1 if matplotlib.__version__.startswith('1') else 2
        filename = 'test-mpl-{version}.x.png'.format(version=version)

        img0 = matplotlib.image.imread('tests/img/{filename}'.format(filename=filename))
        img1 = matplotlib.image.imread('test1.png')

        if os.path.exists('test1.png'):
            os.remove('test1.png')

        pixel_diff = (img0 != img1).flatten()
        num_pixels = pixel_diff.shape[0] + 0.0
        assert pixel_diff.sum() / num_pixels < 0.0001

        with mock.patch('queueing_tool.graph.graph_wrapper.HAS_MATPLOTLIB', False):
            with pytest.raises(ImportError):
                graph.draw_graph()

        kwargs = {'pos': 1}
        graph.set_pos = mock.MagicMock()
        with mock.patch.dict('sys.modules', matplotlib_mock):
            graph.draw_graph(**kwargs)

        graph.set_pos.assert_called_once_with(1)
