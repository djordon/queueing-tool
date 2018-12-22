import os
try:
    import unittest.mock as mock
except ImportError:
    import mock

import networkx as nx
import numpy as np
import pytest

import matplotlib
import matplotlib.image

import queueing_tool as qt


TRAVIS_TEST = os.environ.get('TRAVIS_TEST', False)

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


@pytest.fixture(name='queue_network_graph', scope='module')
def fixture_queue_network_graph():
    np.random.seed(10)
    return qt.QueueNetworkDiGraph(nx.krackhardt_kite_graph())


class TestQueueNetworkDiGraph(object):

    @staticmethod
    @mock.patch.dict('sys.modules', matplotlib_mock)
    def test_lines_scatter_args(queue_network_graph):
        np.random.seed(10)
        ax = mock.Mock()
        ax.transData = mock.Mock()
        line_args = {'linewidths': 77, 'vmax': 107}
        scat_args = {'vmax': 107}
        kwargs = {'pos': {v: (910, 10) for v in queue_network_graph.nodes()}}

        a, b = queue_network_graph.lines_scatter_args(line_args, scat_args, **kwargs)

        assert a['linewidths'] == 77
        assert b['vmax'] == 107
        assert 'beefy' not in a and 'beefy' not in b

    @staticmethod
    def test_draw_graph(queue_network_graph):
        np.random.seed(10)
        pos = np.random.uniform(size=(queue_network_graph.number_of_nodes(), 2))
        kwargs = {
            'fname': 'test1.png',
            'pos': pos
        }
        queue_network_graph.draw_graph(scatter_kwargs={'s': 100}, **kwargs)

        version = matplotlib.__version__[0]
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
                queue_network_graph.draw_graph()

        kwargs = {'pos': 1}
        queue_network_graph.set_pos = mock.MagicMock()
        with mock.patch.dict('sys.modules', matplotlib_mock):
            queue_network_graph.draw_graph(**kwargs)

        queue_network_graph.set_pos.assert_called_once_with(1)
