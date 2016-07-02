import os
import unittest
try:
    import unittest.mock as mock
except ImportError:
    import mock

import networkx as nx
import numpy as np

import matplotlib.image as mpimg

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


class TestQueueNetworkDiGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g = qt.QueueNetworkDiGraph(nx.krackhardt_kite_graph())
        np.random.seed(10)

    @mock.patch.dict('sys.modules', matplotlib_mock)
    def testlines_scatter_args(self):
        ax = mock.Mock()
        ax.transData = mock.Mock()
        line_args = {'linewidths': 77, 'vmax': 107}
        scat_args = {'vmax': 107}
        kwargs = {'pos': {v: (910, 10) for v in self.g.nodes()}}

        a, b = self.g.lines_scatter_args(line_args, scat_args, **kwargs)

        self.assertEqual(a['linewidths'], 77)
        self.assertEqual(b['vmax'], 107)
        self.assertTrue('beefy' not in a and 'beefy' not in b)


    def test_draw_graph(self):
        pos = np.random.uniform(size=(self.g.number_of_nodes(), 2))
        kwargs = {
            'fname': 'test1.png',
            'pos': pos
        }
        self.g.draw_graph(scatter_kwargs={'s': 100}, **kwargs)

        img0 = mpimg.imread('tests/img/test.png')
        img1 = mpimg.imread('test1.png')

        if os.path.exists('test1.png'):
            os.remove('test1.png')

        pixel_diff = (img0 != img1).flatten()
        num_pixels = pixel_diff.shape[0] + 0.0
        self.assertLess(pixel_diff.sum() / num_pixels, 0.0001)

        with mock.patch('queueing_tool.graph.graph_wrapper.HAS_MATPLOTLIB', False):
            with self.assertRaises(ImportError):
                self.g.draw_graph()

        kwargs = {'pos': 1}
        self.g.set_pos = mock.MagicMock()
        with mock.patch.dict('sys.modules', matplotlib_mock):
            self.g.draw_graph(**kwargs)

        self.g.set_pos.assert_called_once_with(1)
