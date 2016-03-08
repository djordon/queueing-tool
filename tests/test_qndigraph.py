import os
import unittest
try:
    import unittest.mock as mock
except ImportError:
    import mock

from numpy.random import randint
import networkx as nx
import numpy as np

import matplotlib.image as mpimg

import queueing_tool as qt


TRAVIS_TEST = os.environ.get('TRAVIS_TEST', False)
#a_mock.pyplot.cm = mock.Mock()
#a_mock.pyplot.style = mock.Mock()
#a_mock.animation.FuncAnimation = mock.Mock()
#a_mock.pyplot.style.use = mock.Mock()
#a_mock.pyplot.cm.ocean_r = mock.Mock()
#a_mock.collections.LineCollection = mock.Mock()

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
        cls.g = qt.generate_pagerank_graph(20, sfdp=True, seed=25)


    @mock.patch.dict('sys.modules', matplotlib_mock)
    def test_lines_scatter_args(self):
        ax = mock.Mock()
        ax.transData = mock.Mock()
        kwargs = {'linewidths': 77, 'vmax': 107, 'beefy': 910}

        a, b = self.g._lines_scatter_args(ax, **kwargs)

        self.assertTrue(a['linewidths'] == 77)
        self.assertTrue(b['vmax'] == 107)
        self.assertTrue('beefy' not in a and 'beefy' not in b)


    def test_draw_graph(self):
        kwargs = {'fname': 'test1.png'}
        self.g.draw_graph(**kwargs)

        img0 = mpimg.imread('tests/img/test.png')
        img1 = mpimg.imread('test1.png')
        self.assertTrue((img0 == img1).all())

        if os.path.exists('test1.png'):
            os.remove('test1.png')
