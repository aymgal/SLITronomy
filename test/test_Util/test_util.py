__auther__ = 'aymgal'

import lenstronomy.Util.util as l_util  # TODO : remove dependency

from slitronomy.Util import util

import numpy as np
import numpy.testing as npt
import pytest
import unittest


def test_soft_threshold():
    thresh = 0.2
    array = np.ones((10, 10))
    array_st = util.soft_threshold(array, thresh)
    assert array_st.shape == array.shape
    npt.assert_equal(array_st, 0.8*np.ones_like(array))

def test_hard_threshold():
    thresh = 0.5
    array = np.random.rand(10, 10)
    array_ht = util.hard_threshold(array, thresh)
    assert array_ht.shape == array.shape
    npt.assert_equal(array_ht[array > 0.5], array[array > 0.5])
    npt.assert_equal(array_ht[array <= 0.5], 0)

def test_indices_conversion():
    num_pix = 99

    x, y = 34, 56
    i = util.index_2d_to_1d(x, y, num_pix)
    x_, y_ = util.index_1d_to_2d(i, num_pix)
    assert x_ == x and y_ == y

    i = 254
    x, y = util.index_1d_to_2d(i, num_pix)
    i_ = util.index_2d_to_1d(x, y, num_pix)
    assert i_ == i

    x_grid_1d, y_grid_1d = l_util.make_grid(num_pix, deltapix=1)
    x_grid_2d, y_grid_2d = util.array2image(x_grid_1d), util.array2image(y_grid_1d)
    i = 254
    x, y = util.index_1d_to_2d(i, num_pix)
    assert x_grid_1d[i] == x_grid_2d[x, y]
    assert y_grid_1d[i] == y_grid_2d[x, y]

class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            array = np.ones((2, 2, 2))
            util.hard_threshold(array, 1)
        with self.assertRaises(ValueError):
            array = np.ones((2, 2, 2))
            util.soft_threshold(array, 1)


if __name__ == '__main__':
    pytest.main()
