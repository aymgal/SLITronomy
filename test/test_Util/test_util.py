__auther__ = 'aymgal'

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
