__auther__ = 'aymgal'

from slitronomy.Optimization import proximals

import numpy as np
import numpy.testing as npt
import pytest
import unittest


def test_prox_sparsity_wavelets():
    n_scales = 3
    n_side_pixels = 5
    coeffs = np.empty((n_scales, n_side_pixels, n_side_pixels))
    coeffs[0, :, :] = 9 * np.ones(25).reshape(n_side_pixels, n_side_pixels)
    coeffs[1, :, :] = 11 * np.ones(25).reshape(n_side_pixels, n_side_pixels)
    coeffs[2, :, :] = np.arange(25).reshape(n_side_pixels, n_side_pixels)

    step = 1
    level_const = 10 * np.ones(n_scales)
    level_pixels = np.ones(75).reshape(n_scales, n_side_pixels, n_side_pixels)

    coeffs_proxed_l0 = proximals.prox_sparsity_wavelets(coeffs, step, level_const, level_pixels, l_norm=0)
    assert coeffs_proxed_l0.shape == coeffs.shape
    assert coeffs_proxed_l0.min() == 0
    npt.assert_equal(coeffs_proxed_l0[0, :, :], np.zeros((n_side_pixels, n_side_pixels)))
    npt.assert_equal(coeffs_proxed_l0[-1, :, :], coeffs[-1, :, :])

    coeffs_proxed_l1 = proximals.prox_sparsity_wavelets(coeffs, step, level_const, level_pixels, l_norm=1)
    assert coeffs_proxed_l1.shape == coeffs.shape
    assert coeffs_proxed_l1.min() == 0
    npt.assert_equal(coeffs_proxed_l1[0, :, :], np.zeros((n_side_pixels, n_side_pixels)))
    npt.assert_equal(coeffs_proxed_l1[1, :, :], np.ones((n_side_pixels, n_side_pixels)))
    npt.assert_equal(coeffs_proxed_l1[-1, :, :], coeffs[-1, :, :])


# class TestRaise(unittest.TestCase):

#     def test_raise(self):
#         with self.assertRaises(ValueError):



if __name__ == '__main__':
    pytest.main()
