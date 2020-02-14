__author__ = 'aymgal'

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

    # test l0-norm
    coeffs_proxed_l0 = proximals.prox_sparsity_wavelets(coeffs, step, level_const, level_pixels, l_norm=0)
    assert coeffs_proxed_l0.shape == coeffs.shape
    assert coeffs_proxed_l0.min() == 0
    npt.assert_equal(coeffs_proxed_l0[0, :, :], np.zeros((n_side_pixels, n_side_pixels)))
    npt.assert_equal(coeffs_proxed_l0[-1, :, :], coeffs[-1, :, :])

    # test l1-norm
    coeffs_proxed_l1 = proximals.prox_sparsity_wavelets(coeffs, step, level_const, level_pixels, l_norm=1)
    assert coeffs_proxed_l1.shape == coeffs.shape
    assert coeffs_proxed_l1.min() == 0
    npt.assert_equal(coeffs_proxed_l1[0, :, :], np.zeros((n_side_pixels, n_side_pixels)))
    npt.assert_equal(coeffs_proxed_l1[1, :, :], np.ones((n_side_pixels, n_side_pixels)))
    npt.assert_equal(coeffs_proxed_l1[-1, :, :], coeffs[-1, :, :])

    # test with minimal arguments
    coeffs_proxed_minimal = proximals.prox_sparsity_wavelets(coeffs, step)
    npt.assert_equal(coeffs_proxed_minimal, coeffs)

    coeffs_proxed_minimal_l1 = proximals.prox_sparsity_wavelets(coeffs, step, l_norm=1)
    npt.assert_equal(coeffs_proxed_minimal_l1, coeffs)

def test_prox_positivity():
    image = np.arange(-8, 8).reshape(4, 4)
    image_pos = proximals.prox_positivity(image)
    assert image.shape == image_pos.shape
    assert not np.all(image >= 0)
    assert np.all(image_pos >= 0)

class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            coeffs = np.empty((3, 5, 5))
            proximals.prox_sparsity_wavelets(coeffs, 1, l_norm=2)


if __name__ == '__main__':
    pytest.main()
