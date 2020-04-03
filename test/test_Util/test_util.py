__author__ = 'aymgal'

from slitronomy.Util import util

from lenstronomy.LightModel.Profiles.starlets import Starlets

import numpy as np
import numpy.testing as npt
import pytest
import unittest

np.random.seed(18)


def test_make_grid():
    numPix = 11
    deltapix = 1.
    grid = util.make_grid(numPix, deltapix)
    assert grid[0][0] == -5
    assert np.sum(grid[0]) == 0
    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=2.)
    print(np.sum(x_grid))
    assert np.sum(x_grid) == 0
    assert x_grid[0] == -5.25

    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=1, left_lower=True)
    assert x_grid[0] == 0
    assert y_grid[0] == 0

def test_array2image():
    array = np.linspace(1, 100, 100)
    image = util.array2image(array)
    assert image[9][9] == 100
    assert image[0][9] == 10

def test_image2array():
    image = np.zeros((10, 10))
    image[1, 2] = 1
    array = util.image2array(image)
    assert array[12] == 1

def test_image2array2image():
    image = np.zeros((20, 10))
    nx, ny = np.shape(image)
    image[1, 2] = 1
    array = util.image2array(image)
    image_new = util.array2image(array, nx, ny)
    assert image_new[1, 2] == image[1, 2]

def test_array2cube():
    array = np.linspace(1, 200, 200)
    image = util.array2cube(array, 2, 100)
    assert image[0][9][9] == 100
    assert image[1][0][9] == 110

def test_cube2array():
    sube = np.zeros((2, 10, 10))
    sube[1, 2, 2] = 1
    array = util.cube2array(sube)
    assert array[122] == 1

def test_cube2array2cube():
    cube = np.zeros((2, 10, 10))
    ns, nx, ny = np.shape(cube)
    assert nx == ny  # condition required
    nxy = nx*ny
    cube[1, 2, 2] = 1
    array = util.cube2array(cube)
    cube_new = util.array2cube(array, ns, nxy)
    assert cube_new[1, 2, 2] == cube[1, 2, 2]

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

    x_grid_1d, y_grid_1d = util.make_grid(num_pix, deltapix=1)
    x_grid_2d, y_grid_2d = util.array2image(x_grid_1d), util.array2image(y_grid_1d)
    i = 254
    x, y = util.index_1d_to_2d(i, num_pix)
    assert x_grid_1d[i] == x_grid_2d[x, y]
    assert y_grid_1d[i] == y_grid_2d[x, y]

def test_dirac_impulse():
    dirac_even = util.dirac_impulse(20)
    assert dirac_even[10, 10] == 1
    dirac_odd = util.dirac_impulse(21)
    assert dirac_odd[10, 10] == 1



def test_spectral_norm():
    num_pix = 10
    operator = lambda X: X
    inverse_operator = lambda X: X
    npt.assert_almost_equal(1.0, util.spectral_norm(num_pix, operator, inverse_operator), decimal=5)

    operator = lambda X: X**2
    inverse_operator = lambda X: np.sqrt(X)
    npt.assert_almost_equal(1.0, util.spectral_norm(num_pix, operator, inverse_operator), decimal=5)

    A = np.diag(np.arange(1, num_pix+1))
    operator = lambda X: A.dot(X)
    inverse_operator = lambda X: A.T.dot(X)
    true_norm = np.real(np.linalg.eigvals(A.T.dot(A)).max())
    npt.assert_almost_equal(true_norm, util.spectral_norm(num_pix, operator, inverse_operator), decimal=2)

    from lenstronomy.LightModel.Profiles.starlets import Starlets
    starlets = Starlets()
    n_scales = 3
    n_pixels = num_pix**2
    operator = lambda X: starlets.decomposition_2d(X, n_scales)
    inverse_operator = lambda X: starlets.function_2d(X, n_scales, n_pixels)
    npt.assert_almost_equal(1.0, util.spectral_norm(num_pix, operator, inverse_operator), decimal=8)


def test_generate_initial_guess():
    num_pix = 20
    n_scales = 3
    n_pixels = num_pix**2
    starlets = Starlets()
    transform = lambda X: starlets.decomposition_2d(X, n_scales)
    inverse_transform = lambda X: starlets.function_2d(X, n_scales, n_pixels)

    guess_direct_space, guess_transf_space = util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                                                                         guess_type='null')
    npt.assert_equal(np.zeros((num_pix, num_pix)), guess_direct_space)
    npt.assert_equal(np.zeros((n_scales, num_pix, num_pix)), guess_transf_space)
    
    guess_direct_space, guess_transf_space = util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                           formulation='analysis', guess_type='background_rms',
                           background_rms=1)
    assert guess_direct_space.shape == (num_pix, num_pix)
    assert guess_transf_space.shape == (n_scales, num_pix, num_pix)

    noise_map = np.random.rand(num_pix, num_pix)
    guess_direct_space, guess_transf_space = util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                           formulation='analysis', guess_type='noise_map',
                           noise_map=noise_map)
    assert guess_direct_space.shape == (num_pix, num_pix)
    assert guess_transf_space.shape == (n_scales, num_pix, num_pix)

    noise_map_synthesis = np.random.rand(n_scales, num_pix, num_pix)
    guess_direct_space, guess_transf_space = util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                           formulation='synthesis', guess_type='noise_map',
                           noise_map_synthesis=noise_map_synthesis)
    assert guess_direct_space.shape == (num_pix, num_pix)
    assert guess_transf_space.shape == (n_scales, num_pix, num_pix)


def test_generate_initial_guess_simple():
    num_pix = 10
    n_scales = 3
    starlets = Starlets()
    transform = lambda X: starlets.decomposition_2d(X, n_scales)
    background_rms = 0.1
    guess_direct_space, guess_transf_space = util.generate_initial_guess_simple(num_pix, transform, background_rms)
    assert guess_direct_space.shape == (num_pix, num_pix)
    assert guess_transf_space.shape == (n_scales, num_pix, num_pix)


def test_linear_decrease_at_iter():
    n_iter_min_value = 10
    n_iter = 30
    iterations = np.arange(0, n_iter)
    thresh_init = 1000
    thresh_min = 5
    k_i = None
    thresholds = []
    for i in iterations:
        k_i = util.linear_decrease_at_iter(i, thresh_init, thresh_min, n_iter, n_iter_min_value)
        thresholds.append(k_i)
    thresholds = np.array(thresholds)
    assert thresholds[0] == thresh_init
    np.testing.assert_almost_equal(thresholds[-n_iter_min_value:], thresh_min*np.ones((n_iter_min_value,)), decimal=10)


def test_exponential_decrease_at_iter():
    n_iter_min_value = 10
    n_iter = 30
    iterations = np.arange(0, n_iter)
    thresh_init = 1000
    thresh_min = 5
    k_i = None
    thresholds = []
    for i in iterations:
        k_i = util.exponential_decrease_at_iter(i, thresh_init, thresh_min, n_iter, n_iter_min_value)
        thresholds.append(k_i)
    thresholds = np.array(thresholds)
    assert thresholds[0] == thresh_init
    np.testing.assert_almost_equal(thresholds[-n_iter_min_value:], thresh_min*np.ones((n_iter_min_value,)), decimal=10)



class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            array = np.ones((2, 2, 2))
            util.hard_threshold(array, 1)
        with self.assertRaises(ValueError):
            array = np.ones((2, 2, 2))
            util.soft_threshold(array, 1)
        with self.assertRaises(ValueError):
            array = np.ones(5)
            util.array2image(array)
        with self.assertRaises(ValueError):
            array = np.ones((2, 2))
            util.array2cube(array, 2, 2)


if __name__ == '__main__':
    pytest.main()
