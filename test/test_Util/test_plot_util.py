__author__ = 'aymgal'

import matplotlib
matplotlib.use('agg')

import pytest
import numpy as np
import matplotlib.pyplot as plt

from slitronomy.Util import plot_util


def test_nice_colorbars():
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.random.rand(10, 10))
    plot_util.nice_colorbar(im)
    # plt.show()
    plt.close()
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.random.rand(10, 10))
    res_map = 10*np.random.rand(10, 10)
    plot_util.nice_colorbar_residuals(im, res_map, vmin=-6, vmax=6)
    # plt.show()
    plt.close()

def test_std_colorbars():
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.random.rand(10, 10))
    plot_util.std_colorbar(im)
    # plt.show()
    plt.close()
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.random.rand(10, 10))
    res_map = 10*np.random.rand(10, 10)
    plot_util.std_colorbar_residuals(im, res_map, vmin=-6, vmax=6)
    # plt.show()
    plt.close()

def test_log_cmap():
    cmap = plt.get_cmap('cubehelix')
    cmap_mod = plot_util.log_cmap('cubehelix', vmin=1e-2, vmax=1e1)
    assert cmap.name == cmap_mod.name
    assert cmap.N == cmap_mod.N
    assert cmap.is_gray() == cmap_mod.is_gray()


if __name__ == '__main__':
    pytest.main()
