__author__ = 'aymgal'

import matplotlib
matplotlib.use('agg')

import pytest
import numpy as np
import matplotlib.pyplot as plt

from slitronomy.Util import plot_util


def test_nice_colorbar():
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.random.rand(10, 10))
    plot_util.nice_colorbar(im)
    # plt.show()
    plt.close()

if __name__ == '__main__':
    pytest.main()
