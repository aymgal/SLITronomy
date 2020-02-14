__author__ = 'aymgal'

from slitronomy.Optimization import algorithms

import numpy as np
import numpy.testing as npt
import pytest
import unittest


# TODO: proper test of algorithm convergence, etc... ?

def test_step_FB():
    np.random.seed(18)
    Y = np.ones((10, 10))
    X = np.random.rand(10, 10)
    grad = lambda x: -(Y - x)
    prox = lambda x, y: x
    step_size = 0.1
    X_updt = algorithms.step_FB(X, grad, prox, step_size)
    assert X.shape == X_updt.shape
    X_FB = prox(X - step_size * grad(X), step_size)
    npt.assert_equal(X_updt, X_FB)

def test_step_FISTA():
    np.random.seed(18)
    Y = np.ones((10, 10))
    X = np.random.rand(10, 10)
    grad = lambda x: -(Y - x)
    prox = lambda x, y: x
    step_size = 0.1
    y_algo, t_algo = np.copy(X), 1
    X_updt, y_algo, t_algo = algorithms.step_FISTA(X, y_algo, t_algo, grad, prox, step_size)
    assert X.shape == X_updt.shape

if __name__ == '__main__':
    pytest.main()
