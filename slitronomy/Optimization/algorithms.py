__author__ = 'aymgal'

# implementations of gradient descent and proximal algorithms

import numpy as np


def step_FB(x, grad, prox_list, step_size):
    """Forward Backward Algorithm"""
    x_ = np.copy(x)
    for prox in prox_list:
        x_next = prox(x_ - step_size * grad(x_), step_size)
        x_ = x_next
    return x_next

def step_FISTA(x, y, t, grad, prox_list, step_size):
    """Fast Iterative Shrinkage-Thresholding Algorithm"""
    #TODO: check correctness of FISTA with arbitrary number of constraints
    y_ = np.copy(y)
    for prox in prox_list:
        x_next = prox(y_ - step_size * grad(y_), step_size)
        y_ = x_next
    t_next = 0.5 * (1 + np.sqrt(1 + 4*t**2))
    factor = (t - 1) / t_next
    y_next = x_next + factor * (x_next - x)
    return x_next, y_next, t_next
