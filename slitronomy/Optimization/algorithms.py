__author__ = 'aymgal'

# implementations of gradient descent and proximal algorithms

import numpy as np


def step_FB(x, grad, prox, step_size):
    """Forward Backward algorithm"""
    x_next = prox(x - step_size * grad(x), step_size)
    return x_next


def step_FISTA(x, y, t, grad, prox, step_size):
    """Fast Iterative Shrinkage-Thresholding Algorithm"""
    x_next = prox(y - step_size * grad(y), step_size)
    t_next = 0.5 * (1 + np.sqrt(1 + 4*t**2))
    factor = (t - 1) / t_next
    y_next = x_next + factor * (x_next - x)
    return x_next, y_next, t_next

def step_PD(x, u, grad_f, prox_g, prox_h, step_size, tau, sigma, phi, phi_T):
    """Condat-Vu Primal-Dual algorithm"""
    #x_tmp = x + tau * (grad_f(x) - phi(u))
    x_tmp = x - tau * grad_f(x) - phi(u)
    x_next = prox_h(x_tmp)
    u_tmp = u + sigma * phi_T(2. * x_next - x)
    u_next = u_tmp - prox_g(u_tmp, step_size)
    return x_next, u_next
