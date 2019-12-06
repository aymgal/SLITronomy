__author__ = 'aymgal'

import numpy as np


def soft_threshold(array, thresh):
    if len(array.shape) > 2:
        raise ValueError("Soft thresholding only supported for 1D or 2D arrays")
    array_th = np.sign(array) * np.maximum(np.abs(array) - thresh, 0.)
    return array_th


def hard_threshold(array, thresh):
    if len(array.shape) > 2:
        raise ValueError("Hard thresholding only supported for 1D or 2D arrays")
    array_th = np.copy(array)
    array_th[np.abs(array) <= thresh] = 0.
    return array_th
