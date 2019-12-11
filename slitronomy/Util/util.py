__author__ = 'aymgal'

import numpy as np


def array2image(array, nx=0, ny=0):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)
    Original method implemented in lenstronomy

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    if nx == 0 or ny == 0:
        n = int(np.sqrt(len(array)))
        if n**2 != len(array):
            raise ValueError("lenght of input array given as %s is not square of integer number!" %(len(array)))
        nx, ny = n, n
    image = array.reshape(int(nx), int(ny))
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array
    Original method implemented in lenstronomy

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh


def array2cube(array, n_1, n_23):
    """
    """
    n = int(np.sqrt(n_23))
    if n**2 != n_23:
        raise ValueError("2nd and 3rd dims (%s) are not square of integer number!" % n_23)
    n_2, n_3 = n, n
    cube = array.reshape(n_1, n_2, n_3)
    return cube


def cube2array(cube):
    """
    """
    n_1, n_2, n_3 = cube.shape
    array = cube.reshape(n_1 * n_2 * n_3)
    return array


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

def spectral_norm(num_pix, operator, inverse_operator, num_iter=20, tol=1e-10):
    """compute spectral norm from operator and its inverse"""
    random_array = np.random.randn(num_pix, num_pix)
    norm = np.linalg.norm(random_array)
    random_array /= norm

    i = 0
    err = abs(tol)
    while i < num_iter and err >= tol:
        # print(i, norm)
        coeffs = operator(random_array)
        random_array = inverse_operator(coeffs)
        norm_new = np.linalg.norm(random_array)
        random_array /= norm_new
        err = abs(norm_new - norm)/norm_new
        norm = norm_new
        i += 1
    return norm
