__author__ = 'aymgal'

import numpy as np


def make_grid(numPix, deltapix, subgrid_res=1, left_lower=False):
    """
    Credits to S. Birrer (lenstronomy)

    :param numPix: number of pixels per axis
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x_grid = matrix[:, 0] * deltapix_eff
    y_grid = matrix[:, 1] * deltapix_eff
    if left_lower is True:
        shift = -1. / 2 + 1. / (2 * subgrid_res)
    else:
        shift = np.sum(x_grid) / numPix_eff**2
    return x_grid - shift, y_grid - shift


# def grid_from_coordinate_transform(nx, ny, Mpix2coord, ra_at_xy_0, dec_at_xy_0):
#     """
#     Credits to S. Birrer (lenstronomy)
    
#     return a grid in x and y coordinates that satisfy the coordinate system


#     :param nx: number of pixels in x-axis
#     :param ny: number of pixels in y-axis
#     :param Mpix2coord: transformation matrix (2x2) of pixels into coordinate displacements
#     :param ra_at_xy_0: RA coordinate at (x,y) = (0,0)
#     :param dec_at_xy_0: DEC coordinate at (x,y) = (0,0)
#     :return: RA coordinate grid, DEC coordinate grid
#     """
#     a = np.arange(nx)
#     b = np.arange(ny)
#     matrix = np.dstack(np.meshgrid(a, b)).reshape(-1, 2)
#     x_grid = matrix[:, 0]
#     y_grid = matrix[:, 1]
#     ra_grid = x_grid * Mpix2coord[0, 0] + y_grid * Mpix2coord[0, 1] + ra_at_xy_0
#     dec_grid = x_grid * Mpix2coord[1, 0] + y_grid * Mpix2coord[1, 1] + dec_at_xy_0
#     return ra_grid, dec_grid


def array2image(array, nx=0, ny=0):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)
    Credits to S. Birrer (lenstronomy)

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
    Credits to S. Birrer (lenstronomy)

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


def index_2d_to_1d(x, y, num_pix):
    i = y + x*num_pix
    return i


def index_1d_to_2d(i, num_pix):
    x = int(i / num_pix)
    y = int(i % num_pix)
    return (x, y)


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
    random_array = np.random.rand(num_pix, num_pix)
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


def dirac_impulse(num_pix):
    """
    returns the 2d array of a Dirac impulse at the center of the image

    :return: 2d numpy array
    """
    dirac = np.zeros((num_pix, num_pix), dtype=float)
    dirac[int(num_pix/2), int(num_pix/2)] = 1.
    return dirac


def generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                           formulation='analysis', guess_type='background_rms',
                           background_rms=None, noise_map=None, noise_map_synthesis=None):
    if guess_type == 'null':
        X = np.zeros((num_pix, num_pix))
        alpha_X = np.zeros((n_scales, num_pix, num_pix))
    elif guess_type == 'background_rms':
        if formulation == 'analysis':
            X = background_rms * np.random.randn(num_pix, num_pix)
            alpha_X = transform(X)
        elif formulation == 'synthesis':
            raise ValueError("initial guess type 'background_rms' not compatible with synthesis formulation")
    elif guess_type == 'noise_map':
        if formulation == 'analysis':
            X = np.copy(noise_map)  # np.median(noise_map) * np.random.randn(num_pix, num_pix)
            alpha_X = transform(X)
        elif formulation == 'synthesis':
            alpha_X = np.copy(noise_map_synthesis)
            X = inverse_transform(alpha_X)
    else:
        raise ValueError("Initial guess type '{}' not supported".format(guess_type))
    return X, alpha_X

def generate_initial_guess_simple(num_pix, transform, background_rms):
    X = background_rms * np.random.randn(num_pix, num_pix)
    alpha_X = transform(X)
    return X, alpha_X
