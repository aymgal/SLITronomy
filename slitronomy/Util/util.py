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


def spectral_norm(num_pix, operator, inverse_operator, num_iter=20, tol=1e-10, seed=None):
    """

    compute spectral norm from operator and its inverse

    Sometimes referred to as the Lipschitz constant, it is essentially the greatest singular value of a matrix/operator
    (in this case the Starlet transform operator)
    This is particularly important for determining the optimal amplitude of gradient descent step in a minimization problem.

    See e.g. http://fourier.eng.hmc.edu/e161/lectures/algebra/node12.html

    """
    if seed is not None:
        np.random.seed(seed)
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
                           background_rms=None, noise_map=None, noise_map_synthesis=None,
                           seed=None):
    """Generates a random image and its transform for sparse optimization initialisation.
    This supports both analysis and synthesis types of initial guess.
    
    
    Parameters
    ----------
    num_pix : int
        Number of side pixels.
    n_scales : int
        Number of decomposition scales, consistent with the `transform` operator.
    transform : callable
        Operator (e.g. wavelet transform) for transformed random guess.
    inverse_transform : callable
        Inverse operator (e.g. wavelet inverse transform) for direct space random guess.
    formulation : str, 'analysis'
        'analysis' generates the random image in direct space, then transforms it.
        'synthesis' generates the random coefficients in transformed space, then inverse transforms it for direct space image.
    guess_type : str, 'background_rms'
        'background_rms' uses the background noise RMS value for random image generate.
        'noise_map' uses the whole noise (diagonal) covariance for random image generate.
    background_rms : float, None
        Background noise RMS value. Required if `guess_type` is 'background_rms'
    noise_map : array_like, None
        Noise map, diagonal noise covariance per pixel. Required if `guess_type` is 'noise_map' and `formulation` is 'analysis'.
    noise_map_synthesis : array_like, None
        Diagonal noise covariance per pixel, in transformed space. Required if `guess_type` is 'noise_map' and `formulation` is 'synthesis'.
    
    Returns
    -------
    array_like
        Random image.
    array_like
        Transform of the above array, after call to the `transform` callable.
    """
    if seed is not None:
        np.random.seed(seed)
    if formulation not in ['analysis', 'synthesis']:
        raise ValueError("Formulation type '{}' not supported".format(formulation))
    if guess_type not in ['null', 'background_rms', 'noise_map']:
        raise ValueError("Initial guess type '{}' not supported".format(guess_type))
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
    return X, alpha_X


def generate_initial_guess_simple(num_pix, transform, background_rms, seed=None):
    """Generates a random image and its transform for sparse optimization initialisation.
    
    Parameters
    ----------
    num_pix : int
        Number of side pixels.
    transform : callable
        Operator (e.g. wavelet transform) for transformed random image.
    background_rms : float
        Background noise RMS value.
    
    Returns
    -------
    array_like
        Random image, normal distribution with std dev `background_rms`.
    array_like
        Transform of the above array, after call to the `transform` callable.
    """
    if seed is not None:
        np.random.seed(seed)
    X = background_rms * np.random.randn(num_pix, num_pix)
    alpha_X = transform(X)
    return X, alpha_X

def linear_decrease(curr_value, init_value, min_value, num_iter, num_iter_at_min_value):
    """Computes a linearly decreasing value, for a given loop index, starting at a specified value.
    
    Parameters
    ----------
    curr_value : float
        Current value to be updated
    init_value : float
        Value at iteration 0.
    min_value : float
        Minimum value, reached at iteration num_iter - num_iter_at_min_value - 1.
    num_iter : int
        Total number of iterations.
    num_iter_at_min_value : int
        Number of iteration for which the returned value equals `min_value`.
    
    Returns
    -------
    float
        Linearly decreased value.
    
    Raises
    ------
    ValueError
        If num_iter - num_iter_at_min_value < 1, cannot compute the value.
    """
    num_iter_eff = num_iter - num_iter_at_min_value
    if num_iter_eff < 1:
        raise ValueError("Too low number of iterations ({}) to decrease threshold".format(num_iter))
    delta_k = (min_value - init_value) / num_iter_eff
    new_value = curr_value + delta_k
    return max(new_value, min_value)


def exponential_decrease(curr_value, init_value, min_value, num_iter, num_iter_at_min_value):
    """Computes a exponentially decreasing value, for a given loop index, starting at a specified value.
    
    Parameters
    ----------
    curr_value : float
        Current value to be updated
    init_value : float
        Value at iteration 0.
    min_value : float
        Minimum value, reached at iteration num_iter - num_iter_at_min_value - 1.
    num_iter : int
        Total number of iterations.
    num_iter_at_min_value : int
        Number of iteration for which the returned value equals `min_value`.
    
    Returns
    -------
    float
        Exponentially decreased value.
    
    Raises
    ------
    ValueError
        If num_iter - num_iter_at_min_value < 1, cannot compute the value.
    """
    num_iter_eff = num_iter - num_iter_at_min_value
    if num_iter_eff < 1:
        raise ValueError("Too low number of iterations ({}) to decrease threshold".format(num_iter))
    exp_factor = np.exp(np.log(min_value/init_value) / num_iter_eff)
    new_value = curr_value * exp_factor
    return max(new_value, min_value)


def regridding_error_map_squared(mag_map=None, data_image=None, image_pixel_scale=None, source_pixel_scale=None,
                                 noise_map2_prefactor=None):
    """Computes the regridding error map as defined in Suyu et al. 2009 (https://ui.adsabs.harvard.edu/abs/2009ApJ...691..277S/abstract)
    The error is a 2D array corresponding to the noise variance sigma^2 per pixel.
    
    Parameters
    ----------
    mag_map : array_like, optional
        Magnification map, as a 2D array.
    data_image : array_like, optional
        Imaging data, as a 2D array.
    image_pixel_scale : float, optional
        Pixel scale of image plane grid.
    source_pixel_scale : None, optional
        Pixel scale of source plane grid.
    noise_map2_prefactor : None, optional
        Prefactor pre-computed. If provided, this won't be computed again.
        This is the second returned array of this function.
    
    Returns
    -------
    array_like
        Full regridding error map.
    array_like
        Part of the map that is independent of the magnification, for speedup next call to the function.
    """

    if noise_map2_prefactor is None:
        d = data_image
        noise_map2_prefactor = 1/12. * (source_pixel_scale / image_pixel_scale)**2 * np.ones_like(d)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                sum_adj, n_adj = 0, 0
                try:
                    sum_adj += (d[i, j] - d[i-1, j])**2
                    n_adj += 1
                except IndexError:
                    pass
                try:
                    sum_adj += (d[i, j] - d[i+1, j])**2
                    n_adj += 1
                except IndexError:
                    pass
                try:
                    sum_adj += (d[i, j] - d[i, j-1])**2
                    n_adj += 1
                except IndexError:
                    pass
                try:
                    sum_adj += (d[i, j] - d[i, j+1])**2
                    n_adj += 1
                except IndexError:
                    pass
                noise_map2_prefactor[i, j] *= sum_adj / n_adj
    if mag_map is None:
        noise_map2 = None
    else:
        mu = np.abs(mag_map)
        noise_map2 = noise_map2_prefactor * mu
    return noise_map2, noise_map2_prefactor


def Downsample(image, factor=1):
    """
    resizes image with nx x ny to nx/factor x ny/factor
    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor == 1:
        return image
    if factor < 1:
        raise ValueError('scaling factor in re-sizing %s < 1' %factor)
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx/f) == nx/f and int(ny/f) == ny/f:
        small = image.reshape([int(nx/f), f, int(ny/f), f]).mean(3).mean(1)
        return small
    else:
        raise ValueError("scaling with factor %s is not possible with grid size %s, %s" %(f, nx, ny))

def Upsample(image, factor=1):
    if factor == 1:
        return image
    if factor < 1:
        raise ValueError('scaling factor in re-sizing %s < 1' %factor)
    f = int(factor)
    n1, n2 = image.shape
    upimage = np.zeros((n1*f, n2*f))
    x, y = np.where(upimage == 0)
    x_, y_ = (x/f).astype(int), (y/f).astype(int)
    upimage[x, y] = image[x_, y_] / f**2
    return upimage
