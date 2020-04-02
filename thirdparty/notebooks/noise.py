import numpy as np


def regridding_error_map_squared(data_image, magnification_map, image_pixel_scale, source_pixel_scale):
    """
    Computes the regridding error map as defined in Suyu et al. 2009 (https://ui.adsabs.harvard.edu/abs/2009ApJ...691..277S/abstract)
    The output is an image with pixel sigma^2
    """
    d = data_image
    mu = np.abs(magnification_map)
    noise_map2 = 1/12. * (source_pixel_scale / image_pixel_scale)**2 * np.ones_like(d)
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
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
            noise_map2[i, j] *= mu[i, j] * sum_adj / n_adj
    return noise_map2
