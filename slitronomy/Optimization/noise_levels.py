__author__ = 'aymgal'

import numpy as np
from scipy import signal
from slitronomy.Util import util


class NoiseLevels(object):

    """
    Handle noise properties and compute noise levels in wavelets space, 
    taking into account lensing and optionally blurring and regridding error for pixelated reconstructions.
    """

    def __init__(self, data_class, subgrid_res_source=1, boost_where_zero=10, include_regridding_error=False):
        """
        :param subgrid_res_source: resolution factor between image plane and source plane
        :param boost_where_zero: sets the multiplcative factor in fron tof the average noise levels
        at locations where noise is 0
        :param include_regridding_error: if True, includes the regridding error controbution in noise covariance.
        See Suyu et al. 2009 (https://ui.adsabs.harvard.edu/abs/2009ApJ...691..277S/abstract) for details.
        """
        # background noise
        self._background_rms = data_class.background_rms
        # noise full covariance \simeq sqrt(poisson_rms^2 + gaussian_rms^2)
        self._noise_map_data = np.sqrt(data_class.C_D)
        self.include_regridding_error = include_regridding_error
        if self.include_regridding_error:
            self._initialise_regridding_error(data_class.data, data_class.pixel_width, 
                                              data_class.pixel_width/subgrid_res_source)
        # boost noise in pixels that are not mapped to any image plane pixels
        self._boost_where_zero = boost_where_zero

    @property
    def background_rms(self):
        return self._background_rms

    @property
    def noise_map(self):
        return self._noise_map_data

    @property
    def effective_noise_map(self):
        if not self.include_regridding_error:
            return self.noise_map
        if not hasattr(self, '_noise_map_with_regrid'):
            raise ValueError("Regridding error map has not be updated with magnification map")
        return self._noise_map_with_regrid

    @property
    def levels_source(self):
        if not hasattr(self, '_noise_levels_src'):
            raise ValueError("Source plane noise levels have not been computed")
        return self._noise_levels_src

    @property
    def levels_image(self):
        if not hasattr(self, '_noise_levels_img'):
            raise ValueError("Image plane noise levels have not been computed")
        return self._noise_levels_img

    def update_source_levels(self, num_pix_image, num_pix_source, wavelet_transform_source, 
                             image2source_transform, psf_kernel=None):
        # get transposed blurring operator
        if psf_kernel is None:
            HT = util.dirac_impulse(num_pix_image)
        else:
            HT = psf_kernel.T

        # map noise map to source plane
        HT_noise_diag = self.noise_map * np.sqrt(np.sum(HT**2))
        FT_HT_noise = image2source_transform(HT_noise_diag)

        # introduce artitifically noise to pixels where there are not signal in source plane
        # to ensure threshold of starlet coefficients at these locations
        FT_HT_noise[FT_HT_noise == 0] = self._boost_where_zero * np.mean(FT_HT_noise[FT_HT_noise != 0])

        # \Gamma^2 in  Equation (16) of Joseph+19)
        FT_HT_noise2 = FT_HT_noise**2

        # compute starlet transform of a dirac impulse in source plane
        dirac = util.dirac_impulse(num_pix_source)
        dirac_coeffs = wavelet_transform_source(dirac)

        # \Delta_s^2 in  Equation (16) of Joseph+19)
        dirac_coeffs2 = dirac_coeffs**2

        n_scale, n_pix1, npix2 = dirac_coeffs2.shape
        noise_levels = np.zeros((n_scale, n_pix1, npix2))
        for scale_idx in range(n_scale):
            # starlet transform of dirac impulse at a given scale
            dirac_scale2 = dirac_coeffs2[scale_idx, :, :]
            # Equation (16) of Joseph+19
            levels = signal.fftconvolve(dirac_scale2, FT_HT_noise2, mode='same')
            # save noise at each pixel for this scale
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        self._noise_levels_src = noise_levels

    def update_image_levels(self, num_pix_image, wavelet_transform_image):
        # starlet transform of a dirac impulse in image plane
        dirac = util.dirac_impulse(num_pix_image)
        dirac_coeffs2 = wavelet_transform_image(dirac)**2

        n_scale, n_pix1, npix2 = dirac_coeffs2.shape
        noise_levels = np.zeros((n_scale, n_pix1, npix2))
        for scale_idx in range(n_scale):
            scale_power2 = np.sum(dirac_coeffs2[scale_idx, :, :])
            noise_levels[scale_idx, :, :] = self.noise_map * np.sqrt(scale_power2)
        self._noise_levels_img = noise_levels

    def _initialise_regridding_error(self, data_image, image_pixel_scale, source_pixel_scale):
        _, self._regrid_error_prefac = util.regridding_error_map_squared(mag_map=None, data_image=data_image,
                                                                         image_pixel_scale=image_pixel_scale, source_pixel_scale=source_pixel_scale)

    def update_regridding_error(self, magnification_map):
        if not self.include_regridding_error:
            return  # do nothing
        regrid_error_map2, _ = util.regridding_error_map_squared(mag_map=magnification_map, noise_map2_prefactor=self._regrid_error_prefac)
        self._noise_map_with_regrid = np.sqrt(self._noise_map_data**2 + regrid_error_map2)
