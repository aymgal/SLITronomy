__author__ = 'aymgal'

import numpy as np
from scipy import signal
from slitronomy.Util import util
from skimage import filters


class NoiseLevels(object):

    """
    Handle noise properties and compute noise levels in wavelets space, 
    taking into account lensing and optionally blurring and regridding error for pixelated reconstructions.
    """

    def __init__(self, data_class, subgrid_res_source=1, 
                 include_regridding_error=False, include_point_source_error=False):
        """
        :param subgrid_res_source: resolution factor between image plane and source plane
        :param boost_where_zero: sets the multiplcative factor in fron tof the average noise levels
        at locations where noise is 0
        :param include_regridding_error: if True, includes the regridding error controbution in noise covariance.
        See Suyu et al. 2009 (https://ui.adsabs.harvard.edu/abs/2009ApJ...691..277S/abstract) for details.
        """
        # background noise
        self._background_rms = data_class.background_rms
        # exposure map / time
        self._exposure_map = data_class.exposure_map
        # noise full covariance \simeq sqrt(poisson_rms^2 + gaussian_rms^2)
        self._noise_map_data = np.sqrt(data_class.C_D)
        self.include_regridding_error = include_regridding_error
        if self.include_regridding_error:
            self._initialise_regridding_error(data_class.data, data_class.pixel_width, 
                                              data_class.pixel_width/subgrid_res_source)
        self.include_point_source_error = include_point_source_error

    @property
    def background_rms(self):
        return self._background_rms

    @property
    def effective_noise_map(self):
        """Add quadratically the regridding error map and point source error map"""
        if self.include_regridding_error:
            regrid_error_map = self.regridding_error_map
        else:
            regrid_error_map = np.zeros_like(self.noise_map)
        if self.include_point_source_error:
            ps_error_map = self.point_source_error_map
        else:
            ps_error_map = np.zeros_like(self.noise_map)
        return np.sqrt(self.noise_map**2 + regrid_error_map**2 + ps_error_map**2)

    @property
    def noise_map(self):
        return self._noise_map_data

    @property
    def regridding_error_map(self):
        if not hasattr(self, '_regridding_error_map'):
            raise ValueError("Regridding error map has not be updated with magnification map")
        return self._regridding_error_map

    @property
    def point_source_error_map(self):
        if not hasattr(self, '_ps_error_map'):
            raise ValueError("Point source error map has not been passed to solver")
        return self._ps_error_map

    def re_estimate_noise_map_for_ps(self, data, ps_mask, ps_model):
        data_pos = (data - ps_model)*ps_mask + data*(1-ps_mask)
        data_pos[data_pos < 0] = 0.
        sigma = data_pos / self._exposure_map + self.background_rms ** 2
        self._noise_map_data = np.sqrt(sigma) 
        # TODO: create another field to backup the original noise map (which would be consistent with the original data)

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
                             image2source_transform, upscale_transform, psf_kernel=None):
        # get transposed blurring operator
        if psf_kernel is None:
            psf_T = util.dirac_impulse(num_pix_image)
        else:
            psf_T = psf_kernel.T

        # here we don't use the 'effective' noise map (that includes regridding and point source errors)
        # so that the wavelet thresholding is only measured based on the 'original' data noise
        noise_map = self.noise_map

        # map noise values to source plane
        noise_diag = noise_map * np.sqrt(np.sum(psf_T**2))
        noise_diag_up = upscale_transform(noise_diag)
        noise_source = image2source_transform(noise_diag_up)

        # we gaussian filter the noise map with sigma adatpted to supersampling factor
        # to fill adequately pixels that are not mapped to any image plane pixels
        filter_width = num_pix_source / num_pix_image
        noise_source_filtered = filters.gaussian(noise_source, sigma=filter_width)
        # renormalize amplitudes
        noise_source = noise_source_filtered * noise_source.max() / noise_source_filtered.max()

        # old way:
        # introduce artitifically noise to pixels where there are not signal in source plane
        # to ensure threshold of starlet coefficients at these locations
        # boost_where_zero = 10
        # noise_source[noise_source == 0] = boost_where_zero * np.mean(noise_source[noise_source != 0])

        # \Gamma^2 in  Equation (16) of Joseph+19
        noise_source2 = noise_source**2

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
            levels2 = signal.fftconvolve(dirac_scale2, noise_source2, mode='same')
            levels = np.sqrt(np.abs(levels2))

            # save noise at each pixel for this scale
            noise_levels[scale_idx, :, :] = levels
        self._noise_levels_src = noise_levels

    def update_image_levels(self, num_pix_image, wavelet_transform_image):
        # starlet transform of a dirac impulse in image plane
        dirac = util.dirac_impulse(num_pix_image)
        dirac_coeffs2 = wavelet_transform_image(dirac)**2

        # TODO: if it happens that noise_map is a constant value, not need to initialise a full array
        noise_map = self.effective_noise_map  #self.noise_map

        n_scale, n_pix1, npix2 = dirac_coeffs2.shape
        noise_levels = np.zeros((n_scale, n_pix1, npix2))
        for scale_idx in range(n_scale):
            scale_power2 = np.sum(dirac_coeffs2[scale_idx, :, :])
            noise_levels[scale_idx, :, :] = noise_map * np.sqrt(scale_power2)

        self._noise_levels_img = noise_levels

    def _initialise_regridding_error(self, data_image, image_pixel_scale, source_pixel_scale):
        _, self._regrid_error_prefac = util.regridding_error_map_squared(mag_map=None, data_image=data_image,
                                                                         image_pixel_scale=image_pixel_scale, 
                                                                         source_pixel_scale=source_pixel_scale)

    def update_regridding_error(self, magnification_map):
        if not hasattr(self, '_regrid_error_prefac'):
            raise ValueError("Regridding error has not been initialised properly")
        regrid_error_map2, _ = util.regridding_error_map_squared(mag_map=magnification_map, 
                                                                 noise_map2_prefactor=self._regrid_error_prefac)
        self._regridding_error_map = np.sqrt(regrid_error_map2)

    def update_point_source_error(self, ps_error_map):
        self._ps_error_map = ps_error_map
