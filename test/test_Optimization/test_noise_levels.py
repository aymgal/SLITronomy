__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Optimization.noise_levels import NoiseLevels
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Util import util

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import lenstronomy.Util.util as l_util


np.random.seed(18)

class TestNoiseLevels(object):
    """
    tests the Lensing Operator classes
    """
    def setup(self):
        self.num_pix = 49  # cutout pixel size
        self.subgrid_res_source = 2
        self.num_pix_source = self.num_pix * self.subgrid_res_source
        self.background_rms = 0.05
        self.noise_map = self.background_rms * np.ones((self.num_pix, self.num_pix))
        exposure_map = 1000

        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1,
                                                         inverse=False, left_lower=False)

        self.image_data = np.random.rand(self.num_pix, self.num_pix)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': Mpix2coord,
            'image_data': self.image_data,
            'background_rms': self.background_rms,
            'noise_map': self.noise_map,
            'exposure_time': exposure_map,
        }
        data = ImageData(**kwargs_data)

        gaussian_func = Gaussian()
        x, y = l_util.make_grid(41, 1)
        gaussian = gaussian_func.function(x, y, amp=1, sigma=0.02, center_x=0, center_y=0)
        self.psf_kernel = gaussian / gaussian.sum()

        lens_model = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

        # wavelets scales for lens and source
        self.n_scales_source = 4
        self.n_scales_lens = 3

        # list of source light profiles
        self.source_model = LightModel(['SLIT_STARLETS'])
        self.kwargs_source = [{'n_scales': self.n_scales_source}]

        # list of lens light profiles
        self.lens_light_model = LightModel(['SLIT_STARLETS'])
        self.kwargs_lens_light = [{'n_scales': self.n_scales_lens}]

        # get grid classes
        image_grid_class = NumericsSubFrame(data, PSF('NONE')).grid_class
        source_grid_class = NumericsSubFrame(data, PSF('NONE'), supersampling_factor=self.subgrid_res_source).grid_class

        # get a lensing operator
        self.lensing_op = LensingOperator(lens_model, image_grid_class, source_grid_class, self.num_pix)

        self.noise_class = NoiseLevels(data, subgrid_res_source=self.subgrid_res_source)
        self.noise_class_regrid = NoiseLevels(data, subgrid_res_source=self.subgrid_res_source, 
                                              include_regridding_error=True)
        self.noise_class_ps = NoiseLevels(data, subgrid_res_source=self.subgrid_res_source, 
                                          include_point_source_error=True)

    def test_background_rms(self):
        assert self.background_rms == self.noise_class.background_rms

    def test_noise_map(self):
        npt.assert_equal(self.noise_map, self.noise_class.noise_map)
        npt.assert_equal(self.noise_map, self.noise_class_regrid.noise_map)
        npt.assert_equal(self.noise_map, self.noise_class.effective_noise_map)

    def test_update_source_levels(self):
        wavelet_transform_source = lambda x: self.source_model.func_list[0].decomposition_2d(x, self.kwargs_source[0]['n_scales'])
        image2source_transform = lambda x: self.lensing_op.image2source_2d(x, kwargs_lens=self.kwargs_lens)
        upscale_transform = lambda x: x
        self.noise_class.update_source_levels(self.num_pix, self.num_pix_source, 
                                               wavelet_transform_source,
                                               image2source_transform, 
                                               upscale_transform,
                                               psf_kernel=None)  # without psf_kernel specified
        assert self.noise_class.levels_source.shape == (self.n_scales_source, self.num_pix_source, self.num_pix_source)
        self.noise_class.update_source_levels(self.num_pix, self.num_pix_source, 
                                               wavelet_transform_source,
                                               image2source_transform, 
                                               upscale_transform,
                                               psf_kernel=self.psf_kernel)
        assert self.noise_class.levels_source.shape == (self.n_scales_source, self.num_pix_source, self.num_pix_source)

    def test_update_image_levels(self):
        wavelet_transform_image = lambda x: self.lens_light_model.func_list[0].decomposition_2d(x, self.kwargs_lens_light[0]['n_scales'])
        self.noise_class.update_image_levels(self.num_pix, wavelet_transform_image)
        assert self.noise_class.levels_image.shape == (self.n_scales_lens, self.num_pix, self.num_pix)
        
    def test_update_regridding_error(self):
        magnification_map = self.lensing_op.magnification_map(self.kwargs_lens)
        self.noise_class_regrid.update_regridding_error(magnification_map)
        npt.assert_equal(self.noise_class_regrid.effective_noise_map, 
                         np.sqrt(self.noise_map**2 + self.noise_class_regrid.regridding_error_map**2))

    def test_update_point_source_error(self):
        ps_error_map = np.ones((self.num_pix, self.num_pix))
        self.noise_class_ps.update_point_source_error(ps_error_map)
        npt.assert_equal(self.noise_class_ps.effective_noise_map, 
                         np.sqrt(self.noise_map**2 + self.noise_class_ps.point_source_error_map**2))

    def test_re_estimate_noise_map_for_ps(self):
        noise_map_before = np.copy(self.noise_class_ps.noise_map)
        ps_mask = np.zeros((self.num_pix, self.num_pix))
        ps_mask[10:10, 10:10] = 1  # fake a mask
        ps_model = np.ones((self.num_pix, self.num_pix))  # fake a point source model
        self.noise_class_ps.re_estimate_noise_map_for_ps(self.image_data, ps_mask, ps_model)
        npt.assert_raises(AssertionError, npt.assert_array_equal, 
                          self.noise_class_ps.noise_map, noise_map_before)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            num_pix = 49  # cutout pixel size
            background_rms = 0.05
            subgrid_res_source = 2
            delta_pix = 0.24
            _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
                = l_util.make_grid_with_coordtransform(numPix=num_pix, deltapix=delta_pix, subgrid_res=1,
                                                             inverse=False, left_lower=False)

            kwargs_data = {
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': Mpix2coord,
                'image_data': np.zeros((20, 20)),
                'background_rms': background_rms,
                'noise_map': np.zeros((20, 20)),
            }
            data = ImageData(**kwargs_data)
            noise_class = NoiseLevels(data, subgrid_res_source=subgrid_res_source)
            magnification_map = np.zeros((20, 20))
            noise_class.update_regridding_error(magnification_map)

if __name__ == '__main__':
    pytest.main()
