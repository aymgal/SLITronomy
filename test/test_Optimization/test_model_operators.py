__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Util import util

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
import lenstronomy.Util.util as l_util


np.random.seed(18)

class TestModelOperators(object):
    """
    tests the Lensing Operator classes
    """
    def setup(self):
        self.num_pix = 49  # cutout pixel size
        self.subgrid_res_source = 2
        self.num_pix_source = self.num_pix * self.subgrid_res_source

        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1,
                                                         inverse=False, left_lower=False)

        self.image_data = np.random.rand(self.num_pix, self.num_pix)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': Mpix2coord,
            'image_data': self.image_data,
        }
        data = ImageData(**kwargs_data)

        lens_model = LensModel(['SPEP'])
        kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

        # wavelets scales for lens and source
        self.n_scales_source = 4
        self.n_scales_lens = 3

        # list of source light profiles
        source_model = LightModel(['STARLETS'])
        self.kwargs_source = [{'n_scales': self.n_scales_source}]

        # list of lens light profiles
        lens_light_model = LightModel(['STARLETS'])
        self.kwargs_lens_light = [{'n_scales': self.n_scales_lens}]

        # define some mask
        likelihood_mask = np.ones((self.num_pix, self.num_pix))

        # get a lensing operator
        self.lensing_op = LensingOperator(data, lens_model, subgrid_res_source=self.subgrid_res_source)
        self.lensing_op.update_mapping(kwargs_lens)

        # get a convolution operator
        kernel_pixel = np.zeros((self.num_pix, self.num_pix))
        kernel_pixel[int(self.num_pix/2), int(self.num_pix/2)] = 1  # just a dirac here
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_pixel}
        psf = PSF(**kwargs_psf)
        self.numerics = NumericsSubFrame(pixel_grid=data, psf=psf)

        self.model_op = ModelOperators(data, self.lensing_op, self.numerics,
                                       likelihood_mask=likelihood_mask)
        self.model_op.add_source_light(source_model)
        self.model_op.add_lens_light(lens_light_model)
        self.model_op_nolens = ModelOperators(data, self.lensing_op, self.numerics,
                                       likelihood_mask=likelihood_mask)
        self.model_op_nolens.add_source_light(source_model)

        # define some test images in direct space
        self.X_s = np.random.rand(self.num_pix_source, self.num_pix_source)  # source light
        self.X_l = np.random.rand(self.num_pix, self.num_pix)  # lens light

        # define some test images in wavelets space
        self.alpha_s = np.random.rand(self.n_scales_source, self.num_pix_source, self.num_pix_source)  # source light
        self.alpha_l = np.random.rand(self.n_scales_lens, self.num_pix, self.num_pix)  # lens light

    def test_set_wavelet_scales(self):
        self.model_op.set_source_wavelet_scales(self.n_scales_source)
        Phi_T_s_X = self.model_op.Phi_T_s(self.X_s)
        self.model_op.set_lens_wavelet_scales(self.n_scales_lens)
        Phi_T_l_X = self.model_op.Phi_T_l(self.X_l)
        # test that transformed image has the right shape in terms of number of scales
        assert Phi_T_s_X.shape[0] == self.n_scales_source
        assert Phi_T_l_X.shape[0] == self.n_scales_lens

    def test_subtract_from_data_and_reset(self):
        image_to_subtract = np.eye(self.num_pix, self.num_pix)
        self.model_op.subtract_from_data(image_to_subtract)
        npt.assert_equal(self.model_op.Y, self.image_data)
        npt.assert_equal(self.model_op.Y_eff, self.image_data - image_to_subtract)
        self.model_op.reset_data()
        npt.assert_equal(self.model_op.Y, self.image_data)
        npt.assert_equal(self.model_op.Y_eff, self.image_data)

    def test_spectral_norm_source(self):
        self.model_op.set_source_wavelet_scales(self.n_scales_source)
        npt.assert_almost_equal(self.model_op.spectral_norm_source, 0.999, decimal=3)

    def test_spectral_norm_lens(self):
        self.model_op.set_lens_wavelet_scales(self.n_scales_lens)
        npt.assert_almost_equal(self.model_op.spectral_norm_lens, 0.999, decimal=3)

    def test_data_terms(self):
        npt.assert_equal(self.model_op.Y, self.image_data)
        npt.assert_equal(self.model_op.Y_eff, self.image_data)

    def test_convolution(self):
        H_X_s = self.model_op.H(self.X_s)
        npt.assert_equal(H_X_s, self.numerics.convolution_class.convolution2d(self.X_s))
        H_T_X_s = self.model_op.H_T(self.X_s)
        conv_transpose = self.numerics.convolution_class.copy_transpose()
        npt.assert_equal(H_T_X_s, conv_transpose.convolution2d(self.X_s))

    def test_lensing(self):
        F_X_s = self.model_op.F(self.X_s)
        npt.assert_equal(F_X_s, self.lensing_op.source2image_2d(self.X_s))
        F_T_X_l = self.model_op.F_T(self.X_l)
        npt.assert_equal(F_T_X_l, self.lensing_op.image2source_2d(self.X_l))

    def test_wavelet_transform(self):
        # TODO : do more accurate tests here
        self.model_op.set_source_wavelet_scales(self.n_scales_source)
        self.model_op.set_lens_wavelet_scales(self.n_scales_lens)
        Phi_alpha_s = self.model_op.Phi_s(self.alpha_s)
        Phi_alpha_l = self.model_op.Phi_l(self.alpha_l)
        assert Phi_alpha_s.shape == (self.num_pix*self.subgrid_res_source, self.num_pix*self.subgrid_res_source)
        assert Phi_alpha_l.shape == (self.num_pix, self.num_pix)
        Phi_T_X_s = self.model_op.Phi_T_s(self.X_s)
        Phi_T_X_l = self.model_op.Phi_T_l(self.X_l)
        assert Phi_T_X_s.shape == (self.n_scales_source, self.num_pix*self.subgrid_res_source, self.num_pix*self.subgrid_res_source)
        assert Phi_T_X_l.shape == (self.n_scales_lens, self.num_pix, self.num_pix)


class TestRaise(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRaise, self).__init__(*args, **kwargs)
        self.num_pix = 10
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=0.5, subgrid_res=1,
                                                   inverse=False, left_lower=False)
        kwargs_data_nonsquare = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix+10)),  # non-square image
        }
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix)),  # non-square image
        }
        self.data_nonsquare = ImageData(**kwargs_data_nonsquare)
        self.data = ImageData(**kwargs_data)
        self.numerics = NumericsSubFrame(pixel_grid=self.data, psf=PSF(psf_type='NONE'))
        lens_model = LensModel(['SPEP'])
        self.source_model_class = LightModel(['STARLETS'])
        self.lens_light_model_class = LightModel(['STARLETS'])
        self.lensing_op = LensingOperator(self.data, lens_model)
        self.model_op = ModelOperators(self.data, self.lensing_op, self.numerics)
        self.model_op.add_lens_light(self.lens_light_model_class)
        self.model_op_nolens = ModelOperators(self.data, self.lensing_op, self.numerics)

    def test_raise(self):
        with self.assertRaises(ValueError):
            # no wavelet scales was set, so should raise errors
            X_s = np.random.rand(self.num_pix, self.num_pix)
            Phi_T_s_X = self.model_op.Phi_T_s(X_s)
        with self.assertRaises(ValueError):
            # no wavelet scales was set, so should raise errors
            X_l = np.random.rand(self.num_pix, self.num_pix)
            Phi_T_l_X = self.model_op.Phi_T_l(X_l)
        with self.assertRaises(ValueError):
            # no wavelet scales was set, so should raise errors
            alpha_s = np.random.rand(3, self.num_pix, self.num_pix)
            Phi_s_alpha = self.model_op.Phi_s(alpha_s)
        with self.assertRaises(ValueError):
            # no wavelet scales was set, so should raise errors
            alpha_l = np.random.rand(3, self.num_pix, self.num_pix)
            Phi_l_alpha = self.model_op.Phi_l(alpha_l)
        with self.assertRaises(ValueError):
            # no lens profile was set, so should raise an error
            X_l = np.random.rand(self.num_pix, self.num_pix)
            Phi_T_l_X = self.model_op_nolens.Phi_T_l(X_l)
        with self.assertRaises(ValueError):
            # no lens profile was set, so should raise an error
            alpha_l = np.random.rand(3, self.num_pix, self.num_pix)
            Phi_l_alpha = self.model_op_nolens.Phi_l(alpha_l)
        with self.assertRaises(ValueError):
            # ModelOperators init should raise an error with non square data image
            model_op_error = ModelOperators(self.data_nonsquare, self.lensing_op, self.numerics)


if __name__ == '__main__':
    pytest.main()
