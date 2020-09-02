__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Util import util

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as l_util


class TestLensingOperator(object):
    """
    tests the Lensing Operator class
    """
    def setup(self):
        self.num_pix = 25  # cutout pixel size
        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1,
                                                   inverse=False, left_lower=False)
        kwargs_data = {
            #'background_rms': background_rms,
            #'exposure_time': np.ones((self.num_pix, self.num_pix)) * exp_time,  # individual exposure time/weight per pixel
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix))
        }
        self.data = ImageData(**kwargs_data)

        self.lens_model = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]
        self.kwargs_lens_null = [{'theta_E': 0, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}]

        # PSF specification
        kwargs_psf = {'psf_type': 'NONE'}
        self.psf = PSF(**kwargs_psf)

        # list of source light profiles
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_sersic_ellipse_source = {'amp': 2000, 'R_sersic': 0.6, 'n_sersic': 1, 'e1': 0.1, 'e2': 0.1,
                                        'center_x': 0.3, 'center_y': 0.3}
        kwargs_source = [kwargs_sersic_ellipse_source]
        source_model = LightModel(light_model_list=source_model_list)

        # list of lens light profiles
        lens_light_model_list = []
        kwargs_lens_light = [{}]
        lens_light_model = LightModel(light_model_list=lens_light_model_list)

        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        self.image_model = ImageModel(self.data, self.psf, self.lens_model, source_model,
                                 lens_light_model, point_source_class=None, kwargs_numerics=kwargs_numerics)
        self.image_grid_class = self.image_model.ImageNumerics.grid_class
        self.source_grid_class_default = NumericsSubFrame(self.data, self.psf).grid_class

        # create simulated image
        image_sim_no_noise = self.image_model.image(self.kwargs_lens, kwargs_source, kwargs_lens_light)
        self.source_light_lensed = image_sim_no_noise
        self.data.update_data(image_sim_no_noise)

        # source only, in source plane, on same grid as data
        self.source_light_delensed = self.image_model.source_surface_brightness(kwargs_source, unconvolved=False, de_lensed=True)

        # define some auto mask for tests
        self.likelihood_mask = np.zeros_like(self.source_light_lensed)
        self.likelihood_mask[self.source_light_lensed > 0.1 * self.source_light_lensed.max()] = 1

    def test_matrix_product(self):
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='nearest', matrix_prod=False)
        lensing_op.update_mapping(self.kwargs_lens)

        lensing_op_mat = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                         source_interpolation='nearest', matrix_prod=True)
        lensing_op_mat.update_mapping(self.kwargs_lens)

        source_1d = util.image2array(self.source_light_delensed)
        image_1d = util.image2array(self.source_light_lensed)

        npt.assert_equal(lensing_op.source2image(source_1d), lensing_op_mat.source2image(source_1d))
        npt.assert_equal(lensing_op.image2source(image_1d), lensing_op_mat.image2source(image_1d))

    def test_minimal_source_plane(self):
        source_1d = util.image2array(self.source_light_delensed)

        # test with no mask
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='nearest', minimal_source_plane=True)
        lensing_op.update_mapping(self.kwargs_lens)
        image_1d = util.image2array(self.source_light_lensed)
        assert lensing_op.image2source(image_1d).size < source_1d.size

        # test with mask
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='nearest', minimal_source_plane=True)
        lensing_op.set_likelihood_mask(self.likelihood_mask)
        lensing_op.update_mapping(self.kwargs_lens)
        image_1d = util.image2array(self.source_light_lensed)
        assert lensing_op.image2source(image_1d).size < source_1d.size

        # for 'bilinear' operator, only works with no mask (for now)
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='bilinear', minimal_source_plane=True)
        lensing_op.update_mapping(self.kwargs_lens)
        image_1d = util.image2array(self.source_light_lensed)
        assert lensing_op.image2source(image_1d).size < source_1d.size

    def test_legacy_mapping(self):
        """testing than image2source / source2image are close to the parametric mapping""" 
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='nearest_legacy')
        lensing_op.update_mapping(self.kwargs_lens)

        source_1d = util.image2array(self.source_light_delensed)
        image_1d = util.image2array(self.source_light_lensed)

        source_1d_lensed = lensing_op.source2image(source_1d)
        image_1d_delensed = lensing_op.image2source(image_1d)
        assert source_1d_lensed.shape == image_1d.shape
        assert image_1d_delensed.shape == source_1d.shape

        npt.assert_almost_equal(source_1d_lensed/source_1d_lensed.max(), image_1d/image_1d.max(), decimal=0.6)
        npt.assert_almost_equal(image_1d_delensed/image_1d_delensed.max(), source_1d/source_1d.max(), decimal=0.6)

    def test_simple_mapping(self):
        """testing than image2source / source2image are close to the parametric mapping""" 
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='nearest')
        lensing_op.update_mapping(self.kwargs_lens)

        source_1d = util.image2array(self.source_light_delensed)
        image_1d = util.image2array(self.source_light_lensed)

        source_1d_lensed = lensing_op.source2image(source_1d)
        image_1d_delensed = lensing_op.image2source(image_1d)
        assert source_1d_lensed.shape == image_1d.shape
        assert image_1d_delensed.shape == source_1d.shape

        npt.assert_almost_equal(source_1d_lensed/source_1d_lensed.max(), image_1d/image_1d.max(), decimal=0.6)
        npt.assert_almost_equal(image_1d_delensed/image_1d_delensed.max(), source_1d/source_1d.max(), decimal=0.6)

    def test_interpol_mapping(self):
        """testing than image2source / source2image are close to the parametric mapping""" 
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix,
                                     source_interpolation='bilinear')
        lensing_op.update_mapping(self.kwargs_lens)

        source_1d = util.image2array(self.source_light_delensed)
        image_1d = util.image2array(self.source_light_lensed)

        source_1d_lensed = lensing_op.source2image(source_1d)
        image_1d_delensed = lensing_op.image2source(image_1d)
        assert source_1d_lensed.shape == image_1d.shape
        assert image_1d_delensed.shape == source_1d.shape

        npt.assert_almost_equal(source_1d_lensed/source_1d_lensed.max(), image_1d/image_1d.max(), decimal=0.8)
        npt.assert_almost_equal(image_1d_delensed/image_1d_delensed.max(), source_1d/source_1d.max(), decimal=0.8)

    def test_source2image(self):
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix)
        source_1d = util.image2array(self.source_light_delensed)
        source_1d_lensed = lensing_op.source2image(source_1d, kwargs_lens=self.kwargs_lens)
        assert len(source_1d_lensed.shape) == 1

        source_2d = self.source_light_delensed
        source_2d_lensed = lensing_op.source2image_2d(source_2d, kwargs_lens=self.kwargs_lens, update_mapping=True)
        assert len(source_2d_lensed.shape) == 2

    def test_image2source(self):
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix)
        image_1d = util.image2array(self.source_light_lensed)
        image_1d_delensed = lensing_op.image2source(image_1d, kwargs_lens=self.kwargs_lens)
        assert len(image_1d_delensed.shape) == 1

        image_2d = self.source_light_lensed
        image_2d_delensed = lensing_op.image2source_2d(image_2d, kwargs_lens=self.kwargs_lens, update_mapping=True)
        assert len(image_2d_delensed.shape) == 2

    def test_source_plane_coordinates(self):
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix)
        theta_x, theta_y = lensing_op.source_plane_coordinates
        assert theta_x.size == self.num_pix**2
        assert theta_y.size == self.num_pix**2

        subgrid_res = 2
        source_grid_class = NumericsSubFrame(self.data, self.psf, supersampling_factor=subgrid_res).grid_class
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, source_grid_class, self.num_pix)
        theta_x, theta_y = lensing_op.source_plane_coordinates
        assert theta_x.size == self.num_pix**2 * subgrid_res**2
        assert theta_y.size == self.num_pix**2 * subgrid_res**2

    def test_image_plane_coordinates(self):
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix)
        theta_x, theta_y = lensing_op.image_plane_coordinates
        assert theta_x.size == self.num_pix**2
        assert theta_y.size == self.num_pix**2

    def test_find_source_pixel(self):
        lensing_op = LensingOperator(self.lens_model, self.image_grid_class, self.source_grid_class_default, self.num_pix, 
                                     source_interpolation='nearest')
        beta_x, beta_y = self.lens_model.ray_shooting(lensing_op.imagePlane.theta_x, lensing_op.imagePlane.theta_y,
                                                     self.kwargs_lens)
        i = 10
        j = lensing_op._find_source_pixel_nearest_legacy(i, beta_x, beta_y)
        assert (isinstance(j, int) or isinstance(j, np.int64))


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            num_pix = 10
            data = ImageData(np.zeros((num_pix, num_pix)))
            lens_model = LensModel(['SPEP'])
            image_grid_class = NumericsSubFrame(data, PSF('NONE')).grid_class
            source_grid_class = NumericsSubFrame(data, PSF('NONE')).grid_class
            lensing_op = LensingOperator(lens_model, image_grid_class, source_grid_class, num_pix,
                                         source_interpolation='sth')


if __name__ == '__main__':
    pytest.main()
