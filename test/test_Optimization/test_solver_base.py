__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Util import util
from slitronomy.Optimization.solver_base import SparseSolverBase

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
import lenstronomy.Util.util as l_util


np.random.seed(18)


class TestSparseSolverBase(object):
    """
    tests the Lensing Operator classes
    """
    def setup(self):
        self.num_pix = 49  # cutout pixel size
        self.subgrid_res_source = 1
        self.num_pix_source = self.num_pix * self.subgrid_res_source
        self.min_num_pix_source = 41

        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        
        self.image_data = np.random.rand(self.num_pix, self.num_pix)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': self.image_data,
            'background_rms': 0.01,
            'noise_map': 0.01 * np.ones_like(self.image_data),
        }
        data = ImageData(**kwargs_data)

        lens_model = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

        # wavelets scales for lens and source
        self.n_scales_source = 4
        self.n_scales_lens = 3

        # list of source light profiles
        source_model = LightModel(['SLIT_STARLETS'])
        self.kwargs_source = [{'coeffs': 1, 'n_scales': self.n_scales_source, 
                               'n_pixels': self.num_pix_source**2}]

        # list of lens light profiles
        lens_light_model = LightModel(['SLIT_STARLETS'])
        self.kwargs_lens_light = [{'coeffs': 1, 'n_scales': self.n_scales_lens,
                                   'n_pixels': self.num_pix**2}]

        # define some mask
        self.likelihood_mask = np.zeros((self.num_pix, self.num_pix))
        self.likelihood_mask[5:-5, 5:-5] = 1

        # get a convolution operator
        kernel_pixel = np.zeros((self.num_pix, self.num_pix))
        kernel_pixel[int(self.num_pix/2), int(self.num_pix/2)] = 1  # just a dirac here
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_pixel}
        psf = PSF(**kwargs_psf)
        numerics = NumericsSubFrame(pixel_grid=data, psf=psf)
        source_numerics = NumericsSubFrame(pixel_grid=data, psf=psf, supersampling_factor=self.subgrid_res_source)

        # init the solver
        self.solver = SparseSolverBase(data, lens_model, numerics, source_numerics, 
                 source_interpolation='bilinear', minimal_source_plane=True, 
                 use_mask_for_minimal_source_plane=True, min_num_pix_source=self.min_num_pix_source,
                 sparsity_prior_norm=1, force_positivity=True, formulation='analysis',
                 verbose=False, show_steps=False)
        self.solver.set_likelihood_mask(self.likelihood_mask)

    def test_solve(self):
        pass # tested in inheriting classes

    def test_plot_results(self):
        pass # should run the algorithm

    def test_apply_image_plane_mask(self):
        image_2d = np.ones_like(self.image_data)
        image_2d = self.solver.apply_image_plane_mask(image_2d)
        assert image_2d.shape == self.image_data.shape
        mask_zeros = np.where(self.likelihood_mask == 0)
        assert np.all(image_2d[mask_zeros] == 0)

    def test_apply_source_plane_mask(self):
        source_2d = np.ones((self.num_pix_source, self.num_pix_source))
        source_2d = self.solver.apply_source_plane_mask(source_2d)
        assert source_2d.shape == (self.num_pix_source, self.num_pix_source)

    # def test_project_on_original_grid_source(self):
    #     source_2d = np.ones((self.min_num_pix_source, self.min_num_pix_source))
    #     self.solver.lensingOperator.update_mapping(self.kwargs_lens)
    #     source_2d_proj = self.solver.project_on_original_grid_source(source_2d)
    #     assert source_2d_proj.shape == (self.num_pix_source, self.num_pix_source)

    def test_norm_diff(self):
        img1 = np.random.randn(10, 10)
        img2 = np.random.randn(10, 10)
        true_norm_diff = np.sqrt(np.sum((img1-img2)**2))
        npt.assert_almost_equal(self.solver.norm_diff(img1, img2), true_norm_diff, decimal=12)

    # def test_subtract_source_from_data(self):
    #     self.solver.lensingOperator.update_mapping(self.kwargs_lens)
    #     S = np.ones((self.min_num_pix_source, self.min_num_pix_source))
    #     self.solver.subtract_source_from_data(S)
    #     npt.assert_equal(self.solver.Y - self.solver.H(self.solver.F(S)), self.solver.Y_eff)
    #     self.solver.reset_data()
    #     npt.assert_equal(self.solver.Y, self.solver.Y_eff)

    def test_subtract_lens_from_data(self):
        HG = np.ones((self.num_pix, self.num_pix))
        self.solver.subtract_lens_from_data(HG)
        npt.assert_equal(self.solver.Y_tilde - HG, self.solver.Y_p)
        self.solver.reset_partial_data()
        npt.assert_equal(self.solver.Y_tilde, self.solver.Y_p)


class TestRaise(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRaise, self).__init__(*args, **kwargs)
        self.num_pix = 49  # cutout pixel size
        self.subgrid_res_source = 1
        self.num_pix_source = self.num_pix * self.subgrid_res_source

        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        image_data = np.random.rand(self.num_pix, self.num_pix)
        self.kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': image_data,
            'background_rms': 0.01,
            'noise_map': 0.01 * np.ones_like(image_data),
        }
        self.data = ImageData(**self.kwargs_data)
        self.lens_model = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]
        self.source_model = LightModel(['SLIT_STARLETS'])
        self.lens_light_model = LightModel(['SLIT_STARLETS'])
        self.kwargs_source = [{'coeffs': 1, 'n_scales': 4, 
                               'n_pixels': self.num_pix_source**2}]

        self.kwargs_lens_light = [{'coeffs': 1, 'n_scales': 4,
                                   'n_pixels': self.num_pix**2}]
        psf = PSF(psf_type='NONE')
        self.numerics = NumericsSubFrame(pixel_grid=self.data, psf=psf)
        self.source_numerics = NumericsSubFrame(pixel_grid=self.data, psf=psf, supersampling_factor=self.subgrid_res_source)
        self.solver = SparseSolverBase(self.data, self.lens_model, self.numerics, self.source_numerics)
        
    def test_raise(self):
        with self.assertRaises(ValueError):
            # wrong sparsitiy norm
            solver = SparseSolverBase(self.data, self.lens_model, self.numerics, self.source_numerics,
                                      sparsity_prior_norm=2)
        with self.assertRaises(ValueError):
            # non sqaure image
            kwargs_data = copy.deepcopy(self.kwargs_data)
            kwargs_data['image_data'] = np.ones((49, 60))
            kwargs_data['noise_map'] = 0.01 * np.ones((49, 60))
            data_nonsquare = ImageData(**kwargs_data)
            solver = SparseSolverBase(data_nonsquare, self.lens_model, self.numerics, self.source_numerics)
        with self.assertRaises(NotImplementedError):
            # solve is not fully implemented (on purpose) in the base class
            result = self.solver._ready()
        with self.assertRaises(NotImplementedError):
            # solve is not fully implemented (on purpose) in the base class
            result = self.solver._solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light)
        with self.assertRaises(ValueError):
            image_model = self.solver.image_model()
        with self.assertRaises(ValueError):
            image_model = self.solver.source_model
            
if __name__ == '__main__':
    pytest.main()
