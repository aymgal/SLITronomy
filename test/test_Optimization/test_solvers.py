__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Util import util
from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
import lenstronomy.Util.util as l_util


np.random.seed(18)


class TestSparseSolverSource(object):
    """
    tests the Lensing Operator classes
    """
    def setup(self):
        self.num_pix = 49  # cutout pixel size
        self.subgrid_res_source = 1
        self.num_pix_source = self.num_pix * self.subgrid_res_source

        # wavelets scales for lens and source
        self.n_scales_source = 4
        self.n_scales_lens = 3

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

        # list of source light profiles
        self.source_model = LightModel(['STARLETS'])
        self.kwargs_source = [{'coeffs': 1, 'n_scales': self.n_scales_source, 
                               'n_pixels': self.num_pix_source**2}]

        # list of lens light profiles
        self.lens_light_model = LightModel(['STARLETS'])
        self.kwargs_lens_light = [{'coeffs': 1, 'n_scales': self.n_scales_lens,
                                   'n_pixels': self.num_pix**2}]

        # source grid offsets
        self.kwargs_special = {
            'delta_x_source_grid': 0,
            'delta_y_source_grid': 0,
        }

        # define some mask
        self.likelihood_mask = np.zeros((self.num_pix, self.num_pix))
        self.likelihood_mask[5:-5, 5:-5] = 1

        # get a convolution operator
        kernel_pixel = np.zeros((self.num_pix, self.num_pix))
        kernel_pixel[int(self.num_pix/2), int(self.num_pix/2)] = 1  # just a dirac here
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_pixel}
        psf = PSF(**kwargs_psf)
        conv = PixelKernelConvolution(kernel_pixel)

        self.num_iter_source = 4
        self.num_iter_lens = 3
        self.num_iter_weights = 2

        # init the solver
        self.solver_source_ana = SparseSolverSource(data, lens_model, self.source_model, lens_light_model_class=None,
                 psf_class=psf, convolution_class=conv, likelihood_mask=self.likelihood_mask, lensing_operator='interpol',
                 subgrid_res_source=1, minimal_source_plane=False, fix_minimal_source_plane=True, 
                 use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                 sparsity_prior_norm=1, force_positivity=True, formulation='analysis',
                 verbose=False, show_steps=False,
                 max_threshold=5, max_threshold_high_freq=None, 
                 num_iter=self.num_iter_source, num_iter_weights=self.num_iter_weights)
        self.solver_lens_syn = SparseSolverSourceLens(data, lens_model, self.source_model, lens_light_model_class=self.lens_light_model,
                 psf_class=psf, convolution_class=conv, likelihood_mask=self.likelihood_mask, lensing_operator='interpol',
                 subgrid_res_source=1, minimal_source_plane=False, fix_minimal_source_plane=True, 
                 use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                 sparsity_prior_norm=1, force_positivity=True, formulation='synthesis',
                 verbose=False, show_steps=False,
                 max_threshold=5, max_threshold_high_freq=None, 
                 num_iter_source=self.num_iter_source, num_iter_lens=self.num_iter_lens, 
                 num_iter_weights=self.num_iter_weights)

    def test_solve(self):
        # source solver
        image_model, source_light, lens_light, coeffs, scales = \
            self.solver_source_ana.solve(self.kwargs_lens, self.kwargs_source, kwargs_special=self.kwargs_special)
        assert image_model.shape == self.image_data.shape
        assert source_light.shape == (self.num_pix_source, self.num_pix_source)
        assert lens_light is None
        assert coeffs.shape == (self.num_pix_source**2*self.n_scales_source,)
        assert len(scales) == 4

        # source+lens solver
        image_model, source_light, lens_light, coeffs, scales = \
            self.solver_lens_syn.solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                   kwargs_special=self.kwargs_special)
        assert image_model.shape == self.image_data.shape
        assert source_light.shape == (self.num_pix_source, self.num_pix_source)
        assert lens_light.shape == (self.num_pix, self.num_pix)
        assert coeffs.shape == (self.num_pix**2*self.n_scales_lens + self.num_pix_source**2*self.n_scales_source,)
        assert len(scales) == 4

        # get the track
        track = self.solver_lens_syn.track
        len_track_exp = (self.num_iter_source + 1)*self.num_iter_lens*self.num_iter_weights
        assert len(track['loss'][0, :]) == len_track_exp

        # access models
        image_model = self.solver_lens_syn.image_model()
        assert image_model.shape == self.image_data.shape
        # PSF is dirac so...
        npt.assert_equal(self.solver_lens_syn.image_model(unconvolved=True), 
                         self.solver_lens_syn.image_model(unconvolved=False))
        source_model = self.solver_lens_syn.source_model
        assert image_model.shape == (self.num_pix_source, self.num_pix_source)
        lens_light_model = self.solver_lens_syn.lens_light_model
        assert image_model.shape == (self.num_pix, self.num_pix)

        # loss function
        S = np.random.rand(self.num_pix_source, self.num_pix_source)
        HG = np.random.rand(self.num_pix, self.num_pix)
        loss = self.solver_source_ana.loss(S=S)
        assert loss > 0
        loss = self.solver_lens_syn.loss(S=S, HG=HG)
        assert loss > 0

        # reduced residuals map
        S = np.random.rand(self.num_pix_source, self.num_pix_source)
        HG = np.random.rand(self.num_pix, self.num_pix)
        red_res = self.solver_source_ana.reduced_residuals(S=S)
        assert red_res.shape == self.image_data.shape
        red_res = self.solver_lens_syn.reduced_residuals(S=S, HG=HG)
        assert red_res.shape == self.image_data.shape

        # L2-norm of difference of two arrays
        S_ = np.random.rand(self.num_pix_source, self.num_pix_source)
        assert self.solver_source_ana.norm_diff(S, S_) > 0

        # reduced chi2
        red_chi2 = self.solver_lens_syn.reduced_chi2(S=S, HG=HG)
        assert red_chi2 > 0
        assert self.solver_lens_syn.best_fit_reduced_chi2 == self.solver_lens_syn.reduced_chi2(S=source_model, HG=lens_light_model)

    def test_model_synthesis_analysis(self):
        _, S, _, _, _ = self.solver_source_ana.solve(self.kwargs_lens, self.kwargs_source, 
                                                     kwargs_special=self.kwargs_special)
        alpha_S = self.source_model.func_list[0].decomposition_2d(S, self.n_scales_source)
        ma = self.solver_source_ana.model_analysis(S)
        ms = self.solver_source_ana.model_synthesis(alpha_S)
        npt.assert_almost_equal(ma, ms, decimal=7)

        _, S, HG, _, _ = self.solver_lens_syn.solve(self.kwargs_lens, self.kwargs_source, 
                                                    self.kwargs_lens_light, kwargs_special=self.kwargs_special)
        alpha_S = self.source_model.func_list[0].decomposition_2d(S, self.n_scales_source)
        alpha_HG = self.lens_light_model.func_list[0].decomposition_2d(HG, self.n_scales_lens)
        ma = self.solver_lens_syn.model_analysis(S, HG)
        ms = self.solver_lens_syn.model_synthesis(alpha_S, alpha_HG)
        npt.assert_almost_equal(ma, ms, decimal=7)


if __name__ == '__main__':
    pytest.main()
