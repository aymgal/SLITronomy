__author__ = 'aymgal'

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Util import util
from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens
from slitronomy.Optimization.solver_source_ps import SparseSolverSourcePS

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.util as l_util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.param_util as param_util


np.random.seed(18)


class TestSparseSolverSource(object):
    """
    tests the Sparse Solver classes
    """
    def setup(self):

        # data specifics
        sigma_bkg = .05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        self.num_pix = 49  # cutout pixel size
        delta_pix = 0.24  # pixel size in arcsec (area per pixel = delta_pix**2)
        fwhm = 0.5  # full width half max of PSF

        # supersampling factor for source plane
        self.subgrid_res_source = 1
        self.num_pix_source = self.num_pix * self.subgrid_res_source

        # wavelets scales for lens and source
        self.n_scales_source = 4
        self.n_scales_lens = 3

        # prepare data simulation
        kwargs_data = sim_util.data_configure_simple(self.num_pix, delta_pix, exp_time, sigma_bkg, inverse=True)
        data_class = ImageData(**kwargs_data)
            
        # generate sa pixelated gaussian PSF kernel
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': delta_pix}
        psf_class = PSF(**kwargs_psf)
        kernel = psf_class.kernel_point_source
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel, 'psf_error_map': np.ones_like(kernel) * 0.001}
        psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        self.lens_model_class = LensModel(lens_model_list=lens_model_list)
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)

        # list of lens light profiles
        point_source_class = PointSource(['LENSED_POSITION'])
        lens_eq_solver = LensEquationSolver(lensModel=self.lens_model_class)
        ra_image, dec_image = lens_eq_solver.image_position_from_source(sourcePos_x=kwargs_source[0]['center_x'],
                                                                   sourcePos_y=kwargs_source[0]['center_y'], 
                                                                   kwargs_lens=self.kwargs_lens)
        point_amp = np.ones_like(ra_image)
        kwargs_ps = [{'ra_image': ra_image, 'dec_image': dec_image, 'point_amp': point_amp}]

        # simulate data
        kwargs_numerics = {'supersampling_factor': 1}
        imageModel = ImageModel(data_class, psf_class, self.lens_model_class, source_model_class, lens_light_model_class, point_source_class, 
                                kwargs_numerics=kwargs_numerics)
        self.image_sim = sim_util.simulate_simple(imageModel, self.kwargs_lens, kwargs_source,
                                       kwargs_lens_light, kwargs_ps)
        data_class.update_data(self.image_sim)

        # retrieve the point source data only (for initial guess for source+PS solver)
        self.ps_sim = imageModel.image(self.kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                       source_add=False, lens_light_add=False, point_source_add=True)

        # define some mask
        self.likelihood_mask = np.zeros((self.num_pix, self.num_pix))
        self.likelihood_mask[5:-5, 5:-5] = 1

        # get a numerics classes
        numerics = NumericsSubFrame(pixel_grid=data_class, psf=psf_class)
        source_numerics = NumericsSubFrame(pixel_grid=data_class, psf=psf_class, supersampling_factor=self.subgrid_res_source)

        self.num_iter_source = 20
        self.num_iter_lens = 10
        self.num_iter_global = 7
        self.num_iter_weights = 2

        # source grid offsets
        self.kwargs_special = {
            'delta_x_source_grid': 0,
            'delta_y_source_grid': 0,
        }

        # init the solvers

        # SOLVER SOURCE, with analysis formulation
        self.source_model_class = LightModel(['SLIT_STARLETS'])
        self.kwargs_source = [{'n_scales': self.n_scales_source}]
        self.solver_source_ana = SparseSolverSource(data_class, self.lens_model_class, numerics, source_numerics, 
                                                    self.source_model_class, 
                                                    source_interpolation='bilinear', minimal_source_plane=False, 
                                                    use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                                                    sparsity_prior_norm=1, force_positivity=True, formulation='analysis',
                                                    verbose=False, show_steps=False,
                                                    min_threshold=5, threshold_increment_high_freq=1, threshold_decrease_type='exponential', 
                                                    num_iter_source=self.num_iter_source, num_iter_weights=self.num_iter_weights)
        self.solver_source_ana.set_likelihood_mask(self.likelihood_mask)
        
        # SOLVER SOURCE + LENS, with synthesis formulation
        self.lens_light_model_class = LightModel(['SLIT_STARLETS'])
        self.kwargs_lens_light = [{'n_scales': self.n_scales_lens}]
        self.solver_lens_syn = SparseSolverSourceLens(data_class, self.lens_model_class, numerics, source_numerics, 
                                                      self.source_model_class, self.lens_light_model_class,
                                                      source_interpolation='bilinear', minimal_source_plane=False, 
                                                      use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                                                      sparsity_prior_norm=1, force_positivity=True, formulation='synthesis',
                                                      verbose=False, show_steps=False,
                                                      min_threshold=3, threshold_increment_high_freq=1, threshold_decrease_type='linear', 
                                                      num_iter_global=self.num_iter_global, num_iter_source=self.num_iter_source, num_iter_lens=self.num_iter_lens, num_iter_weights=self.num_iter_weights)
        self.solver_lens_syn.set_likelihood_mask(self.likelihood_mask)

        # SOLVER SOURCE + PS, with analysis formulation
        self.kwargs_ps = kwargs_ps.copy()
        self.n_point_sources = len(point_amp)
        self.solver_source_ps_ana = SparseSolverSourcePS(data_class, self.lens_model_class, numerics, source_numerics, 
                                                         self.source_model_class, 
                                                         source_interpolation='bilinear', minimal_source_plane=False, 
                                                         use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                                                         sparsity_prior_norm=1, force_positivity=True, formulation='analysis',
                                                         verbose=False, show_steps=False,
                                                         min_threshold=5, threshold_increment_high_freq=1, threshold_decrease_type='exponential', 
                                                         num_iter_source=self.num_iter_source, num_iter_global=self.num_iter_global, num_iter_weights=self.num_iter_weights)
        self.solver_source_ps_ana.set_likelihood_mask(self.likelihood_mask)
        # TODO: for now it's a dummy test, with no linear amplitude solver for point sources
        def _dummy_ps_linear_solver(sparse_model, kwargs_lens=None, kwargs_ps=None, kwargs_special=None, inv_bool=False):
            model = np.copy(sparse_model)
            model_error = None
            cov_param = None
            param = np.ones_like(ra_image)
            return model, model_error, cov_param, param
        self.solver_source_ps_ana.set_point_source_solver_func(_dummy_ps_linear_solver)

    def test_solve_source_analysis(self):
        # source solver
        image_model, param, logL_penalty = \
            self.solver_source_ana.solve(self.kwargs_lens, self.kwargs_source, kwargs_special=self.kwargs_special)
        assert image_model.shape == self.image_sim.shape
        assert len(param) == self.num_pix_source**2*self.n_scales_source

        # get the track
        track = self.solver_source_ana.track
        len_track_theory = self.num_iter_source*self.num_iter_weights
        assert len(track['loss'][0, :]) == len_track_theory

        # access models
        image_model = self.solver_source_ana.image_model()
        assert image_model.shape == self.image_sim.shape
        source_light = self.solver_source_ana.source_model
        assert source_light.shape == (self.num_pix_source, self.num_pix_source)

        S = source_light

        # loss function
        # self.solver_lens_syn.reset_data()
        # loss = self.solver_source_ana.loss(S=S)
        # assert loss > 0

        # reduced residuals map
        red_res = self.solver_source_ana.normalized_residuals(S=S)
        assert red_res.shape == self.image_sim.shape

        # L2-norm of difference of two arrays
        S_ = np.random.rand(self.num_pix_source, self.num_pix_source)
        assert self.solver_source_ana.norm_diff(S, S_) > 0

        # reduced chi2
        red_chi2 = self.solver_source_ana.reduced_chi2(S=S)
        assert red_chi2 > 0
        assert self.solver_source_ana.best_fit_reduced_chi2 == self.solver_source_ana.reduced_chi2(S=S)

        # synthesis and analysis models
        alpha_S = self.source_model_class.func_list[0].decomposition_2d(S, n_scales=self.n_scales_source)
        ma = self.solver_source_ana.model_analysis(S)
        ms = self.solver_source_ana.model_synthesis(alpha_S)
        npt.assert_almost_equal(ma, ms, decimal=4)

        # test plot results
        fig = self.solver_source_ana.plot_results()
        plt.close()
        source_truth = np.ones_like(source_light)  # we don't test whether the source reconstruction is ok here
        source_model_list = [source_light, source_light]
        name_list = ['model 1', 'model 2']
        fig = self.solver_source_ana.plot_source_residuals_comparison(source_truth, source_model_list, name_list)
        plt.close()

    def test_solve_source_lens_synthesis(self):
        # source+lens solver
        image_model, param, logL_penalty = \
            self.solver_lens_syn.solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                   kwargs_special=self.kwargs_special)
        assert image_model.shape == self.image_sim.shape
        assert len(param) == self.num_pix**2*self.n_scales_lens + self.num_pix_source**2*self.n_scales_source

        # get the track
        track = self.solver_lens_syn.track
        len_track_theory = (self.num_iter_source + self.num_iter_lens)*self.num_iter_global*self.num_iter_weights
        assert len(track['loss'][0, :]) == len_track_theory

        # access models
        image_model = self.solver_lens_syn.image_model()
        assert image_model.shape == self.image_sim.shape
        source_light = self.solver_lens_syn.source_model
        assert source_light.shape == (self.num_pix_source, self.num_pix_source)
        lens_light = self.solver_lens_syn.lens_light_model
        assert image_model.shape == (self.num_pix, self.num_pix)

        S, HG = source_light, lens_light

        # loss function
        # self.solver_lens_syn.reset_data()
        # loss = self.solver_lens_syn.loss(S=S, HG=HG)
        # assert loss > 0

        # reduced residuals map
        red_res = self.solver_lens_syn.normalized_residuals(S=S, HG=HG)
        assert red_res.shape == self.image_sim.shape

        # reduced chi2
        red_chi2 = self.solver_lens_syn.reduced_chi2(S=S, HG=HG)
        assert red_chi2 > 0
        assert self.solver_lens_syn.best_fit_reduced_chi2 == self.solver_lens_syn.reduced_chi2(S=S, HG=HG)

        # synthesis and analysis models
        alpha_S = self.source_model_class.func_list[0].decomposition_2d(S, self.n_scales_source)
        alpha_HG = self.lens_light_model_class.func_list[0].decomposition_2d(HG, self.n_scales_lens)
        ma = self.solver_lens_syn.model_analysis(S, HG)
        ms = self.solver_lens_syn.model_synthesis(alpha_S, alpha_HG)
        npt.assert_almost_equal(ma, ms, decimal=4)

    def test_solve_source_ps_analysis(self):
        # source solver
        image_model, param, logL_penalty = \
            self.solver_source_ps_ana.solve(self.kwargs_lens, self.kwargs_source, 
                                            kwargs_ps=self.kwargs_ps,
                                            kwargs_special=self.kwargs_special,
                                            init_ps_model=self.ps_sim)
        assert image_model.shape == self.image_sim.shape
        len_param_theory = self.num_pix_source**2*self.n_scales_source + self.n_point_sources
        assert len(param) == len_param_theory

        # get the track
        track = self.solver_source_ps_ana.track
        len_track_theory = self.num_iter_source*self.num_iter_global*self.num_iter_weights
        assert len(track['loss'][0, :]) == len_track_theory

        # access models
        image_model = self.solver_source_ps_ana.image_model()
        assert image_model.shape == self.image_sim.shape
        source_light = self.solver_source_ps_ana.source_model
        assert source_light.shape == (self.num_pix_source, self.num_pix_source)
        ps_light = self.solver_source_ps_ana.point_source_model
        assert ps_light.shape == (self.num_pix, self.num_pix)

        S, P = source_light, ps_light

        # reduced residuals map
        red_res = self.solver_source_ps_ana.normalized_residuals(S=S, P=P)
        assert red_res.shape == self.image_sim.shape

        # L2-norm of difference of two arrays
        S_ = np.random.rand(self.num_pix_source, self.num_pix_source)
        assert self.solver_source_ps_ana.norm_diff(S, S_) > 0

        # reduced chi2
        red_chi2 = self.solver_source_ps_ana.reduced_chi2(S=S, P=P)
        assert red_chi2 > 0
        assert self.solver_source_ps_ana.best_fit_reduced_chi2 == self.solver_source_ps_ana.reduced_chi2(S=S, P=P)

        # test plot results
        fig = self.solver_source_ps_ana.plot_results()
        plt.close()

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
        self.lens_model_class = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]
        self.source_model_class = LightModel(['SLIT_STARLETS'])
        self.lens_light_model_class = LightModel(['SLIT_STARLETS'])
        self.kwargs_source = [{'n_scales': 4}]

        self.kwargs_lens_light = [{'n_scales': 4}]
        psf = PSF(psf_type='NONE')
        self.numerics = NumericsSubFrame(pixel_grid=self.data, psf=psf)
        self.source_numerics = NumericsSubFrame(pixel_grid=self.data, psf=psf, supersampling_factor=self.subgrid_res_source)
        self.solver_source_lens = SparseSolverSourceLens(self.data, self.lens_model_class, self.numerics, self.source_numerics,
                                                         self.source_model_class, self.lens_light_model_class,
                                                         num_iter_source=1, num_iter_lens=1, num_iter_weights=1)
        
    def test_raise(self):
        with self.assertRaises(ValueError):
            _ = self.solver_source_lens.solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light)
            # no deconvolution of lens light is performed, so raises an error
            image_model_deconvolved = self.solver_source_lens.image_model(unconvolved=True)


if __name__ == '__main__':
    pytest.main()
