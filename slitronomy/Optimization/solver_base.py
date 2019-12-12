__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
from scipy import signal

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Plots.solver_plotter import SolverPlotter
from slitronomy.Util import util


class SparseSolverBase(ModelOperators):

    """Base class that generally defines a sparse solver"""

    def __init__(self, data_class, lens_model_class, source_model_class, lens_light_model_class=None,
                 psf_class=None, convolution_class=None, likelihood_mask=None, 
                 subgrid_res_source=1, minimal_source_plane=True, min_num_pix_source=10,
                 sparsity_prior_norm=1, force_positivity=True, formulation='analysis', 
                 verbose=False, show_steps=False):
        # consider only the first light profiles
        source_light_class = source_model_class.func_list[0]
        if lens_light_model_class is not None:
            lens_light_class = lens_light_model_class.func_list[0]
        else:
            lens_light_class = None

        (num_pix_x, num_pix_y) = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._delta_pix = data_class.pixel_width

        # self._noise_map = data_class.noise_map
        self._sigma_bkg = data_class.background_rms

        if psf_class is not None:
            self._psf_kernel = psf_class.kernel_point_source
        else:
            self._psf_kernel = None

        lensing_operator_class = LensingOperator(data_class, lens_model_class, subgrid_res_source=subgrid_res_source, 
                                                 likelihood_mask=likelihood_mask, minimal_source_plane=minimal_source_plane,
                                                 min_num_pix_source=min_num_pix_source, matrix_prod=True)

        super(SparseSolverBase, self).__init__(data_class, lensing_operator_class, 
                                               source_light_class, lens_light_class=lens_light_class, 
                                               convolution_class=convolution_class, likelihood_mask=likelihood_mask)

        self._formulation = formulation

        if sparsity_prior_norm not in [0, 1]:
            raise ValueError("Sparsity prior norm can only be 0 or 1 (l0-norm or l1-norm)")
        self._sparsity_prior_norm = sparsity_prior_norm
        self._force_positivity = force_positivity

        self._verbose = verbose
        self._show_steps = show_steps

        self._plotter = SolverPlotter(self)

    def solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None):
        """
        main method to call from outside

        any class that inherits SparseSolverSource should have this method updated accordingly, with same output.
        """
        # update image <-> source plane mapping from lens model parameters
        self.lensingOperator.update_mapping(kwargs_lens)
        # get number of decomposition scales
        self._n_scales_source = kwargs_source[0]['n_scales']
        if kwargs_lens_light is not None:
            self._n_scales_lens = kwargs_lens_light[0]['n_scales']
        else:
            self._n_scales_lens = None
        self.set_wavelet_scales(self._n_scales_source, self._n_scales_lens)
        # call solver
        image_model, source_light, lens_light, coeffs = self._solve()
        return image_model, source_light, lens_light, coeffs

    @property
    def plotter(self):
        return self._plotter

    @property
    def source_model(self):
        if not hasattr(self, '_source_model'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        return self._source_model

    def image_model(self, unconvolved=False):
        if not hasattr(self, '_source_model'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        image_model = self.F(self._source_model)
        if unconvolved:
            return image_model
        return self.H(image_model)

    @property
    def solve_track(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the track")
        return self._solve_track

    @property
    def best_fit_reduced_chi2(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the track")
        return self._solve_track['red_chi2'][-1]

    def generate_initial_source(self, guess_type='bkg_noise'):
        num_pix = self.lensingOperator.sourcePlane.num_pix
        n_scales = self._n_scales_source
        transform = self.Phi_T_s
        inverse_transform = self.Phi_s
        sigma_bkg_synthesis = self.noise_levels_source_plane
        return util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                           formulation=self._formulation, guess_type=guess_type,
                           sigma_bkg=self._sigma_bkg, sigma_bkg_synthesis=sigma_bkg_synthesis)

    def generate_initial_lens_light(self, guess_type='bkg_noise'):
        num_pix = self.lensingOperator.imagePlane.num_pix
        n_scales = self._n_scales_lens
        transform = self.Phi_T_l
        inverse_transform = self.Phi_l
        sigma_bkg_synthesis = None # TODO 
        return util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                                           formulation=self._formulation, guess_type=guess_type,
                                           sigma_bkg=self._sigma_bkg, sigma_bkg_synthesis=sigma_bkg_synthesis)

    def apply_image_plane_mask(self, image_2d):
        return self.M(image_2d)

    def apply_source_plane_mask(self, source_2d):
        return self.M_s(source_2d)

    def project_original_grid_source(self, source_2d):
        return self.lensingOperator.sourcePlane.project_on_original_grid(source_2d)

    def psf_convolution(self, array_2d):
        return self.H(array_2d)

    @property
    def image_data(self):
        return self._image_data

    @property
    def lensingOperator(self):
        return self._lensing_op

    @property
    def num_data_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return int(np.sum(self._mask))

    def loss(self, S, G=None):
        """ returns f = || Y - HFS - HG ||^2_2 """
        model = self._model_analysis(S, G=G)
        error = self.Y - model
        norm_error = np.linalg.norm(error, ord=2)
        return norm_error**2

    def reduced_residuals(self, S, G=None):
        """ returns || Y - HFS - HG ||^2_2 / sigma^2 """
        model = self._model_analysis(S, G=G)
        error = self.Y - model
        return (error / self._sigma_bkg) * self._mask

    def reduced_chi2(self, S):
        chi2 = self.reduced_residuals(S)**2
        return np.sum(chi2) / self.num_data_evaluate

    def norm_diff(self, S1, S2):
        """ returns || S1 - S2 ||^2_2 """
        diff = S1 - S2
        return np.linalg.norm(diff, ord=2)**2

    def gradient_loss(self, array):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis(array)
        elif self._formulation == 'synthesis':
            return self._gradient_loss_synthesis(array)

    def proximal_sparsity(self, array, step, weights):
        if self._formulation == 'analysis':
            return self._proximal_sparsity_analysis(array, step, weights)
        elif self._formulation == 'synthesis':
            return self._proximal_sparsity_synthesis(array, step, weights)

    @property
    def algorithm(self):
        if self._formulation == 'analysis':
            return 'FB'
        elif self._formulation == 'synthesis':
            return 'FISTA'

    @property
    def noise_levels_source_plane(self):
        if not hasattr(self, '_noise_levels_src'):
            self._noise_levels_src = self._compute_noise_levels_src(boost_where_zero=10)
        return self._noise_levels_src

    def _compute_noise_levels_src(self, boost_where_zero=10):
        n_img = self.lensingOperator.imagePlane.num_pix

        # PSF noise map
        if self._psf_kernel is None:
            HT = util.dirac_impulse(n_img)
        else:
            HT = self._psf_kernel.T
        HT_power = np.sqrt(np.sum(HT**2))
        HT_noise = self._sigma_bkg * HT_power * np.ones((n_img, n_img))
        FT_HT_noise = self.F_T(HT_noise)
        FT_HT_noise[FT_HT_noise == 0.] = np.mean(FT_HT_noise) * boost_where_zero

        # computes noise levels in in source plane in starlet space
        dirac = util.dirac_impulse(n_img)
        dirac_mapped = self.F_T(dirac)

        # model transform of the impulse
        dirac_coeffs = self.Phi_T_s(dirac_mapped)

        noise_levels = np.zeros(dirac_coeffs.shape)
        for scale_idx in range(noise_levels.shape[0]):
            dirac_scale = dirac_coeffs[scale_idx, :, :]
            levels = signal.fftconvolve(FT_HT_noise**2, dirac_scale**2, mode='same')
            levels[levels == 0.] = 0.
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        return noise_levels

    @property
    def spectral_norm_source(self):
        if not hasattr(self, '_spectral_norm_source'):
            def _operator(x):
                x = self.H_T(x)
                x = self.F_T(x)
                x = self.Phi_T_s(x)
                return x
            def _inverse_operator(x):
                x = self.Phi_s(x)
                x = self.F(x)
                x = self.H(x)
                return x
            self._spectral_norm_source = util.spectral_norm(self._num_pix, _operator, _inverse_operator,
                                                            num_iter=20, tol=1e-10)
        return self._spectral_norm_source

    @property
    def spectral_norm_lens(self):
        if not hasattr(self, '_spectral_norm_lens'):
            def _operator(x):
                x = self.H_T(x)
                x = self.Phi_T_l(x)
                return x
            def _inverse_operator(x):
                x = self.Phi_l(x)
                x = self.H(x)
                return x
            self._spectral_norm_lens = util.spectral_norm(self._num_pix, _operator, _inverse_operator,
                                                            num_iter=20, tol=1e-10)
        return self._spectral_norm_lens
