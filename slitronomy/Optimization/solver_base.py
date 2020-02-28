__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
from scipy import signal

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator, LensingOperatorInterpol
from slitronomy.Util.solver_plotter import SolverPlotter
from slitronomy.Util.solver_tracker import SolverTracker
from slitronomy.Util import util


class SparseSolverBase(ModelOperators):
    """
    Base class that generally defines a sparse solver
    """

    def __init__(self, data_class, lens_model_class, source_model_class, lens_light_model_class=None,
                 psf_class=None, convolution_class=None, likelihood_mask=None, lensing_operator='simple',
                 subgrid_res_source=1, minimal_source_plane=True, fix_minimal_source_plane=True, min_num_pix_source=10,
                 sparsity_prior_norm=1, force_positivity=True, formulation='analysis', initial_guess_type='noise_map',
                 verbose=False, show_steps=False):
        # consider only the first light profiles in model lists
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

        # background noise
        self._background_rms = data_class.background_rms

        # noise full covariance \simeq sqrt(poisson_rms^2 + gaussian_rms^2)
        self._noise_map = np.sqrt(data_class.C_D)

        # controls generation of initial guess
        self._initial_guess_type = initial_guess_type

        if psf_class is not None:
            self._psf_kernel = psf_class.kernel_point_source
        else:
            self._psf_kernel = None

        if lensing_operator == 'simple':
            lensing_operator_class = LensingOperator(data_class, lens_model_class, subgrid_res_source=subgrid_res_source, 
                                                     likelihood_mask=likelihood_mask, minimal_source_plane=minimal_source_plane,
                                                     fix_minimal_source_plane=fix_minimal_source_plane, min_num_pix_source=min_num_pix_source, 
                                                     matrix_prod=True)
        if lensing_operator == 'interpol':
            lensing_operator_class = LensingOperatorInterpol(data_class, lens_model_class, subgrid_res_source=subgrid_res_source, 
                                                     likelihood_mask=likelihood_mask, minimal_source_plane=minimal_source_plane,
                                                     fix_minimal_source_plane=fix_minimal_source_plane, min_num_pix_source=min_num_pix_source)

        super(SparseSolverBase, self).__init__(data_class, lensing_operator_class, 
                                               source_light_class, lens_light_class=lens_light_class, 
                                               convolution_class=convolution_class, likelihood_mask=likelihood_mask)

        # fill masked pixel with background noise
        self.fill_masked_data(self._background_rms)

        self._formulation = formulation
        if sparsity_prior_norm not in [0, 1]:
            raise ValueError("Sparsity prior norm can only be 0 or 1 (l0-norm or l1-norm)")
        self._sparsity_prior_norm = sparsity_prior_norm
        self._force_positivity = force_positivity

        self._verbose = verbose
        self._show_steps = show_steps

        self._tracker = SolverTracker(self, verbose=verbose)
        self._plotter = SolverPlotter(self, show_now=True)

    def solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_special=None):
        """
        main method to call from outside the class, calling self._solve()

        any class that inherits SparseSolverSource should have the self._solve() method implemented, with correct output.
        """
        # update image <-> source plane mapping from lens model parameters
        size_image, pixel_scale_image, size_source, pixel_scale_source \
            = self.lensingOperator.update_mapping(kwargs_lens, kwargs_special=kwargs_special)
        # get number of decomposition scales
        n_scales_source = kwargs_source[0]['n_scales']
        if kwargs_lens_light is not None and len(kwargs_lens_light) > 0:
            n_scales_lens_light = kwargs_lens_light[0]['n_scales']
        else:
            n_scales_lens_light = None
        # save number of scales
        self.set_wavelet_scales(n_scales_source, n_scales_lens_light)
        # call solver
        image_model, source_light, lens_light, coeffs_source, coeffs_lens_light = self._solve()
        if lens_light is None: coeffs_lens_light = []
        # concatenate coefficients and fixed parameters
        coeffs = np.concatenate([coeffs_source, coeffs_lens_light])
        scales = [size_source, pixel_scale_source, size_image, pixel_scale_image]
        return image_model, source_light, lens_light, coeffs, scales

    def _solve(self):
        raise ValueError("This method must be implemented in class that inherits SparseSolverBase")

    @property
    def track(self):
        return self._tracker.track

    @property
    def plotter(self):
        return self._plotter

    def plot_results(self, model_log_scale=False, res_vmin=None, res_vmax=None):
        return self.plotter.plot_results(model_log_scale=model_log_scale, res_vmin=res_vmin, res_vmax=res_vmax)

    @property
    def image_data(self):
        return self._image_data

    @property
    def lensingOperator(self):
        return self._lensing_op

    @property
    def source_model(self):
        if not hasattr(self, '_source_model'):
            return None
        return self._source_model

    @property
    def lens_light_model(self):
        if not hasattr(self, '_lens_light_model'):
            return None
        return self._lens_light_model

    def image_model(self, unconvolved=False):
        return self._image_model(unconvolved=unconvolved)

    def _image_model(self, unconvolved=False):
        if not hasattr(self, '_source_model'):
            if unconvolved:
                raise ValueError("Cannot provide *unconvolved* lens light")
            return self._lens_light_model
        elif not hasattr(self, '_lens_light_model'):
            if unconvolved:
                return self.F(self._source_model)
            else:
                return self.H(self.F(self._source_model))
        else:
            return self.H(self.F(self._source_model)) + self._lens_light_model

    @property
    def reduced_residuals_model(self):
        """ returns || Y - HFS - HG ||^2_2 / sigma^2 """
        if hasattr(self, '_lens_light_model'):
            return self.reduced_residuals(self.source_model, HG=self.lens_light_model)
        else:
            return self.reduced_residuals(self.source_model)

    @property
    def solve_track(self):
        if not hasattr(self, '_solve_track'):
            raise ValueError("You must run the optimization before accessing the solver results")
        return self._solve_track

    def generate_initial_source(self):
        num_pix = self.lensingOperator.sourcePlane.num_pix
        n_scales = self._n_scales_source
        transform = self.Phi_T_s
        inverse_transform = self.Phi_s
        noise_map_synthesis = self.noise_levels_source_plane
        return util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                           formulation=self._formulation, guess_type=self._initial_guess_type,
                           background_rms=self._background_rms, noise_map=self._noise_map, 
                           noise_map_synthesis=noise_map_synthesis)

    def generate_initial_lens_light(self):
        num_pix = self.lensingOperator.imagePlane.num_pix
        n_scales = self._n_scales_lens_light
        transform = self.Phi_T_l
        inverse_transform = self.Phi_l
        noise_map_synthesis = self.noise_levels_image_plane
        return util.generate_initial_guess(num_pix, n_scales, transform, inverse_transform, 
                                           formulation=self._formulation, guess_type=self._initial_guess_type,
                                           background_rms=self._background_rms, noise_map=self._noise_map, 
                                           noise_map_synthesis=noise_map_synthesis)

    def apply_image_plane_mask(self, image_2d):
        return self.M(image_2d)

    def apply_source_plane_mask(self, source_2d):
        return self.M_s(source_2d)

    def project_original_grid_source(self, source_2d):
        return self.lensingOperator.sourcePlane.project_on_original_grid(source_2d)

    def psf_convolution(self, array_2d):
        return self.H(array_2d)

    @property
    def num_data_points(self):
        """
        number of effective data points (= number of unmasked pixels)
        """
        return int(np.sum(self._mask))

    @property
    def best_fit_reduced_chi2(self):
        return self.reduced_chi2(S=self.source_model, HG=self.lens_light_model)

    def loss(self, S=None, HG=None):
        """ returns f = || Y' - HFS - HG ||^2_2, where Y' can be Y - HG or Y - HFS """
        model = self.model_analysis(S=S, HG=HG)
        error = self.Y_eff - model
        norm_error = np.linalg.norm(error.flatten(), ord=2)  # flatten to ensure L2-norm
        return 0.5 * norm_error**2

    def reduced_residuals(self, S=None, HG=None):
        """ returns || Y' - HFS - HG ||^2_2 / sigma^2, where Y' can be Y - HG or Y - HFS """
        model = self.model_analysis(S=S, HG=HG)
        error = self.Y_eff - model
        return self.M(error / self._noise_map)

    def reduced_chi2(self, S=None, HG=None):
        chi2 = self.reduced_residuals(S=S, HG=HG)**2
        return np.sum(chi2) / self.num_data_points

    def norm_diff(self, S1, S2):
        """ returns || S1 - S2 ||^2_2 """
        diff = S1 - S2
        return np.linalg.norm(diff.flatten(), ord=2)**2  # flatten to ensure L2-norm

    def model_analysis(self, S=None, HG=None):
        if S is not None and HG is None:
            return self.H(self.F(S))
        elif S is None and HG is not None:
            return HG
        else:
            return self.H(self.F(S)) + HG

    def model_synthesis(self, alpha_S=None, alpha_HG=None):
        if alpha_S is not None and alpha_HG is None:
            return self.H(self.F(self.Phi_s(alpha_S)))
        elif alpha_S is None and alpha_HG is not None:
            return self.Phi_l(alpha_HG)
        else:
            return self.H(self.F(self.Phi_s(alpha_S))) + self.Phi_l(alpha_HG)

    def gradient_loss_source(self, array_S):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis_source(S=array_S)
        elif self._formulation == 'synthesis':
            return self._gradient_loss_synthesis_source(alpha_S=array_S)

    def gradient_loss_lens(self, array_HG):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis_lens(HG=array_HG)
        elif self._formulation == 'synthesis':
            return self._gradient_loss_synthesis_lens(alpha_HG=array_HG)

    def proximal_sparsity_source(self, array, weights):
        if self._formulation == 'analysis':
            return self._proximal_sparsity_analysis_source(array, weights)
        elif self._formulation == 'synthesis':
            return self._proximal_sparsity_synthesis_source(array, weights)

    def proximal_sparsity_lens(self, array, weights):
        if self._formulation == 'analysis':
            return self._proximal_sparsity_analysis_lens(array, weights)
        elif self._formulation == 'synthesis':
            return self._proximal_sparsity_synthesis_lens(array, weights)

    def subtract_source_from_data(self, S):
        """Update "effective" data by subtracting the input source light estimation"""
        source_model = self.model_analysis(S=S, HG=None)
        self.subtract_from_data(source_model)

    def subtract_lens_from_data(self, HG):
        """Update "effective" data by subtracting the input (convolved) lens light estimation"""
        lens_model = self.model_analysis(S=None, HG=HG)
        self.subtract_from_data(lens_model)

    @property
    def algorithm(self):
        if self._formulation == 'analysis':
            return 'FB'
        elif self._formulation == 'synthesis':
            return 'FISTA'

    @property
    def noise_levels_image_plane(self):
        if not hasattr(self, '_noise_levels_img'):
            self._noise_levels_img = self._compute_noise_levels_img()
        return self._noise_levels_img

    def _compute_noise_levels_img(self):
        # TODO : check this method
        n_img = self.lensingOperator.imagePlane.num_pix

        # computes noise levels in in source plane in starlet space
        dirac = util.dirac_impulse(n_img)

        # model transform of the impulse
        dirac_coeffs = self.Phi_T_l(dirac)

        #noise_levels = self._background_rms * np.ones_like(dirac_coeffs)
        noise_levels = self._noise_map

        for scale_idx in range(noise_levels.shape[0]):
            scale_power = np.sum(dirac_coeffs[scale_idx, :, :]**2)
            noise_levels[scale_idx, :, :] *= np.sqrt(scale_power)
        return noise_levels

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
        # HT_noise = self._background_rms * HT_power * np.ones((n_img, n_img))
        HT_noise = self._noise_map * HT_power
        FT_HT_noise = self.F_T(HT_noise)
        FT_HT_noise[FT_HT_noise == 0.] = np.mean(FT_HT_noise) * boost_where_zero

        # computes noise levels in in source plane in starlet space
        dirac = util.dirac_impulse(n_img)
        dirac_mapped = self.F_T(dirac)

        # model transform of the impulse
        dirac_coeffs = self.Phi_T_s(dirac_mapped)

        noise_levels = np.zeros_like(dirac_coeffs)
        for scale_idx in range(noise_levels.shape[0]):
            dirac_scale = dirac_coeffs[scale_idx, :, :]
            levels = signal.fftconvolve(FT_HT_noise**2, dirac_scale**2, mode='same')
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        return noise_levels

    def _update_weights(self, alpha_S, alpha_HG=None):
        lambda_S = self.noise_levels_source_plane
        lambda_S[0, :, :]  *= self._k_max_high_freq
        lambda_S[1:, :, :] *= self._k_max
        weights_S  = 1. / ( 1 + np.exp(-10 * (lambda_S - alpha_S)) )  # Eq. (11) of Joseph et al. 2018
        if alpha_HG is not None:
            lambda_HG = self.noise_levels_image_plane
            lambda_HG[0, :, :]  *= self._k_max_high_freq
            lambda_HG[1:, :, :] *= self._k_max
            weights_HG = 1. / ( 1 + np.exp(-10 * (lambda_HG - alpha_HG)) )  # Eq. (11) of Joseph et al. 2018
        else:
            weights_HG = None
        return weights_S, weights_HG
