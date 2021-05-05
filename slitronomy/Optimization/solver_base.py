__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Optimization.noise_levels import NoiseLevels
from slitronomy.Optimization import proximals
from slitronomy.Util.solver_plotter import SolverPlotter
from slitronomy.Util.solver_tracker import SolverTracker
from slitronomy.Util import util
from slitronomy.Util import mask_util


class SparseSolverBase(ModelOperators):
    """
    Base class that generally defines a sparse solver
    """

    #TODO: raises an error when number of decomposition scales is not consistent with image size
    # (also when reducing source plane size, re-check consistency)

    #TODO: create classes for lens, source and point source models.
    # E.g. the method project_on_original_grid_source should be attached to some new "SourceModel" class, not to the solver.
    # or _get_ps_coordinates should be in a point source class.

    def __init__(self, data_class, image_numerics_class, source_numerics_class,
                 lens_model_class=None, lens_light_mask=None, source_interpolation='bilinear', 
                 minimal_source_plane=False, use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                 min_threshold=3, threshold_increment_high_freq=1, threshold_decrease_type='exponential',
                 fixed_spectral_norm_source=0.98, include_regridding_error=False, include_point_source_error=False,
                 sparsity_prior_norm=1, force_positivity=True, formulation='analysis',
                 external_likelihood_penalty=False, random_seed=None,
                 verbose=False, show_steps=False, thread_count=1):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param lens_model_class: lenstronomy.lens_model.LensModel instance describing the lens mass model.
        :param image_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for image plane.
        :param source_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for source plane.
        :param lens_light_mask: boolean mask with False/0 to exclude pixels that are assumed to contain only lens light flux.
        Defaults to None.
        :param source_interpolation: type of interpolation of source pixels on the source plane grid.
        It can be 'nearest' for nearest-neighbor or 'bilinear' for bilinear interpolation. Defaults to 'bilinear'.
        :param minimal_source_plane: if True, reduce the source plane grid size to the minimum set by min_num_pix_source.
         Defaults to False.
        :param use_mask_for_minimal_source_plane: if True, use the likelihood_mask to compute minimal source plane.
         Defaults to True.
        :param min_num_pix_source: minimal number of pixels on a side of the square source grid.
        Only used when minimal_source_plane is True. Defaults to 20.
        :param min_threshold: in unit of the noise (sigma), minimum threshold for wavelets denoising.
        Typically between 3 (more conservative thresholding) and 5 (more aggressive thresholding). Defaults to 3.
        :param threshold_increment_high_freq: additive number to the threshold level (in unit of the noise) for the highest frequencies on wavelets space.
        Defaults to 1.
        :param threshold_decrease_type: strategy for decreasing the threshold level at each iteration. Can be 'none' (no decrease, directly sets to min_threshold), 'linear' or 'exponential'.
        Defaults to None, which is 'exponential' for the source-only solver, 'linear' for the source-lens solver.
        :param fixed_spectral_norm_source: if None, update the spectral norm for the source operator, for optimal gradient descent step size.
        Defaults to 0.98, which is a conservative value typical of most lens models.
        :param sparsity_prior_norm: prior l-norm (0 or 1). If 1, l1-norm and soft-thresholding are applied.
        If 0, it is l0-norm and hard-thresholding. Defaults to 1.
        :param force_positivity: if True, apply positivity constraint to the source flux.
        Defaults to True.
        :param formulation: type of formalism for the minimization problem. 'analysis' solves the problem in direct space.
        'synthesis' solves the peoblem in wavelets space. Defaults to 'analysis'.
        :param external_likelihood_penalty: if True, the solve() method returns a non-zero penalty, 
        e.g. for penalize more a given lens model during lens model optimization. Defaults to False.
        :param random_seed: seed for random number generator, used to initialise the algorithm. None for no seed.
        Defaults to None.
        :param verbose: if True, prints statements during optimization.
        Defaults to False.
        :param show_steps: if True, displays plot of the reconstructed light profiles during optimization.
        Defaults to False.
        :param thread_count: number of threads (multithreading) to speedup wavelets computations (only works if pySAP is properly installed).
        Defaults to 1.
        """
        num_pix_x, num_pix_y = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        image_grid_class = image_numerics_class.grid_class
        source_grid_class = source_numerics_class.grid_class
        if lens_model_class is None or len(lens_model_class.lens_model_list) == 0:
            lensing_operator_class = None
        else:
            lensing_operator_class = LensingOperator(lens_model_class, image_grid_class, source_grid_class, num_pix_x,
                                                     lens_light_mask=lens_light_mask,
                                                     minimal_source_plane=minimal_source_plane, min_num_pix_source=min_num_pix_source,
                                                     use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
                                                     source_interpolation=source_interpolation, verbose=verbose)

        super(SparseSolverBase, self).__init__(data_class, lensing_operator_class, image_numerics_class,
                                               fixed_spectral_norm_source=fixed_spectral_norm_source,
                                               thread_count=thread_count, random_seed=random_seed)
        
        # engine that computes noise levels in image / source plane, in wavelets space
        self.noise = NoiseLevels(data_class, subgrid_res_source=source_grid_class.supersampling_factor,
                                 include_regridding_error=include_regridding_error,
                                 include_point_source_error=include_point_source_error)

        # threshold level k_min (in units of the noise)
        self._k_min = min_threshold
        if threshold_increment_high_freq < 0:
            raise ValueError("threshold_increment_high_freq cannot be negative")
        else:
            self._increm_high_freq = threshold_increment_high_freq

        # strategy to decrease threshold up to the max threshold above
        if threshold_decrease_type not in ['none', 'lin', 'linear', 'exp', 'exponential']:
            raise ValueError("threshold_decrease_type must be in ['none', 'lin', 'linear', 'exp', 'exponential']")
        self._threshold_decrease_type = threshold_decrease_type

        if sparsity_prior_norm not in [0, 1]:
            raise ValueError("Sparsity prior norm can only be 0 or 1 (l0-norm or l1-norm)")
        self._sparsity_prior_norm = sparsity_prior_norm
        self._formulation = formulation
        self._force_positivity = force_positivity

        self._external_likelihood_penalty = external_likelihood_penalty

        self._verbose = verbose
        self._show_steps = show_steps

        self._tracker = SolverTracker(self, verbose=verbose)
        self._plotter = SolverPlotter(self, show_now=True)

    def set_likelihood_mask(self, mask=None):
        self._set_likelihood_mask(mask)

    def solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None,
              init_lens_light_model=None, init_ps_model=None, init_ps_amp=None, ps_error_map=None):
        """
        main method to call from outside the class, calling self._solve()

        any class that inherits SparseSolverSource should have self._ready() and self._solve() methods implemented, 
        with correct output.
        """
        if not self._ready(): return

        # update lensing operator and noise levels
        self.prepare_solver(kwargs_lens, kwargs_source, kwargs_lens_light=kwargs_lens_light, 
                            kwargs_ps=kwargs_ps, kwargs_special=kwargs_special, 
                            init_lens_light_model=init_lens_light_model, 
                            init_ps_model=init_ps_model, init_ps_amp=init_ps_amp, 
                            ps_error_map=ps_error_map)

        # call solver
        image_model, coeffs_source, coeffs_lens_light, amps_ps \
            = self._solve(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, kwargs_special=kwargs_special)

        # concatenate optimized parameters (wavelets coefficients, point source amplitudes)
        all_param = np.concatenate([coeffs_source, coeffs_lens_light, amps_ps])

        # WIP: allows for an extra term to be returned by the solver
        if self._external_likelihood_penalty:
            if self.no_lens_light:
                logL_penalty = self.regularization(S=self.source_model)
            else:
                logL_penalty = self.regularization(S=self.source_model, HG=self.lens_light_model)
        else:
            logL_penalty = 0

        return image_model, all_param, logL_penalty

    def _solve(self, kwargs_lens=None, kwargs_ps=None, kwargs_special=None):
        raise NotImplementedError("This method must be implemented in class that inherits SparseSolverBase")

    def _ready(self):
        raise NotImplementedError("This method must be implemented in class that inherits SparseSolverBase")

    @property
    def track(self):
        return self._tracker.track

    @property
    def component_names(self):
        return 'S', 'HG', 'P'

    @property
    def prior_l_norm(self):
        return self._sparsity_prior_norm

    def plot_results(self, **kwargs):
        return self._plotter.plot_results(**kwargs)

    def plot_source_residuals_comparison(self, *args, **kwargs):
        return self._plotter.plot_source_residuals_comparison(*args, **kwargs)

    @property
    def source_model(self):
        if self.no_source_light:
            return np.zeros_like(self.image_data)
        if not hasattr(self, '_source_model'):
            raise RuntimeError("No source model has been optimised")
        return self._source_model

    @property
    def lens_light_model(self):
        if self.no_lens_light:
            return np.zeros_like(self.image_data)
        if not hasattr(self, '_lens_light_model'):
            raise RuntimeError("No lens light model has been optimised")
        return self._lens_light_model

    @property
    def point_source_model(self):
        if self.no_point_source:
            return np.zeros_like(self.image_data)
        if not hasattr(self, '_ps_model'):
            raise RuntimeError("No point source model has been optimised")
        return self._ps_model
        
    def image_model(self, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True):
        if unconvolved and self.no_lens_light is False:
            raise ValueError("Deconvolution is only supported for source model")
        if unconvolved and self.no_source_light is False:
            S = self.source_model
            model = self.F(S)
        else:
            model = np.zeros_like(self.image_data)
            if source_add and self.no_source_light is False:
                S = self.source_model
                model += self.H(self.R(self.F(S)))
            if lens_light_add and self.no_lens_light is False:
                HG = self.lens_light_model
                model += HG
            if point_source_add and self.no_point_source is False:
                P = self.point_source_model
                model += P
        return model

    @property
    def normalized_residuals_model(self):
        """ returns ( HFS + HG + P - Y ) / sigma """
        return self.normalized_residuals(S=self.source_model, 
                                         HG=self.lens_light_model, 
                                         P=self.point_source_model)

    @property
    def residuals_model(self):
        """ returns ( HFS + HG + P - Y ) """
        return self.residuals(S=self.source_model, 
                              HG=self.lens_light_model, 
                              P=self.point_source_model)

    def generate_initial_source(self):
        num_pix = self.num_pix_source
        transform = self.Phi_T_s
        return util.generate_initial_guess_simple(num_pix, transform, self.noise.background_rms, seed=self.random_seed)

    def generate_initial_lens_light(self):
        num_pix = self.num_pix_image
        transform = self.Phi_T_l
        return util.generate_initial_guess_simple(num_pix, transform, self.noise.background_rms, seed=self.random_seed)

    def apply_image_plane_mask(self, image_2d):
        return self.M(image_2d)

    def apply_source_plane_mask(self, source_2d):
        return self.M_s(source_2d)

    def project_on_original_grid_source(self, source):
        return self.lensingOperator.sourcePlane.project_on_original_grid(source)

    def psf_convolution(self, array_2d):
        return self.H(array_2d)

    @property
    def num_data_points(self):
        """
        number of effective data points (= number of unmasked pixels)
        """
        mask = self.likelihood_mask
        if mask is None:
            return np.prod(self.image_data.shape)
        return int(np.sum(self._mask))

    @property
    def best_fit_reduced_chi2(self):
        return self.reduced_chi2(S=self.source_model,
                                 HG=self.lens_light_model, 
                                 P=self.point_source_model)

    @property
    def best_fit_mean_squared_error(self):
        return self.mean_squared_error(S=self.source_model, 
                                       HG=self.lens_light_model, 
                                       P=self.point_source_model)

    def loss(self, S=None, HG=None, P=None):
        """ returns f = || Y - HFS - HG - P ||^2_2 """
        model = self.model_analysis(S=S, HG=HG, P=P)
        error = self.effective_image_data - model
        norm_error = np.linalg.norm(error.flatten(), ord=2)  # flatten to ensure L2-norm
        return 0.5 * norm_error**2

    def regularization(self, S=None, HG=None, P=None):
        """ returns p = lambda * || W_S ø alpha_S ||_0,1 + lambda * || W_HG ø alpha_HG ||_0,1 """
        if S is not None:
            reg_S = self._regularization(S, self.Phi_T_s, self.M_s, self.noise.levels_source)
        else:
            reg_S = 0
        if HG is not None:
            reg_HG = self._regularization(HG, self.Phi_T_l, self.M, self.noise.levels_image)
        else:
            reg_HG = 0
        return reg_S + reg_HG

    def _regularization(self, image, transform, mask_func, noise_levels):
        lambda_ = np.copy(noise_levels)
        lambda_[0, :, :]  *= (self._k_min + self._increm_high_freq)
        lambda_[1:, :, :] *= self._k_min
        alpha_image = mask_func(transform(image))
        norm_alpha = np.linalg.norm((lambda_ * alpha_image).flatten(), ord=self._sparsity_prior_norm)
        return norm_alpha

    def residuals(self, S=None, HG=None, P=None):
        model = self.model_analysis(S=S, HG=HG, P=P)
        return model - self.effective_image_data

    def normalized_residuals(self, S=None, HG=None, P=None):
        """ returns ( HFS + HG + P - Y ) / sigma """
        residuals = self.residuals(S=S, HG=HG, P=P)
        sigma = self.noise.effective_noise_map
        return self.M(residuals / sigma)

    def reduced_chi2(self, S=None, HG=None, P=None):
        norm_res = self.normalized_residuals(S=S, HG=HG, P=P)
        chi2 = np.sum(norm_res**2)
        return chi2 / self.num_data_points

    def mean_squared_error(self, S=None, HG=None, P=None):
        res = self.residuals(S=S, HG=HG, P=P)
        return np.sum(res**2) / self.num_data_points

    @staticmethod
    def norm_diff(S1, S2):
        """ returns || S1 - S2 ||_2 """
        diff = S1 - S2
        return np.linalg.norm(diff.flatten(), ord=2)  # flatten to ensure L2-norm

    def model_analysis(self, S=None, HG=None, P=None):
        model = 0
        if S is not None:
            model += self.H(self.R(self.F(S)))
        if HG is not None:
            model += self.R(HG)
        if P is not None:
            model += P
        return model

    def model_synthesis(self, alpha_S=None, alpha_HG=None, P=None):
        model = 0
        if alpha_S is not None:
            model = self.H(self.R(self.F(self.Phi_s(alpha_S))))
        if alpha_HG is not None:
            model += self.R(self.Phi_l(alpha_HG))
        if P is not None:
            model += P
        return model

    def gradient_loss_source(self, array_S):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis_source(S=array_S)
        elif self._formulation == 'synthesis':
            return self._gradient_loss_synthesis_source(alpha_S=array_S)

    def _gradient_loss_analysis_source(self, S):
        """
        returns the gradient of f = || Y' - HFS ||^2_2, where Y' = Y - HG
        with respect to S
        """
        Y = self.model_analysis(S, HG=None)
        error = self.Y_p - Y
        grad  = - self.F_T(self.R_T(self.H_T(error)))
        return grad

    def _gradient_loss_synthesis_source(self, alpha_S):
        """
        returns the gradient of f = || Y' - H F Phi alpha_S ||^2_2, where Y' = Y - Phi_l alpha_HG
        with respect to alpha_S
        """
        Y = self.model_synthesis(alpha_S, alpha_HG=None)
        error = self.Y_p - Y
        grad  = - self.Phi_T_s(self.F_T(self.R_T(self.H_T(error))))
        return grad

    def gradient_loss_source_ps(self, array_S, array_P):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis_source_ps(S=array_S, P=array_P)
        elif self._formulation == 'synthesis':
            raise NotImplementedError("'synthesis' formulation for source + PS is not supported" )

    def _gradient_loss_analysis_source_ps(self, S, P):
        """
        returns the gradient of f = || Y' - (HFS + P) ||^2_2, where Y' = Y - HG
        with respect to S.
        """
        if self.fixed_point_source_model:
            P_ = self._init_ps_model
        else:
            P_ = P
        model = self.model_analysis(S, HG=None, P=P_)
        error = self.Y_p - model
        grad  = - self.F_T(self.R_T(self.H_T(error)))
        return grad

    def gradient_loss_lens(self, array_HG):
        if self._formulation == 'analysis':
            return self._gradient_loss_analysis_lens(HG=array_HG)
        elif self._formulation == 'synthesis':
            return self._gradient_loss_synthesis_lens(alpha_HG=array_HG)

    def _gradient_loss_analysis_lens(self, HG):
        """
        returns the gradient of f = || Y' - HG ||^2_2, where Y' = Y - HFS
        with respect to HG
        """
        model = self.model_analysis(S=None, HG=HG)
        error = self.Y_p - model
        grad  = - error
        return grad

    def _gradient_loss_synthesis_lens(self, alpha_HG):
        """
        returns the gradient of f = || Y' - Phi_l alpha_HG ||^2_2, where Y' = Y - H F Phi_s alpha_S
        with respect to alpha_HG
        """
        model = self.model_synthesis(alpha_S=None, alpha_HG=alpha_HG)
        error = self.Y_p - model
        grad  = - self.Phi_T_l(error)
        return grad

    def proximal_sparsity_source(self, array_S, threshold, weights):
        array_proxed =  proximals.full_prox_sparsity_positivity(array_S, self.Phi_T_s, self.Phi_s, 
                                                                weights, self.noise.levels_source, 
                                                                threshold, self._increm_high_freq,
                                                                self._n_scales_source, self._sparsity_prior_norm,
                                                                self._formulation, self._force_positivity)
        #  then we set to 0 every pixel that is outside the 'support' in source plane
        array_proxed = self.apply_source_plane_mask(array_proxed)
        return array_proxed

    def proximal_sparsity_lens(self, array_HG, threshold, weights):
        return proximals.full_prox_sparsity_positivity(array_HG, self.Phi_T_l, self.Phi_l, 
                                                       weights, self.noise.levels_image, 
                                                       threshold, self._increm_high_freq,
                                                       self._n_scales_lens_light, self._sparsity_prior_norm,
                                                       self._formulation, self._force_positivity)

    def subtract_source_from_data(self, S):
        """Update "effective" data by subtracting the input source light estimation"""
        source_model = self.model_analysis(S=S, HG=None)
        self.subtract_from_data(source_model)

    def subtract_lens_from_data(self, HG):
        """Update "effective" data by subtracting the input (convolved) lens light estimation"""
        lens_model = self.model_analysis(S=None, HG=HG)
        self.subtract_from_data(lens_model)

    def subtract_point_source_from_data(self, P):
        """Update "effective" data by subtracting the input (convolved) lens light estimation"""
        self.subtract_from_data(P)

    @property
    def algorithm(self):
        if self._formulation == 'analysis':
            return 'FB'
        elif self._formulation == 'synthesis':
            return 'FISTA'

    def prepare_solver(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None,
                       kwargs_special=None, init_lens_light_model=None, init_ps_model=None, init_ps_amp=None,
                       ps_error_map=None):
        """
        Update state of the solver : operators, noise levels, ...
        The order of the steps matters!
        """
        if self.lensingOperator is not None:
            # update lensing operator with lens model
            _, _ = self.lensingOperator.update_mapping(kwargs_lens, kwargs_special=kwargs_special)

            # initialiase noise terms
            if self.noise.include_regridding_error is True:
                magnification_map = self.lensingOperator.magnification_map(kwargs_lens)
                self.noise.update_regridding_error(magnification_map)
            if self.noise.include_point_source_error is True:
                self.noise.update_point_source_error(ps_error_map)
        
        # setup and initialize the point source components
        if self.no_point_source is False:
            self._prepare_point_source(kwargs_ps, kwargs_special, init_ps_model, init_ps_amp)

        # fill masked pixels with background noise
        self.clean_masked_data(self.noise.background_rms, init_ps_model=init_ps_model)

        # update the noise map used for thresholding based on 'cleaned' data from point sources
        if self.no_point_source is False and self._ps_filter_residuals is True:
            self.noise.re_estimate_noise_map_for_ps(self.effective_image_data, 
                                                    self.likelihood_mask,
                                                    init_ps_model)

        # setup and initialize the rest of the components of the models
        # (that might depend on the update noise map above)
        if self.no_source_light is False:
            self._prepare_source(kwargs_source)
        if self.no_lens_light is False:
            self._prepare_lens_light(kwargs_lens_light, init_lens_light_model)

    def _prepare_source(self, kwargs_source):
        """
        updates source number of decomposition scales, spectral norm and noise levels
        related to the operator H(F(Phi_T_s( . )))
        """
        # update number of decomposition scales
        n_scales_new = kwargs_source[0]['n_scales']
        if n_scales_new == -1:
            num_pix_source = self.lensingOperator.sourcePlane.num_pix
            n_scales_new = int(np.log2(num_pix_source))
            if self._verbose:
                print("Set number of source scales to maximal value J={}".format(n_scales_new))
        self.set_source_wavelet_scales(n_scales_new)
        # update spectral norm of operators
        self.update_spectral_norm_source()
        # update wavelet noise levels in source plane
        self.update_source_noise_levels()

    def _prepare_lens_light(self, kwargs_lens_light, init_lens_light_model):
        """
        updates lens light number of decomposition scales, spectral norm and noise levels
        related to the operator Phi_T_l( . )

        Spectral norm and noise levels related to the Phi_T_l operator
        are not updated if the number of decomposition scales has not changed
        """
        # TODO: support upsampling/downsampling operator for image plane noise levels
        # get n_scales for lens light before update
        n_scales_old = self.n_scales_lens_light
        n_scales_new = kwargs_lens_light[0]['n_scales']
        if n_scales_new == -1:
            num_pix_image = self.lensingOperator.imagePlane.num_pix
            n_scales_new = int(np.log2(num_pix_image))
            if self._verbose:
                print("Set number of lens light scales to maximal value J={}".format(n_scales_new))
        # update number of decomposition scales
        self.set_lens_wavelet_scales(n_scales_new)
        if n_scales_old is None or n_scales_new != n_scales_old:
            # update spectral norm of operators
            self.update_spectral_norm_lens()
            # update wavelet noise levels in image plane
            self.update_image_noise_levels()
        # lens light initial model, if any
        if getattr(self, '_init_lens_light_model', None) is not None:
            print("SparseSolverBase: warning, initial guess for lens light is being updated")
        self._init_lens_light_model = init_lens_light_model

    def _prepare_point_source(self, kwargs_ps, kwargs_special, init_ps_model, init_ps_amp):
        if init_ps_model is None:
            raise ValueError("A rough point source model is required")
        # initialize point source model (a map of pixels) and individual amplitudes
        self._init_ps_model = init_ps_model
        self._init_ps_amp = init_ps_amp
        # build a mask for point source regions
        if self._ps_filter_residuals is True:
            mask_shape = self.image_data.shape
            delta_pix = self.data_pixel_width
            ra_ps_list, dec_ps_list = self._get_ps_coordinates(kwargs_ps, kwargs_special)
            # translate the PS coordinates so origin is lower left
            ra_ps_pix, dec_ps_pix = self.data_coord2pix(ra_ps_list, dec_ps_list)
            ra_ps_lowerleft, dec_ps_lowerleft = ra_ps_pix * delta_pix, dec_ps_pix * delta_pix
            # construct the mask with 0s in point source regions, 1s elsewhere
            ps_mask_list = mask_util.get_point_source_mask(mask_shape, delta_pix,
                                                           dec_ps_lowerleft, ra_ps_lowerleft,
                                                           self._ps_radius_regions,
                                                           smoothed=True)
            self._set_point_source_mask(ps_mask_list)

    @staticmethod
    def _get_ps_coordinates(kwargs_ps, kwargs_special):
        ra_ps, dec_ps = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        if 'delta_x_image' in kwargs_special:
            delta_x, delta_y = kwargs_special['delta_x_image'], kwargs_special['delta_y_image']
            delta_x_new = np.zeros(len(ra_ps))
            delta_x_new[0:len(delta_x)] = delta_x[:]
            delta_y_new = np.zeros(len(dec_ps))
            delta_y_new[0:len(delta_y)] = delta_y[:]
            ra_ps  = ra_ps  + delta_x_new
            dec_ps = dec_ps + delta_y_new
        return ra_ps, dec_ps

    def update_source_noise_levels(self):
        self.noise.update_source_levels(self.num_pix_image, self.num_pix_source,
                                        self.Phi_T_s, self.F_T, self.R_T,
                                        psf_kernel=self.psf_kernel)

    def update_image_noise_levels(self):
        self.noise.update_image_levels(self.num_pix_image, self.Phi_T_l)

    def _update_weights(self, alpha_S=None, alpha_HG=None, threshold=None):
        if alpha_S is not None:
            lambda_S = np.copy(self.noise.levels_source)
            if threshold is None:
                threshold = self._k_min
            lambda_S[1:, :, :] *= threshold
            lambda_S[0, :, :] *= (threshold + self._increm_high_freq)
            weights_S  = 1. / ( 1 + np.exp(10 * (alpha_S - lambda_S)) )  # fixed Eq. (C.1)
        else:
            weights_S = None
        if alpha_HG is not None:
            lambda_HG = np.copy(self.noise.levels_image)
            lambda_HG[1:, :, :] *= threshold
            lambda_HG[0, :, :] *= (threshold + self._increm_high_freq)
            weights_HG = 1. / ( 1 + np.exp(10 * (alpha_HG - lambda_HG)) )  # fixed Eq. (C.1)
        else:
            weights_HG = None
        return weights_S, weights_HG

    def _estimate_threshold_source(self, data, fraction=0.9, exclude_mask=None):
        """
        estimate maximum threshold, in units of noise, used for thresholding wavelets
        coefficients during optimization
        
        Parameters
        ----------
        data : array_like
            Imaging data.
        fraction : float, optional
            From 0 to 1, fraction of the maximum value of the image in transformed space, normalized by noise, that is returned as a threshold.
        exclude_mask : array_like
            Binary mask to exclude (where == 1) some pixels from being included in the threshold estimate (e.g. high residuals at point source locations)
        
        Returns
        -------
        float
            Threshold level.
        """
        if self._threshold_decrease_type == 'none':
            return self._k_min

        if exclude_mask is None:
            exclude_mask = np.ones_like(data)

        # get pre-computed noise esimate in source plane
        noise_no_coarse = self.noise.levels_source[:-1, :, :]

        # compute coefficients of the source component
        data_  = data * exclude_mask
        coeffs = self.Phi_T_s(self.F_T(self.R_T(self.H_T(data_))))
        coeffs_no_coarse = coeffs[:-1, :, :]
        coeffs_norm = self.M_s(coeffs_no_coarse / noise_no_coarse)

        # indices to consider
        indices = np.where(noise_no_coarse != 0)
        max_value = np.max(coeffs_norm[indices])

        # fraction of max value, so only the highest coeffs is able to enter the solution
        threshold = fraction * max_value
        return threshold

    def _estimate_threshold_lens(self, data, fraction=0.9):
        """
        estimate maximum threshold, in units of noise, used for thresholding wavelets
        coefficients during optimization
        
        Parameters
        ----------
        data : array_like
            Imaging data.
        fraction : float, optional
            From 0 to 1, fraction of the maximum value of the image in transformed space, normalized by noise, that is returned as a threshold.
        
        Returns
        -------
        float
            Threshold level.
        """
        if self._threshold_decrease_type == 'none':
            return self._k_min

        # get pre-computed noise esimate in source plane
        noise_no_coarse = self.noise.levels_image[:-1, :, :]

        # compute coefficients of the source component
        coeffs = self.Phi_T_l(data)
        coeffs_no_coarse = coeffs[:-1, :, :]
        coeffs_norm = self.M(coeffs_no_coarse / noise_no_coarse)

        # indices to consider
        indices = np.where(noise_no_coarse != 0)
        max_value = np.max(coeffs_norm[indices])

        # fraction of max value, so only the highest coeffs is able to enter the solution
        threshold = fraction * max_value
        return threshold

    def _estimate_threshold_MOM(self, data_minus_HFS, data_minus_HG=None):
        """
        Follows a mean-of-maximum strategy (MOM) to estimate thresholds for blind source separation with two components,
        typically in a problem solved through morphological component analysis (see Bobin et al. 2007).
        Note that we compute the MOM in image plane, even for the source component.
        
        Parameters
        ----------
        data_minus_HFS : array_like
            2D array of the imaging data with lensed convolved source subtracted.
        data_minus_HG : array_like, optional
            2D array of the imaging data with convolved lens light subtracted.
        
        Returns
        -------
        float
            Estimated threshold in the sense of the MOM.
        """
        if self._threshold_decrease_type == 'none':
            return self._k_min

        noise_no_coarse = self.noise.levels_image[:-1, :, :]

        coeffs1_no_coarse = self.Phi_T_l(self.R_T(data_minus_HFS))[:-1, :, :]
        coeffs1_norm = self.M(coeffs1_no_coarse / noise_no_coarse)
        coeffs1_norm[noise_no_coarse == 0] = 0
        max_HFS = np.max(np.abs(coeffs1_norm))

        if data_minus_HG is not None:
            coeffs2_no_coarse = self.Phi_T_l(self.R_T(data_minus_HG))[:-1, :, :]
            coeffs2_norm = self.M(coeffs2_no_coarse / noise_no_coarse)
            coeffs2_norm[noise_no_coarse == 0] = 0
            max_HG = np.max(np.abs(coeffs2_norm))
        else:
            max_HG = max_HFS

        maxs = np.array([max_HFS, max_HG])
        return maxs.min() + 0.001 * np.abs(max_HFS - max_HG)      # SLIT_MCA version
        # return maxs.min() - 0.01 * ( maxs.max() - maxs.min() )  # MuSCADeT version
        # return np.mean(maxs)                                    # original mean-of-max from Bobin et al. 2007

    def _update_threshold(self, k, k_init, n_iter, n_iter_fix=5):
        """Computes a exponentially decreasing value, for a given loop index, starting at a specified value.
    
        Parameters
        ----------
        k : float
            Current threshold.
        k_init : float
            Threshold value at iteration 0.
        n_iter : int
            Total number of iterations.
        n_iter_fix : int, optional.
            Number of iteration for which the threshold equals its minimum set vaélue `self._k_min`.
            Defaults to 5.
        
        Returns
        -------
        float
            Decreased threshold, corresponding to the type of decrease.
        """
        if self._threshold_decrease_type == 'none':
            return self._k_min
        elif self._threshold_decrease_type in ['lin', 'linear']:
            return util.linear_decrease(k, k_init, self._k_min, n_iter, n_iter_fix)
        elif self._threshold_decrease_type in ['exp', 'exponential']:
            return util.exponential_decrease(k, k_init, self._k_min, n_iter, n_iter_fix)
