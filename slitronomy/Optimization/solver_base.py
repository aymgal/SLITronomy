__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Optimization.noise_levels import NoiseLevels
from slitronomy.Util.solver_plotter import SolverPlotter
from slitronomy.Util.solver_tracker import SolverTracker
from slitronomy.Util import util


class SparseSolverBase(ModelOperators):
    """
    Base class that generally defines a sparse solver
    """

    #TODO: raises an error when number of decomposition scales is not consistent with image size
    # (also when reducing source plane size, re-check consistency)

    #TODO: create classes for lens and source models.
    # E.g. the method project_on_original_grid should be attached to a SourceModel class, not to the solver.

    def __init__(self, data_class, lens_model_class, numerics_class,
                 likelihood_mask=None, source_interpolation='bilinear',
                 subgrid_res_source=1, minimal_source_plane=False, fix_minimal_source_plane=True,
                 use_mask_for_minimal_source_plane=True, min_num_pix_source=20,
                 min_threshold=3, threshold_increment_high_freq=1, threshold_decrease_type='none',
                 fixed_spectral_norm_source=0.95, include_regridding_error=False,
                 sparsity_prior_norm=1, force_positivity=True, formulation='analysis',
                 verbose=False, show_steps=False, thread_count=1):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param lens_model_class: lenstronomy.lens_model.LensModel instance describing the lens mass model.
        :param numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance.
        :param likelihood_mask: boolean mask to exclude pixels from the optimization and chi2 computation.
        Defaults to None.
        :param source_interpolation: type of interpolation of source pixels on the source plane grid.
        It can be 'nearest' for nearest-neighbor or 'bilinear' for bilinear interpolation. Defaults to 'bilinear'.
        :param subgrid_res_source: resolution factor of the source plane wrt to image plane.
        subgrid_res_source = 2 leads to source pixels two times smaller than data pixels. Defaults to 1.
        :param minimal_source_plane: if True, reduce the source plane grid size to the minimum set by min_num_pix_source.
         Defaults to False.
        :param fix_minimal_source_plane: if True, the reduced source grid size will not be updated for a new lens model.
         Defaults to 1.
        :param use_mask_for_minimal_source_plane: if True, use the likelihood_mask to compute minimal source plane.
         Defaults to True.
        :param min_num_pix_source: minimal number of pixels on a side of the square source grid.
        Only used when minimal_source_plane is True. Defaults to 20.
        :param min_threshold: in unit of the noise (sigma), minimum threshold for wavelets denoising.
        Typically between 3 (more conservative thresholding) and 5 (more aggressive thresholding). Defaults to 3.
        :param threshold_increment_high_freq: additive number to the threshold level for highest frequencies on wavelets space.
        Defaults to 1.
        :param threshold_decrease_type: strategy for decreasing the threshold level at each iteration. Can be 'none' (no decrease, directly sets to min_threshold), 'linear' or 'exponential'.
        Defaults to 'exponential'.
        :param fixed_spectral_norm_source: if None, update the spectral norm for the source operator, for optimal gradient descent step size.
        Defaults to 0.97, which is a conservative value typical of most lens models.
        :param sparsity_prior_norm: prior l-norm (0 or 1). If 1, l1-norm and soft-thresholding are applied.
        If 0, it is l0-norm and hard-thresholding. Defaults to 1.
        :param force_positivity: if True, apply positivity constraint to the source flux.
        Defaults to True.
        :param formulation: type of formalism for the minimization problem. 'analysis' solves the problem in direct space.
        'synthesis' solves the peoblem in wavelets space. Defaults to 'analysis'.
        :param verbose: if True, prints statements during optimization.
        Defaults to False.
        :param show_steps: if True, displays plot of the reconstructed light profiles during optimization.
        Defaults to False.
        :param thread_count: number of threads (multithreading) to speedup wavelets computations (only works if pySAP is properly installed).
        Defaults to 1.
        """
        lensing_operator_class = LensingOperator(data_class, lens_model_class, subgrid_res_source=subgrid_res_source,
                                                 likelihood_mask=likelihood_mask, minimal_source_plane=minimal_source_plane,
                                                 fix_minimal_source_plane=fix_minimal_source_plane, min_num_pix_source=min_num_pix_source,
                                                 use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
                                                 source_interpolation=source_interpolation, matrix_prod=True, verbose=verbose)

        super(SparseSolverBase, self).__init__(data_class, lensing_operator_class, numerics_class,
                                               fixed_spectral_norm_source=fixed_spectral_norm_source,
                                               subgrid_res_source=subgrid_res_source, likelihood_mask=likelihood_mask, 
                                               thread_count=thread_count)
        
        # engine that computes noise levels in image / source plane, in wavelets space
        self.noise = NoiseLevels(data_class, subgrid_res_source=subgrid_res_source, boost_where_zero=1,
                                 include_regridding_error=include_regridding_error)

        # fill masked pixels with background noise
        self.fill_masked_data(self.noise.background_rms)

        # threshold level k_min (in units of the noise)
        self._k_min = min_threshold
        if threshold_increment_high_freq < 0:
            raise ValueError("threshold_increment_high_freq cannot be negative")
        else:
            self._increm_high_freq = threshold_increment_high_freq

        # strategy to decrease threshold up to the max threshold above
        self._threshold_decrease_type = threshold_decrease_type

        if sparsity_prior_norm not in [0, 1]:
            raise ValueError("Sparsity prior norm can only be 0 or 1 (l0-norm or l1-norm)")
        self._sparsity_prior_norm = sparsity_prior_norm
        self._formulation = formulation
        self._force_positivity = force_positivity

        self._verbose = verbose
        self._show_steps = show_steps

        self._tracker = SolverTracker(self, verbose=verbose)
        self._plotter = SolverPlotter(self, show_now=True)

    def solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None,
              init_lens_light_model=None, init_ps_model=None):
        """
        main method to call from outside the class, calling self._solve()

        any class that inherits SparseSolverSource should have self._ready() and self._solve() methods implemented, 
        with correct output.
        """
        if not self._ready(): return

        # update lensing operator and noise levels
        prepare_bool = self.prepare_solver(kwargs_lens, kwargs_source, kwargs_lens_light=kwargs_lens_light,
                                           kwargs_special=kwargs_special, init_lens_light_model=init_lens_light_model, init_ps_model=init_ps_model)
        if prepare_bool is False:
            return None, None

        # call solver
        image_model, coeffs_source, coeffs_lens_light, amps_ps = self._solve(kwargs_lens=kwargs_lens, 
                                                                             kwargs_ps=kwargs_ps,
                                                                             kwargs_special=kwargs_special)

        # concatenate optimized parameters (wavelets coefficients, point source amplitudes)
        all_param = np.concatenate([coeffs_source, coeffs_lens_light, amps_ps])
        return image_model, all_param

    def _solve(self, kwargs_lens=None, kwargs_ps=None, kwargs_special=None):
        raise ValueError("This method must be implemented in class that inherits SparseSolverBase")

    def _ready(self):
        raise ValueError("This method must be implemented in class that inherits SparseSolverBase")

    @property
    def track(self):
        return self._tracker.track

    @property
    def component_names(self):
        return 'S', 'HG', 'P'

    def plot_results(self, **kwargs):
        return self._plotter.plot_results(**kwargs)

    @property
    def source_model(self):
        if not hasattr(self, '_source_model'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        return self._source_model

    @property
    def lens_light_model(self):
        if not hasattr(self, '_lens_light_model') and not self.no_lens_light:
            raise ValueError("You must run the optimization before accessing the lens estimate")
        if self.no_lens_light:
            return np.zeros_like(self.image_data)
        return self._lens_light_model

    @property
    def point_source_model(self):
        if not hasattr(self, '_ps_model') and not self.no_point_source:
            raise ValueError("You must run the optimization before accessing the point source estimate")
        if self.no_point_source:
            return np.zeros_like(self.image_data)
        return self._ps_model

    def image_model(self, unconvolved=False):
        if self.no_lens_light and self.no_point_source:
            S = self.source_model
            if unconvolved:
                return self.F(S)
            return self.H(self.F(S))
        elif not self.no_point_source:
            S, P = self.source_model, self.point_source_model
            if unconvolved:
                raise ValueError("Deconvolution is only supported for source light")
            return self.H(self.F(S)) + P
        else:
            S, HG = self.source_model, self.lens_light_model
            if unconvolved:
                raise ValueError("Deconvolution is only supported for source light")
            return self.H(self.F(S)) + HG

    @property
    def reduced_residuals_model(self):
        """ returns || Y - HFS - HG - P ||^2_2 / sigma^2 """
        return self.reduced_residuals(S=self.source_model, 
                                      HG=self.lens_light_model, 
                                      P=self.point_source_model)

    def generate_initial_source(self):
        num_pix = self.num_pix_source
        transform = self.Phi_T_s
        return util.generate_initial_guess_simple(num_pix, transform, self.noise.background_rms)

    def generate_initial_lens_light(self):
        num_pix = self.num_pix_image
        transform = self.Phi_T_l
        return util.generate_initial_guess_simple(num_pix, transform, self.noise.background_rms)

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
        return int(np.sum(self._mask))

    @property
    def best_fit_reduced_chi2(self):
        return self.reduced_chi2(S=self.source_model, HG=self.lens_light_model, P=self.point_source_model)

    def loss(self, S=None, HG=None, P=None):
        """ returns f = || Y - HFS - HG - P ||^2_2 """
        model = self.model_analysis(S=S, HG=HG, P=P)
        error = self.Y_eff - model
        norm_error = np.linalg.norm(error.flatten(), ord=2)  # flatten to ensure L2-norm
        return 0.5 * norm_error**2

    def regularization(self, S=None, HG=None, P=None):
        """ returns p = lambda * || W_S ø alpha_S ||_0,1 + lambda * || W_HG ø alpha_HG ||_0,1 """
        return self.reg_source(S) + self.reg_lens(HG)

    def reg_source(self, S):
        if S is None:
            return 0
        WS = self.noise.levels_source
        lambda_WS = np.zeros_like(WS)
        lambda_WS[0, :, :]  = (self._k_min + self._increm_high_freq) * WS[0, :, :]
        lambda_WS[1:, :, :] = self._k_min * WS[1:, :, :]
        lambda_WS_alpha_S = self.M_s(lambda_WS * self.Phi_T_s(S))
        return np.linalg.norm(lambda_WS_alpha_S.flatten(), ord=self._sparsity_prior_norm)

    def reg_lens(self, HG):
        if HG is None:
            return 0
        WHG = self.noise.levels_image
        lambda_WHG = np.zeros_like(WHG)
        lambda_WHG[0, :, :]  = (self._k_min + self._increm_high_freq) * WHG[0, :, :]
        lambda_WHG[1:, :, :] = self._k_min * WHG[1:, :, :]
        lambda_WHG_alpha_HG = self.M(lambda_WHG * self.Phi_T_l(HG))
        return np.linalg.norm(lambda_WHG_alpha_HG.flatten(), ord=self._sparsity_prior_norm)

    def reduced_residuals(self, S=None, HG=None, P=None):
        """ returns ( Y - HFS - HG - P ) / sigma """
        model = self.model_analysis(S=S, HG=HG, P=P)
        error = self.Y_eff - model
        if hasattr(self, '_ps_error'):
            sigma = self.noise.effective_noise_map + self._ps_error
        else:
            sigma = self.noise.effective_noise_map
        return self.M(error / sigma)

    def reduced_chi2(self, S=None, HG=None, P=None):
        red_res = self.reduced_residuals(S=S, HG=HG, P=P)
        chi2 = np.sum(red_res**2)
        return chi2 / self.num_data_points

    @staticmethod
    def norm_diff(S1, S2):
        """ returns || S1 - S2 ||_2 """
        diff = S1 - S2
        return np.linalg.norm(diff.flatten(), ord=2)  # flatten to ensure L2-norm

    def model_analysis(self, S=None, HG=None, P=None):
        model = 0
        if S is not None:
            model += self.H(self.F(S))
        if HG is not None:
            model += HG
        if P is not None:
            model += P
        return model

    def model_synthesis(self, alpha_S=None, alpha_HG=None, P=None):
        model = 0
        if alpha_S is not None:
            model = self.H(self.F(self.Phi_s(alpha_S)))
        if alpha_HG is not None:
            model += self.Phi_l(alpha_HG)
        if P is not None:
            model += P
        return model

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

    def proximal_sparsity_source(self, array, threshold, weights):
        if self._formulation == 'analysis':
            return self._proximal_sparsity_analysis_source(array, threshold, weights)
        elif self._formulation == 'synthesis':
            return self._proximal_sparsity_synthesis_source(array, threshold, weights)

    def proximal_sparsity_lens(self, array, threshold, weights):
        if self._formulation == 'analysis':
            return self._proximal_sparsity_analysis_lens(array, threshold, weights)
        elif self._formulation == 'synthesis':
            return self._proximal_sparsity_synthesis_lens(array, threshold, weights)

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

    def prepare_solver(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, 
                       kwargs_special=None, init_lens_light_model=None, init_ps_model=None):
        """
        Update state of the solver : operators, noise levels, ...
        The order of the following updates matters!
        """
        # update image <-> source plane mapping from lens model parameters
        try:
            _, _ = self.lensingOperator.update_mapping(kwargs_lens, kwargs_special=kwargs_special)
        except IndexError as e:
            if self._verbose:
                print("LENSING OPERATOR: error during lensing operator construction: {}".format(e))
                print("The above error happened with the following parameters:")
                print("kwargs_lens:", kwargs_lens)
                print("kwargs_special:", kwargs_special)
            return False

        if self.noise.include_regridding_error is True:
            magnification_map = self.lensingOperator.magnification_map(kwargs_lens)
            self.noise.update_regridding_error(magnification_map)

        self._prepare_source(kwargs_source)
        if not self.no_lens_light:
            self._prepare_lens_light(kwargs_lens_light)

        # lens light initial model, if any
        self._init_lens_light_model = init_lens_light_model
        
        # point source initial model, if any
        if not self.no_point_source and init_ps_model is None:
            raise ValueError("A rough point source model is required to optimize point source amplitudes")
        self._init_ps_model = init_ps_model
        
        return True  # roger that

    def _prepare_source(self, kwargs_source):
        """
        updates source number of decomposition scales, spectral norm and noise levels
        related to the operator H(F(Phi_T_s( . )))
        """
        # update number of decomposition scales
        n_scales_new = kwargs_source[0]['n_scales']
        self.set_source_wavelet_scales(n_scales_new)
        # update spectral norm of operators
        self.update_spectral_norm_source()
        # update wavelets noise levels in source plane
        self.update_source_noise_levels()

    def _prepare_lens_light(self, kwargs_lens_light):
        """
        updates lens light number of decomposition scales, spectral norm and noise levels
        related to the operator Phi_T_l( . )

        Spectral norm and noise levels related to the Phi_T_l operator
        are not updated if the number of decomposition scales has not changed
        """
        # get n_scales for lens light before update
        n_scales_old = self.n_scales_lens_light
        n_scales_new = kwargs_lens_light[0]['n_scales']
        # update number of decomposition scales
        self.set_lens_wavelet_scales(n_scales_new)
        if n_scales_old is None or n_scales_new != n_scales_old:
            # update spectral norm of operators
            self.update_spectral_norm_lens()
            # update wavelets noise levels in image plane
            self.update_image_noise_levels()

    def update_source_noise_levels(self):
        self.noise.update_source_levels(self.num_pix_image, self.num_pix_source,
                                        self.Phi_T_s, self.F_T, psf_kernel=self.psf_kernel)

    def update_image_noise_levels(self):
        self.noise.update_image_levels(self.num_pix_image, self.Phi_T_l)

    def _update_weights(self, alpha_S, alpha_HG=None, threshold=None):
        if threshold is None:
            threshold = self._k_min
        lambda_S = self.noise.levels_source
        weights_S  = 1. / ( 1 + np.exp(-10 * (threshold * lambda_S - alpha_S)) )  # fixed Eq. (11) of Joseph et al. 2018
        if alpha_HG is not None:
            lambda_HG = self.noise.levels_image
            weights_HG = 1. / ( 1 + np.exp(-10 * (threshold * lambda_HG - alpha_HG)) )  # fixed Eq. (11) of Joseph et al. 2018
        else:
            weights_HG = None
        return weights_S, weights_HG

    def _estimate_threshold_source(self, data, fraction=0.9):
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
        noise_no_coarse = self.noise.levels_source[:-1, :, :]
        # compute threshold wrt to the source component
        coeffs = self.Phi_T_s(self.F_T(self.H_T(data)))
        coeffs_no_coarse = coeffs[:-1, :, :]
        coeffs_norm = self.M_s(coeffs_no_coarse / noise_no_coarse)
        coeffs_norm[noise_no_coarse == 0] = 0
        return fraction * np.max(coeffs_norm)  # returns a fraction of max value, so only the highest coeffs is able to enter the solution

    def _estimate_threshold_MOM(self, data_minus_HFS, data_minus_HG=None):
        """
        Follows a mean-of-maximum strategy (MOM) to estimate thresholds for blind source separation with two components,
        typically in a problem solved through moprhological component analysis (see Bobin et al. 2007).
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

        coeffs1_no_coarse = self.Phi_T_l(data_minus_HFS)[:-1, :, :]
        coeffs1_norm = self.M(coeffs1_no_coarse / noise_no_coarse)
        coeffs1_norm[noise_no_coarse == 0] = 0
        max_HFS = np.max(np.abs(coeffs1_norm))

        if data_minus_HG is not None:
            coeffs2_no_coarse = self.Phi_T_l(data_minus_HG)[:-1, :, :]
            coeffs2_norm = self.M(coeffs2_no_coarse / noise_no_coarse)
            coeffs2_norm[noise_no_coarse == 0] = 0
            max_HG = np.max(np.abs(coeffs2_norm))
        else:
            max_HG = max_HFS

        maxs = np.array([max_HFS, max_HG])
        return maxs.min() + 0.001 * np.abs(max_HFS - max_HG)      # SLIT_MCA version
        # return maxs.min() - 0.01 * ( maxs.max() - maxs.min() )  # MuSCADeT version
        # return np.mean(maxs)                                    # original mean-of-max from Bobin et al. 2007

    def _threshold_at_iter(self, i, k_init, n_iter, n_iter_fix=5):
        """Computes a exponentially decreasing value, for a given loop index, starting at a specified value.
    
        Parameters
        ----------
        i : int
            Iteration count, from 0 to n_iter - 1.
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
            Decreases threshold at iteration `i`.
        """
        if self._threshold_decrease_type == 'none':
            return self._k_min
        elif self._threshold_decrease_type in ['lin', 'linear']:
            return util.linear_decrease_at_iter(i, k_init, self._k_min, n_iter, n_iter_fix)
        elif self._threshold_decrease_type in ['exp', 'exponential']:
            return util.exponential_decrease_at_iter(i, k_init, self._k_min, n_iter, n_iter_fix)
