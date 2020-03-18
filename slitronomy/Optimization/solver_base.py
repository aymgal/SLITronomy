__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
from scipy import signal

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator
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
                 max_threshold=3, max_threshold_high_freq=None,
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
        :param max_threshold: in unit of the noise (sigma), maximum threshold for wavelets denoising.
        Typically between 3 (more conservative thresholding) and 5 (more aggressive thresholding). Defaults to 3.
        :param max_threshold_high_freq: same than max_threshold, but for highest frequencies on wavelets space.
        If None, equals to max_threshold + 1. Defaults to None.
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
        (num_pix_x, num_pix_y) = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._delta_pix = data_class.pixel_width

        # background noise
        self._background_rms = data_class.background_rms

        # noise full covariance \simeq sqrt(poisson_rms^2 + gaussian_rms^2)
        self._noise_map = np.sqrt(data_class.C_D)

        # threshold level k_max (in units of the noise)
        self._k_max = max_threshold
        if max_threshold_high_freq is None:
            self._k_max_high_freq = self._k_max + 1
        else:
            self._k_max_high_freq = max_threshold_high_freq

        lensing_operator_class = LensingOperator(data_class, lens_model_class, subgrid_res_source=subgrid_res_source,
                                                 likelihood_mask=likelihood_mask, minimal_source_plane=minimal_source_plane,
                                                 fix_minimal_source_plane=fix_minimal_source_plane, min_num_pix_source=min_num_pix_source,
                                                 use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
                                                 source_interpolation=source_interpolation)

        super(SparseSolverBase, self).__init__(data_class, lensing_operator_class, numerics_class,
                                               subgrid_res_source=subgrid_res_source,
                                               likelihood_mask=likelihood_mask, thread_count=thread_count)

        # fill masked pixels with background noise
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

    def solve(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None,
              init_ps_model=None):
        """
        main method to call from outside the class, calling self._solve()

        any class that inherits SparseSolverSource should have the self._solve() method implemented, with correct output.
        """
        # update lensing operator and noise levels
        update_bool = self.prepare_solver(kwargs_lens, kwargs_source, kwargs_lens_light=kwargs_lens_light,
                                          kwargs_ps=kwargs_ps, kwargs_special=kwargs_special, 
                                          init_ps_model=init_ps_model)
        if update_bool is False:
            #TODO
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

    @property
    def track(self):
        return self._tracker.track

    def plot_results(self, **kwargs):
        return self._plotter.plot_results(**kwargs)

    @property
    def image_data(self):
        return self._image_data

    @property
    def lensingOperator(self):
        return self._lensing_op

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
        num_pix = self.lensingOperator.sourcePlane.num_pix
        transform = self.Phi_T_s
        return util.generate_initial_guess_simple(num_pix, transform, self._background_rms)

    def generate_initial_lens_light(self):
        num_pix = self.lensingOperator.imagePlane.num_pix
        transform = self.Phi_T_l
        return util.generate_initial_guess_simple(num_pix, transform, self._background_rms)

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
        if S is None: return 0
        lambda_WS = self.noise_levels_source_plane
        lambda_WS[0, :, :]  *= self._k_max_high_freq
        lambda_WS[1:, :, :] *= self._k_max
        lambda_WS_alpha_S = lambda_WS * self.Phi_T_s(S)
        return np.linalg.norm(lambda_WS_alpha_S.flatten(), ord=self._sparsity_prior_norm)

    def reg_lens(self, HG):
        if HG is None: return 0
        lambda_WHG = self.noise_levels_image_plane
        lambda_WHG[0, :, :]  *= self._k_max_high_freq
        lambda_WHG[1:, :, :] *= self._k_max
        lambda_WHG_alpha_HG = lambda_WHG * self.Phi_T_l(HG)
        return np.linalg.norm(lambda_WHG_alpha_HG.flatten(), ord=self._sparsity_prior_norm)

    def reduced_residuals(self, S=None, HG=None, P=None):
        """ returns ( Y - HFS - HG - P ) / sigma """
        model = self.model_analysis(S=S, HG=HG, P=P)
        error = self.Y_eff - model
        if hasattr(self, '_ps_error'):
            sigma = self._noise_map + self._ps_error
        else:
            sigma = self._noise_map
        return self.M(error / sigma)

    def reduced_chi2(self, S=None, HG=None, P=None):
        chi2 = np.sum(self.reduced_residuals(S=S, HG=HG, P=P)**2)
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

    def subtract_point_source_from_data(self, P):
        """Update "effective" data by subtracting the input (convolved) lens light estimation"""
        self.subtract_from_data(P)

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

    @property
    def noise_levels_image_plane(self):
        if not hasattr(self, '_noise_levels_img'):
            self._noise_levels_img = self._compute_noise_levels_img()
        return self._noise_levels_img

    def prepare_solver(self, kwargs_lens, kwargs_source, kwargs_lens_light=None, 
                       kwargs_ps=None, kwargs_special=None, init_ps_model=None):
        # update image <-> source plane mapping from lens model parameters
        try:
            _, _ = self.lensingOperator.update_mapping(kwargs_lens, kwargs_special=kwargs_special)
        except Exception as e:
            if isinstance(e, IndexError) or isinstance(e, ValueError):
                return False #TODO
            raise e

        # update number of decomposition scales
        self.set_source_wavelet_scales(kwargs_source[0]['n_scales'])
        if not self.no_lens_light:
            self.set_lens_wavelet_scales(kwargs_lens_light[0]['n_scales'])

        # update noise levels
        self.recompute_noise_levels()
        
        # point source initial model, if any
        if self.no_point_source:
            self._init_ps_model = None
        else:
            if init_ps_model is None:
                raise ValueError("A rough point source model is meeded as input to optimize point source amplitudes")
            self._init_ps_model = init_ps_model
        return True

    def recompute_noise_levels(self):
        self._noise_levels_src = self._compute_noise_levels_src(boost_where_zero=10)
        if not self.no_lens_light:
            self._noise_levels_img = self._compute_noise_levels_img()

    def _compute_noise_levels_img(self):
        # starlet transform of a dirac impulse in image plane
        dirac = util.dirac_impulse(self.lensingOperator.imagePlane.num_pix)
        dirac_coeffs2 = self.Phi_T_l(dirac)**2

        n_scale, n_pix1, npix2 = dirac_coeffs2.shape
        noise_levels = np.zeros((n_scale, n_pix1, npix2))
        for scale_idx in range(n_scale):
            scale_power2 = np.sum(dirac_coeffs2[scale_idx, :, :])
            noise_levels[scale_idx, :, :] = self._noise_map * np.sqrt(scale_power2)
        return noise_levels

    def _compute_noise_levels_src(self, boost_where_zero):
        """boost_where_zero sets the multiplcative factor in fron tof the average noise levels
        at locations where noise is 0"""
        # get transposed blurring operator
        if self.psf_kernel is None:
            HT = util.dirac_impulse(self.lensingOperator.imagePlane.num_pix)
        else:
            HT = self.psf_kernel.T

        HT_noise_diag = self._noise_map * np.sqrt(np.sum(HT**2))
        FT_HT_noise = self.F_T(HT_noise_diag)

        # introduce artitifically noise to pixels where there are not signal in source plane
        # to ensure threshold of starlet coefficients at these locations
        FT_HT_noise[FT_HT_noise == 0] = np.mean(FT_HT_noise[FT_HT_noise != 0]) * boost_where_zero

        # \Gamma^2 in  Equation (16) of Joseph+18)
        FT_HT_noise2 = FT_HT_noise**2

        # compute starlet transform of a dirac impulse in image plane
        dirac_coeffs = self.Phi_T_s(util.dirac_impulse(self.lensingOperator.sourcePlane.num_pix))

        # \Delta_s^2 in  Equation (16) of Joseph+18)
        dirac_coeffs2 = dirac_coeffs**2

        n_scale, n_pix1, npix2 = dirac_coeffs2.shape
        noise_levels = np.zeros((n_scale, n_pix1, npix2))
        for scale_idx in range(n_scale):
            # starlet transform of dirac impulse at a given scale
            dirac_scale2 = dirac_coeffs2[scale_idx, :, :]
            # Equation (16) of Joseph+18
            levels = signal.fftconvolve(dirac_scale2, FT_HT_noise2, mode='same')
            # save noise at each pixel for this scale
            noise_levels[scale_idx, :, :] = np.sqrt(np.abs(levels))
        return noise_levels

    def _update_weights(self, alpha_S, alpha_HG=None):
        lambda_S = self.noise_levels_source_plane
        weights_S  = 1. / ( 1 + np.exp(-10 * (lambda_S - alpha_S)) )  # fixed Eq. (11) of Joseph et al. 2018
        if alpha_HG is not None:
            lambda_HG = self.noise_levels_image_plane
            weights_HG = 1. / ( 1 + np.exp(-10 * (lambda_HG - alpha_HG)) )  # fixed Eq. (11) of Joseph et al. 2018
        else:
            weights_HG = None
        return weights_S, weights_HG
