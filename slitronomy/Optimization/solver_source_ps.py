__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_base import SparseSolverBase
from slitronomy.Optimization import algorithms
from slitronomy.Util import util


class SparseSolverSourcePS(SparseSolverBase):

    """Implements the original SLIT algorithm with point source support"""

    def __init__(self, data_class, image_numerics_class, 
                 source_numerics_class, source_model_class, lens_model_class,
                 num_iter_source=10, num_iter_global=10, num_iter_weights=3, 
                 fix_point_source_model=False, filter_point_source_residuals=False,
                 min_scale_point_source_residuals=2, radius_point_source_residuals=0.2,
                 check_point_source_residuals=False, **base_kwargs):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param lens_model_class: lenstronomy.lens_model.LensModel instance describing the lens mass model.
        :param image_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for image plane.
        :param source_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for source plane.
        :param source_model_class: lenstronomy.light_model.LightModel instance describing the source light.
        :param point_source_linear_solver: method that linearly solve the amplitude of point sources,
        given a source subtracted image. This might change in the future.
        :param num_iter_source: number of iterations for sparse optimization of the source light. 
        :param num_iter_global: number of iterations to alternate between source and point source optimisation. 
        :param num_iter_weights: number of iterations for l1-norm re-weighting scheme.
        :param fix_point_source_model: if True, does not invert for linear amplitude of point sources during optimization
        Default: False.
        :param filter_point_source_residuals: if True, filter pixels in regions around point sources. Default: False.
        :param min_scale_point_source_residuals: if filter_point_source_residuals is True, minimum starlet scale arcs
        to be included in point source regions.
        :param radius_point_source_residuals: if filter_point_source_residuals is True, radius (arcsec) of point source regions to consider.
        :param check_point_source_residuals: if True, show a plot for checking point source residuals filtering. Default: False.
        :param base_kwargs: keyword arguments for SparseSolverBase.
        
        If not set or set to None, 'threshold_decrease_type' in base_kwargs defaults to 'exponential'.
        """
        super(SparseSolverSourcePS, self).__init__(data_class, image_numerics_class, source_numerics_class, 
                                                   lens_model_class=lens_model_class,
                                                   **base_kwargs)
        # define default threshold decrease strategy
        if 'threshold_decrease_type' not in base_kwargs:
            self._threshold_decrease_type = 'exponential'

        self.add_source_light(source_model_class)
        self.add_point_source(fix_point_source_model, filter_point_source_residuals,
                              radius_point_source_residuals, min_scale_point_source_residuals,
                              check_point_source_residuals)

        self._n_iter_global = num_iter_global
        self._n_iter_source = num_iter_source
        if self._sparsity_prior_norm == 1:
            self._n_iter_weights = num_iter_weights
        else:
            self._n_iter_weights = 1   # reweighting scheme only defined for l1-norm sparsity

    def _ready(self):
        return not self.no_source_light and not self.no_point_source

    def _solve(self, kwargs_lens, kwargs_ps, kwargs_special):
        """
        implements the SLIT algorithm with point source support
        """
        if self._ps_solver is None:
            raise ValueError("No function has been provided for point source amplitude inversion")

        # set the gradient step
        mu = 1. / self.spectral_norm_source

        # initial guess as background random noise
        S, alpha_S = self.generate_initial_source()
        if self._show_steps:
            self._plotter.plot_init(S)

        # initial point source model
        P = self._init_ps_model

        # initialise weights
        weights = 1.

        # initialise tracker
        self._tracker.init()

        ######### Loop to update weights ########
        loss_list = []
        red_chi2_list = []
        step_diff_list = []
        for j in range(self._n_iter_weights):

            ######### Loop over point source optimization at fixed weights ########

            for i in range(self._n_iter_global):

                ######### Loop over source light at fixed weights ########

                # estimate initial threshold after subtraction of point sources
                exclude_mask = self.point_source_mask(split=False)
                thresh_init = self._estimate_threshold_source(self.Y_p - P, exclude_mask=exclude_mask)
                thresh = thresh_init

                # get the gradient of the cost function, which is f = || Y - (HFS+P) ||^2_2
                grad_f = lambda x: self.gradient_loss_source_ps(x, P)

                # initial hidden variables
                if j == 0 and self.algorithm == 'FISTA':
                    fista_xi = np.copy(alpha_S)
                    fista_t  = 1.

                for i_s in range(self._n_iter_source):

                    # get the proximal operator with current weights, convention is that it takes 2 arguments
                    prox_g = lambda x, y: self.proximal_sparsity_source(x, threshold=thresh, weights=weights)

                    if self.algorithm == 'FISTA':
                        alpha_S_next, fista_xi_next, fista_t_next \
                            = algorithms.step_FISTA(alpha_S, fista_xi, fista_t, grad_f, prox_g, mu)
                        S_next = self.Phi_s(alpha_S_next)

                    elif self.algorithm == 'FB':
                        S_next = algorithms.step_FB(S, grad_f, prox_g, mu)
                        alpha_S_next = self.Phi_T_s(S_next)

                    # save current step to track
                    self._tracker.save(S=S, S_next=S_next, P=P,
                                       print_bool=(i % 10 == 0 and i_s % 10 == 0),
                                       iteration_text="{:03}-{:03}-{:03}".format(j, i, i_s))

                    if self._show_steps and (i_s % ma.ceil(self._n_iter_source/2) == 0):
                        self._plotter.plot_step(S_next, iter_1=j, iter_2=i, iter_3=i_s)

                    # update current estimate of source light and local parameters
                    S = S_next
                    alpha_S = alpha_S_next
                    if self.algorithm == 'FISTA':
                        fista_xi, fista_t = fista_xi_next, fista_t_next

                    # update adaptive threshold
                    thresh = self._update_threshold(thresh, thresh_init, self._n_iter_source)

                ######### ######## end source light ######## ########

                if self.fixed_point_source_model is True:
                    # break the loop for iterative point source amplitude refinement
                    break
                else:
                    # based on current source model, re-estimate individual point amplitudes
                    P, _, _, ps_param = self._solve_point_source_amplitudes(S, kwargs_lens, kwargs_ps, kwargs_special)

                    if self._show_steps and i % ma.ceil(self._n_iter_global/2) == 0 and i_s == self._n_iter_source-1:
                        self._plotter.plot_step(S_next, iter_1=j, iter_2=i, iter_3=i_s)

            ######### ######## end point source ######## ########

            # update weights if necessary
            if self._n_iter_weights > 1:
                weights, _ = self._update_weights(alpha_S)

        ######### ######## end weights ######## ########

        # re-estimate individual point amplitudes through weighted least squares
        P, _, _, ps_param = self._solve_point_source_amplitudes(S, kwargs_lens, kwargs_ps, kwargs_special)

        # reset effective data to original data
        self.reset_partial_data()

        # store results
        self._tracker.finalize()
        self._source_model = S
        self._ps_model = P

        # optimized starlet coefficients (flattened)
        alpha_S_final = self.Phi_T_s(self.project_on_original_grid_source(S))
        coeffs_S_1d = util.cube2array(alpha_S_final)

        # optimized point source amplitudes (if not fixed)
        if self.fixed_point_source_model:
            amps_P = self._init_ps_amp
        else:
            amps_P = ps_param

        if self._show_steps:
            self._plotter.plot_final(self._source_model)

        model = self.image_model(unconvolved=False)
        return model, coeffs_S_1d, [], amps_P

    def _solve_point_source_amplitudes(self, S, kwargs_lens, kwargs_ps, kwargs_special):
        # subtract source light for point source linear amplitude optimization
        self.subtract_source_from_data(S)

        # solve for point source amplitudes
        data_response = util.image2array(self.Y_p)[self._mask_1d]
        P, ps_error, ps_cov_param, ps_param = self._ps_solver(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, 
                                                              kwargs_special=kwargs_special, inv_bool=False,
                                                              data_response_external=data_response)
        self.reset_partial_data()
        return P, ps_error, ps_cov_param, ps_param
