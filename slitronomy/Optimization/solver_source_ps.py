__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization.solver_source import SparseSolverBase
from slitronomy.Optimization import algorithms
from slitronomy.Optimization import proximals
from slitronomy.Util import util


class SparseSolverSourcePS(SparseSolverSource):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, lens_model_class, source_model_class, numerics_class, 
                 point_source_painter, point_source_solver,
                 likelihood_mask=None, lensing_operator='interpol',
                 subgrid_res_source=1, minimal_source_plane=True, fix_minimal_source_plane=True, 
                 use_mask_for_minimal_source_plane=True, min_num_pix_source=10,
                 max_threshold=5, max_threshold_high_freq=None, num_iter_source=50, num_iter_ps=50, num_iter_weights=1, 
                 sparsity_prior_norm=1, force_positivity=True, 
                 formulation='analysis', verbose=False, show_steps=False):

        super(SparseSolverSourcePS, self).__init__(data_class, lens_model_class, source_model_class,
                                                     numerics_class, likelihood_mask=likelihood_mask, 
                                                     lensing_operator=lensing_operator, subgrid_res_source=subgrid_res_source, 
                                                     minimal_source_plane=minimal_source_plane, fix_minimal_source_plane=fix_minimal_source_plane,
                                                     min_num_pix_source=min_num_pix_source, use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
                                                     sparsity_prior_norm=sparsity_prior_norm, force_positivity=force_positivity, 
                                                     formulation=formulation, verbose=verbose, show_steps=show_steps,
                                                     max_threshold=max_threshold, max_threshold_high_freq=max_threshold_high_freq, 
                                                     num_iter_source=num_iter_source, num_iter_weights=num_iter_weights)
        self._n_iter_ps = num_iter_ps
        self._ps_painter = point_source_painter
        self._ps_solver = point_source_solver
        self.add_point_source()

    def solve(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_special=None):
        """
        main method to call from outside the class, calling self._solve()

        any class that inherits SparseSolverSource should have the self._solve() method implemented, with correct output.
        """
        # update image <-> source plane mapping from lens model parameters
        size_image, pixel_scale_image, size_source, pixel_scale_source \
            = self.lensingOperator.update_mapping(kwargs_lens, kwargs_special=kwargs_special)
        # get number of decomposition scales
        n_scales_source = kwargs_source[0]['n_scales']
        # save number of scales
        self.set_wavelet_scales(n_scales_source)
        # call solver
        image_model, source_light, lens_light, coeffs_source, coeffs_lens_light, amps_point_source \
            = self._solve(kwargs_lens, kwargs_ps)

        coeffs_lens_light = []

        # concatenate coefficients and fixed parameters
        coeffs = np.concatenate([coeffs_source, coeffs_lens_light])
        scales = [size_source, pixel_scale_source, size_image, pixel_scale_image]
        return image_model, source_light, lens_light, coeffs, scales

    def _solve(self, kwargs_lens, kwargs_ps):
        """
        implements the SLIT algorithm with point source support
        """
        # set the gradient step
        mu = 1. / self.spectral_norm_source

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2
        grad_f = lambda x : self.gradient_loss_source(x)

        # initial guess as background random noise
        S, alpha_S = self.generate_initial_source()
        if self._show_steps:
            self._plotter.plot_init(S)

        # initial point source model
        P = self._ps_painter(kwargs_ps)

        # initialise weights
        weights = 1.

        # initialise tracker
        self._tracker.init()

        ######### Loop to update weights ########
        loss_list = []
        red_chi2_list = []
        step_diff_list = []
        for j in range(self._n_weights):

            ######### Loop over point source optimization at fixed weights ########

            for i_p in range(self._n_iter_ps):

                ######### Loop over source light at fixed weights ########

                # subtract point sources from data
                self.subtract_from_data(P)

                for i_s in range(self._n_iter_source):

                    if j == 0 and self.algorithm == 'FISTA':
                        fista_xi = np.copy(alpha_S)
                        fista_t  = 1.

                    # get the proximal operator with current weights, convention is that it takes 2 arguments
                    prox_g = lambda x, y: self.proximal_sparsity_source(x, weights=weights)

                    if self.algorithm == 'FISTA':
                        alpha_S_next, fista_xi_next, fista_t_next \
                            = algorithms.step_FISTA(alpha_S, fista_xi, fista_t, grad_f, prox_g, mu)
                        S_next = self.Phi_s(alpha_S_next)

                    elif self.algorithm == 'FB':
                        S_next = algorithms.step_FB(S, grad_f, prox_g, mu)
                        alpha_S_next = self.Phi_T_s(S_next)

                    # save current step to track
                    self._tracker.save(S=S, S_next=S_next, 
                                       print_bool=(i_p % 30 == 0 and i_s % 30 == 0),
                                       iteration_text="*** iteration {}-{}-{} ***".format(j, i_p, i_s))

                    if self._show_steps and (i_s % ma.ceil(self._n_iter_source/2) == 0):
                        self._plotter.plot_step(S_next, iter_1=j, iter_2=i)

                    # update current estimate of source light and local parameters
                    S = S_next
                    alpha_S = alpha_S_next
                    if self.algorithm == 'FISTA':
                        fista_xi, fista_t = fista_xi_next, fista_t_next


                ######### ######## end source light ######## ########


                # subtract source light for point source linear amplitude optimization
                self.subtract_source_from_data(S)

                # solve for point source amplitudes
                current_model = self.model_analysis(S)
                model_without_ps_1d = util.image2array(current_model)  # current model without point sources
                P, P_error, ps_cov_param, ps_param = self._ps_solver(model_without_ps_1d, kwargs_lens, kwargs_ps, 
                                                                     kwargs_special=None, inv_bool=False)

                if self._show_steps and i_p % ma.ceil(self._n_iter_ps/2) == 0 and i_s == self._n_iter_source-1:
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i_p, iter_3=i_s)
                    # self._plotter.plot_step(HG_next, iter_1=j, iter_2=i_l, iter_3=i_s)

            ######### ######## end point source ######## ########

            # update weights if necessary
            if self._n_weights > 1:
                weights, _ = self._update_weights(alpha_S)

        ######### ######## end weights ######## ########

        # reset effective data to original data
        self.reset_data()

        # store results
        self._tracker.finalize()
        self._source_model = S
        self._ps_model = P

        # all optimized coefficients (flattened)
        coeffs_S_1d = util.cube2array(self.Phi_T_s(S))
        amps_P = ps_param

        if self._show_steps:
            self._plotter.plot_final(self._source_model)

        model = self.image_model(unconvolved=False)
        return model, S, None, coeffs_S_1d, None, amps_P
