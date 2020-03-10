__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_base import SparseSolverBase
from slitronomy.Optimization import algorithms
from slitronomy.Optimization import proximals
from slitronomy.Util import util


class SparseSolverSource(SparseSolverBase):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, lens_model_class, source_model_class, lens_light_model_class=None,
                 psf_class=None, convolution_class=None, likelihood_mask=None, lensing_operator='interpol',
                 subgrid_res_source=1, minimal_source_plane=True, fix_minimal_source_plane=True,
                 min_num_pix_source=10, use_mask_for_minimal_source_plane=True,
                 max_threshold=5, max_threshold_high_freq=None, num_iter=50, num_iter_weights=1,
                 sparsity_prior_norm=1, force_positivity=True,
                 formulation='analysis', verbose=False, show_steps=False):

        # TODO: remove duplicated parameters in __init__ call (use *args and **kwargs)
        super(SparseSolverSource, self).__init__(data_class, lens_model_class, source_model_class,
                                                 lens_light_model_class=lens_light_model_class, psf_class=psf_class,
                                                 convolution_class=convolution_class, likelihood_mask=likelihood_mask,
                                                 lensing_operator=lensing_operator, subgrid_res_source=subgrid_res_source,
                                                 minimal_source_plane=minimal_source_plane, fix_minimal_source_plane=fix_minimal_source_plane,
                                                 use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
                                                 min_num_pix_source=min_num_pix_source,
                                                 sparsity_prior_norm=sparsity_prior_norm, force_positivity=force_positivity,
                                                 formulation=formulation, verbose=verbose, show_steps=show_steps)

        self._k_max = max_threshold
        if max_threshold_high_freq is None:
            self._k_max_high_freq = self._k_max + 1
        else:
            self._k_max_high_freq = max_threshold_high_freq
        self._n_iter = num_iter
        self._n_weights = num_iter_weights

    def _solve(self):
        """
        implements the SLIT algorithm
        """
        # set the gradient step
        mu = 1. / self.spectral_norm_source

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2
        grad_f = lambda x : self.gradient_loss_source(x)

        # initial guess as background random noise
        S, alpha_S = self.generate_initial_source()
        if self._show_steps:
            self._plotter.plot_init(S)

        # initialise weights
        weights = 1.

        # initialise tracker
        self._tracker.init()

        ######### Loop to update weights ########
        loss_list = []
        red_chi2_list = []
        step_diff_list = []
        for j in range(self._n_weights):

            if j == 0 and self.algorithm == 'FISTA':
                fista_xi = np.copy(alpha_S)
                fista_t  = 1.

            # get the proximal operator with current weights, convention is that it takes 2 arguments
            prox_g = lambda x, y: self.proximal_sparsity_source(x, weights=weights)

            ######### Loop over iterations at fixed weights ########
            for i in range(self._n_iter):

                if self.algorithm == 'FISTA':
                    alpha_S_next, fista_xi_next, fista_t_next \
                        = algorithms.step_FISTA(alpha_S, fista_xi, fista_t, grad_f, prox_g, mu)
                    S_next = self.Phi_s(alpha_S_next)

                elif self.algorithm == 'FB':
                    S_next = algorithms.step_FB(S, grad_f, prox_g, mu)
                    alpha_S_next = self.Phi_T_s(S_next)

                # save current step to track
                self._tracker.save(S=S, S_next=S_next, print_bool=(i % 30 == 0),
                                   iteration_text="=== iteration {}-{} ===".format(j, i))

                if self._show_steps and (i % ma.ceil(self._n_iter/2) == 0):
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i)

                # update current estimate of source light and local parameters
                S = S_next
                alpha_S = alpha_S_next
                if self.algorithm == 'FISTA':
                    fista_xi, fista_t = fista_xi_next, fista_t_next

            # update weights if necessary
            if self._n_weights > 1:
                weights, _ = self._update_weights(alpha_S)

            # if j > 0:
            #     import matplotlib.pyplot as plt
            #     fig, axes = plt.subplots(1, alpha_S.shape[0], figsize=(20, 4))
            #     for ns in range(alpha_S.shape[0]):
            #         im = axes[ns].imshow(weights[ns], origin='lower', cmap='gist_stern')
            #         plt.colorbar(im, ax=axes[ns])
            #     plt.show()

        # store results
        self._tracker.finalize()
        self._source_model = S

        # all optimized coefficients (flattened)
        coeffs_S_1d = util.cube2array(self.Phi_T_s(S))

        if self._show_steps:
            self._plotter.plot_final(self._source_model)

        model = self.image_model(unconvolved=False)
        return model, S, None, coeffs_S_1d, None

    def _gradient_loss_analysis_source(self, S):
        """
        returns the gradient of f = || Y' - HFS ||^2_2, where Y' = Y - HG
        with respect to S
        """
        model = self.model_analysis(S, HG=None)
        error = self.Y_eff - model
        grad  = - self.F_T(self.H_T(error))
        return grad

    def _gradient_loss_synthesis_source(self, alpha_S):
        """
        returns the gradient of f = || Y' - H F Phi alpha_S ||^2_2, where Y' = Y - Phi_l alpha_HG
        with respect to alpha_S
        """
        model = self.model_synthesis(alpha_S, alpha_HG=None)
        error = self.Y_eff - model
        grad  = - self.Phi_T_s(self.F_T(self.H_T(error)))
        return grad

    def _proximal_sparsity_analysis_source(self, S, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T S|_0
        or
            g = lambda * |Phi^T S|_1
        """
        n_scales = self._n_scales_source
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_source_plane

        alpha_S = self.Phi_T_s(S)

        # apply proximal operator
        step = 1  # because threshold is already expressed in data units
        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step=step, level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)
        S_proxed = self.Phi_s(alpha_S_proxed)

        if self._force_positivity:
            S_proxed = proximals.prox_positivity(S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        S_proxed = self.apply_source_plane_mask(S_proxed)
        return S_proxed

    def _proximal_sparsity_synthesis_source(self, alpha_S, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |alpha_S|_0
        or
            g = lambda * |alpha_S|_1
        """
        n_scales = self._n_scales_source
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_source_plane

        # apply proximal operator
        step = 1  # because threshold is already expressed in data units
        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step=step, level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)

        if self._force_positivity:
            alpha_S_proxed = proximals.prox_positivity(alpha_S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        for ns in range(n_scales):
            alpha_S_proxed[ns, :, :] = self.apply_source_plane_mask(alpha_S_proxed[ns, :, :])
        return alpha_S_proxed
