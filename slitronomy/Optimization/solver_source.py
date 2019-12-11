__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np

from slitronomy.Optimization.abstract_solver import AbstractSolver
from slitronomy.Optimization import algorithms
from slitronomy.Optimization import proximals
from slitronomy.Util import util


class SparseSolverSource(AbstractSolver):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, lens_model_class, source_model_class, psf_class=None, 
                 convolution_class=None, likelihood_mask=None, 
                 subgrid_res_source=1, minimal_source_plane=True, min_num_pix_source=10,
                 k_max=5, n_iter=50, n_weights=1, sparsity_prior_norm=1, force_positivity=True, 
                 formulation='analysis', verbose=False, show_steps=False):

        super(SparseSolverSource, self).__init__(data_class, lens_model_class, source_model_class, 
                                                 lens_light_model_class=None, psf_class=psf_class, 
                                                 convolution_class=convolution_class, likelihood_mask=likelihood_mask, 
                                                 subgrid_res_source=subgrid_res_source, minimal_source_plane=minimal_source_plane, 
                                                 min_num_pix_source=min_num_pix_source,
                                                 sparsity_prior_norm=sparsity_prior_norm, force_positivity=force_positivity, 
                                                 formulation=formulation, verbose=verbose, show_steps=show_steps)
        self._k_max = k_max
        self._n_iter = n_iter
        self._n_weights = n_weights

    def _solve(self):
        """
        implements the SLIT algorithm
        """
        # set the gradient step
        mu = 1. / self.spectral_norm_source

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2  
        grad_f = lambda x : self.gradient_loss(x)

        # initial guess as background random noise
        S, alpha_S = self.generate_initial_source(guess_type='bkg_noise')
        if self._show_steps:
            self._plotter.plot_init(S, show_now=True)

        # initialise weights
        weights = 1.

        loss_list = []
        red_chi2_list = []
        step_diff_list = []

        ######### Loop to update weights ########
        for j in range(self._n_weights):

            if j == 0 and self.algorithm == 'FISTA':
                fista_xi = np.copy(alpha_S)
                fista_t  = 1.

            # get the proximal operator with current weights
            prox_g = lambda x, y: self.proximal_sparsity(x, y, weights)

            ######### Loop over iterations at fixed weights ########
            for i in range(self._n_iter):

                if self.algorithm == 'FISTA':
                    alpha_S_next, fista_xi_next, fista_t_next \
                        = algorithms.step_FISTA(alpha_S, fista_xi, fista_t, grad_f, prox_g, mu)
                    S_next = self.Phi_s(alpha_S_next)

                elif self.algorithm == 'FB':
                    S_next = algorithms.step_FB(S, grad_f, prox_g, mu)
                    alpha_S_next = self.Phi_T_s(S_next)

                loss = self.loss(S_next)
                red_chi2 = self.reduced_chi2(S_next)
                step_diff = self.norm_diff(S, S_next)
                loss_list.append(loss)
                red_chi2_list.append(red_chi2)
                step_diff_list.append(step_diff)

                if i % 10 == 0 and self._verbose:
                    print("iteration {}-{} : loss = {:.4f}, red-chi2 = {:.4f}, step_diff = {:.2e}"
                          .format(j, i, loss, red_chi2, step_diff))

                if i % int(self._n_iter/2) == 0 and self._show_steps:
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i, show_now=True)

                # update current estimate of source light and local parameters
                S = S_next
                alpha_S = alpha_S_next
                if self.algorithm == 'FISTA':
                    fista_xi, fista_t = fista_xi_next, fista_t_next

            # update weights
            lambda_ = self._k_max * self.noise_levels_source_plane
            weights = 2. / ( 1. + np.exp(-10. * (lambda_ - alpha_S)) )

        # if wanted, pad the final source to original grid
        # S = self.original_grid_source(S)

        # store results
        source_coeffs_1d = util.cube2array(self.Phi_T_s(S))
        self._source_model = S
        self._solve_track = {
            'loss': np.asarray(loss_list),
            'red_chi2': np.asarray(red_chi2_list),
            'step_diff': np.asarray(step_diff_list),
        }

        if self._show_steps:
            self._plotter.plot_final(S_next, show_now=True)
        
        image_model = self.image_model(unconvolved=False)
        return image_model, self.source_model, None, source_coeffs_1d

    def loss(self, S):
        """ returns f = || Y - HFS ||^2_2 """
        model = self.H(self.F(S))
        error = self.Y - model
        norm_error = np.linalg.norm(error, ord=2)
        return norm_error**2

    def reduced_residuals(self, S):
        """ returns || Y - HFS ||^2_2 / sigma^2 """
        model = self.H(self.F(S))
        error = self.Y - model
        return (error / self._sigma_bkg) * self._mask

    def _gradient_loss_analysis(self, S):
        """ returns the gradient of f = || Y - HFS ||^2_2 """
        model = self.H(self.F(S))
        error = self.Y - model
        grad  = - self.F_T(self.H_T(error))
        return grad

    def _gradient_loss_synthesis(self, alpha_S):
        """ returns the gradient of f = || Y - H F Phi alphaS ||^2_2 """
        model = self.H(self.F(self.Phi_s(alpha_S)))
        error = self.Y - model
        grad  = - self.Phi_T_s(self.F_T(self.H_T(error)))
        return grad

    def _proximal_sparsity_analysis(self, S, step, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T S|_0
        or
            g = lambda * |Phi^T S|_1
        """
        n_scales = self._n_scales_source
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max + 1  # means a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_source_plane

        alpha_S = self.Phi_T_s(S)
        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step, level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)
        S_proxed = self.Phi_s(alpha_S_proxed)

        if self._force_positivity:
            S_proxed = proximals.prox_positivity(S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        S_proxed = self.apply_source_plane_mask(S_proxed)
        return S_proxed

    def _proximal_sparsity_synthesis(self, alpha_S, step, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |alpha_S|_0
        or
            g = lambda * |alpha_S|_1
        """
        n_scales = self._n_scales_source
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max + 1  # means a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_source_plane

        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step, level_const=level_const, level_pixels=level_pixels,
                                                          force_positivity=self._force_positivity, norm=self._sparsity_prior_norm)

        if self._force_positivity:
            alpha_S_proxed = proximals.prox_positivity(alpha_S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        alpha_S_proxed = self.apply_source_plane_mask(alpha_S_proxed)
        return alpha_S_proxed
