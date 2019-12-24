__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization import algorithms
from slitronomy.Optimization import proximals
from slitronomy.Util import util


class SparseSolverSourceLens(SparseSolverSource):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, lens_model_class, source_model_class, lens_light_model_class,
                 psf_class=None, convolution_class=None, likelihood_mask=None, lensing_operator='simple',
                 subgrid_res_source=1, minimal_source_plane=True, fix_minimal_source_plane=True, min_num_pix_source=10,
                 max_threshold=5, max_threshold_high_freq=None, num_iter_source=50, num_iter_lens=50, num_weights=1, sparsity_prior_norm=1, force_positivity=True, 
                 formulation='analysis', verbose=False, show_steps=False):

        super(SparseSolverSourceLens, self).__init__(data_class, lens_model_class, source_model_class, lens_light_model_class=lens_light_model_class,
                                                     psf_class=psf_class, convolution_class=convolution_class, likelihood_mask=likelihood_mask, 
                                                 lensing_operator=lensing_operator, subgrid_res_source=subgrid_res_source, 
                                                 minimal_source_plane=minimal_source_plane, fix_minimal_source_plane=fix_minimal_source_plane,
                                                 min_num_pix_source=min_num_pix_source,
                                                 sparsity_prior_norm=sparsity_prior_norm, force_positivity=force_positivity, 
                                                 formulation=formulation, verbose=verbose, show_steps=show_steps,
                                                 max_threshold=max_threshold, max_threshold_high_freq=max_threshold_high_freq, 
                                                 num_iter=num_iter_source, num_weights=num_weights)
        self._n_iter_lens = num_iter_lens

    def _solve(self):
        """
        implements the SLIT_MCA algorithm
        """
        # set the gradient step
        mu_s = 1. / self.spectral_norm_source
        mu_l = 1. / self.spectral_norm_lens

        # initial guess as background random noise
        S, alpha_S = self.generate_initial_source(guess_type='bkg_noise')
        HG, alpha_HG = self.generate_initial_lens_light(guess_type='bkg_noise')
        if self._show_steps:
            self._plotter.plot_init(S, show_now=True)
            self._plotter.plot_init(HG, show_now=True)

        # initialise weights
        weights_source = 1.
        weights_lens = 1.

        loss_list = []
        red_chi2_list = []
        step_diff_list = []

        ######### Loop to update weights ########

        for j in range(self._n_weights):

            if j == 0 and self.algorithm == 'FISTA':
                fista_xi_l = np.copy(alpha_HG)
                fista_t_l  = 1.
                fista_xi_s = np.copy(alpha_S)
                fista_t_s  = 1.

            # get the proximal operator with current weights
            prox_g_l = lambda x, y: self.proximal_sparsity_lens(x, weights_lens)
            prox_g_s = lambda x, y: self.proximal_sparsity_source(x, weights_source)

            ######### Loop over lens light at fixed weights ########

            for i_l in range(self._n_iter_lens):

                ######### Loop over source light at fixed weights ########

                # get the gradient of the cost function f = || Y - HFS - HG ||^2_2  wth respect to S
                grad_f_s = lambda x: self.gradient_loss_source(x, array_HG=HG)

                # subtract lens light from original image
                self.subtract_from_data(HG)

                for i_s in range(self._n_iter):

                    if self.algorithm == 'FISTA':
                        alpha_S_next, fista_xi_s_next, fista_t_s_next \
                            = algorithms.step_FISTA(alpha_S, fista_xi_s, fista_t_s, grad_f_s, prox_g_s, mu_s)
                        S_next = self.Phi_s(alpha_S_next)

                    elif self.algorithm == 'FB':
                        S_next = algorithms.step_FB(S, grad_f_s, prox_g_s, mu_s)
                        alpha_S_next = self.Phi_T_s(S_next)

                    loss = self.loss(S_next, HG=HG)
                    red_chi2 = self.reduced_chi2(S_next, HG=HG)
                    step_diff = self.norm_diff(S, S_next)
                    loss_list.append(loss)
                    red_chi2_list.append(red_chi2)
                    step_diff_list.append(step_diff)

                    if self._verbose and i_s % 30 == 0:
                        print("iteration {}-{}-{} : loss = {:.4f}, red-chi2 = {:.4f}, step_diff = {:.2e}"
                              .format(j, i_l, i_s, loss, red_chi2, step_diff))

                    if self._show_steps and i_s % int(self._n_iter/2) == 0:
                        self._plotter.plot_step(S_next, iter_1=j, iter_2=i_l, iter_3=i_s, show_now=True)

                    # update current estimate of source light and local parameters
                    S = S_next
                    alpha_S = alpha_S_next
                    if self.algorithm == 'FISTA':
                        fista_xi_s, fista_t_s = fista_xi_s_next, fista_t_s_next

                ######### ######## end source light ######## ########

                # get the gradient of the cost function f = || Y - HFS - HG ||^2_2  wth respect to HG
                grad_f_l = lambda x: self.gradient_loss_lens(S, x)

                # subtract source light from original image
                self.subtract_from_data(self.H(self.F(S)))

                if self.algorithm == 'FISTA':
                    alpha_HG_next, fista_xi_l_next, fista_t_l_next \
                        = algorithms.step_FISTA(alpha_HG, fista_xi_l, fista_t_l, grad_f_l, prox_g_l, mu_l)
                    HG_next = self.Phi_l(alpha_HG_next)

                elif self.algorithm == 'FB':
                    HG_next = algorithms.step_FB(HG, grad_f_l, prox_g_l, mu_l)
                    alpha_HG_next = self.Phi_T_l(HG_next)

                loss = self.loss(S, HG=HG_next)
                red_chi2 = self.reduced_chi2(S, HG=HG_next)
                step_diff = self.norm_diff(HG, HG_next)
                loss_list.append(loss)
                red_chi2_list.append(red_chi2)
                step_diff_list.append(step_diff)

                if self._verbose and i_l % 30 == 0:
                    print("iteration {}-{}-{} : loss = {:.4f}, red-chi2 = {:.4f}, step_diff = {:.2e}"
                          .format(j, i_l, i_s, loss, red_chi2, step_diff))

                if self._show_steps and i_l % int(self._n_iter_lens/2) == 0:
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i_l, iter_3=i_s, show_now=True)

                # update current estimate of source light and local parameters
                HG = HG_next
                alpha_HG = alpha_HG_next
                if self.algorithm == 'FISTA':
                    fista_xi_l, fista_t_l = fista_xi_l_next, fista_t_l_next

            ######### ######## end lens light ######## ########

            # update weights
            weights_source, weights_lens = self._update_weights(alpha_S, alpha_HG=alpha_HG)

        ######### ######## end weights ######## ########

        # store results
        source_coeffs_1d = util.cube2array(self.Phi_T_s(S))
        lens_light_coeffs_1d = util.cube2array(self.Phi_T_l(HG))
        all_coeffs_1d = np.concatenate([source_coeffs_1d, lens_light_coeffs_1d])

        self._source_model = S
        self._lens_light_model = HG
        self._solve_track = {
            'loss': np.asarray(loss_list),
            'red_chi2': np.asarray(red_chi2_list),
            'step_diff': np.asarray(step_diff_list),
        }

        if self._show_steps:
            self._plotter.plot_final(self._source_model, show_now=True)
        
        image_model = self.image_model()
        return image_model, self.source_model, self.lens_light_model, all_coeffs_1d

    def _image_model(self, unconvolved=False):
        if not hasattr(self, '_source_model') or not hasattr(self, '_lens_light_model'):
            raise ValueError("You must run the optimization before accessing the source estimate")
        if unconvolved:
            print("WARNING : can not provide a the full deconvolved model when solving source and lens light")
        return self.H(self.F(self._source_model)) + self._lens_light_model

    def _model_analysis(self, S, HG=None):
        return self.H(self.F(S)) + HG

    def _model_synthesis(self, alpha_S, alpha_HG=None):
        return self.H(self.F(self.Phi_s(alpha_S))) + self.Phi_l(alpha_HG)

    def _gradient_loss_analysis_lens(self, S, HG):
        """
        returns the gradient of f = || Y - HFS - HG ||^2_2
        with respect to HG
        """
        model = self._model_analysis(S, HG=HG)
        error = self.Y - model
        grad  = - error
        return grad

    def _gradient_loss_synthesis_lens(self, alpha_S, alpha_HG):
        """
        returns the gradient of f = || Y - H F Phi alphaS - alpha_HG ||^2_2
        with respect to alpha_HG
        """
        model = self._model_synthesis(alpha_S, alpha_HG=alpha_HG)
        error = self.Y - model
        grad  = - self.Phi_T_l(error)
        return grad

    def _proximal_sparsity_analysis_lens(self, HG, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T HG|_0
        or
            g = lambda * |Phi^T HG|_1
        """
        n_scales = self._n_scales_lens_light
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_image_plane

        alpha_HG = self.Phi_T_l(HG)

        # apply proximal operator, with step=1
        alpha_HG_proxed = proximals.prox_sparsity_wavelets(alpha_HG, step=1, level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)
        HG_proxed = self.Phi_l(alpha_HG_proxed)

        if self._force_positivity:
            HG_proxed = proximals.prox_positivity(HG_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        HG_proxed = self.apply_image_plane_mask(HG_proxed)
        return HG_proxed

    def _proximal_sparsity_synthesis_lens(self, alpha_HG, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |alpha_HG|_0
        or
            g = lambda * |alpha_HG|_1
        """
        n_scales = self._n_scales_lens_light
        level_const = self._k_max * np.ones(n_scales)
        level_const[0] = self._k_max_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise_levels_image_plane

        # apply proximal operator, with step=1
        alpha_HG_proxed = proximals.prox_sparsity_wavelets(alpha_HG, step=1, level_const=level_const, level_pixels=level_pixels,
                                                          force_positivity=self._force_positivity, norm=self._sparsity_prior_norm)

        if self._force_positivity:
            alpha_HG_proxed = proximals.prox_positivity(alpha_HG_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        for ns in range(n_scales):
            alpha_HG_proxed[ns, :, :] = self.apply_image_plane_mask(alpha_HG_proxed[ns, :, :])
        return alpha_HG_proxed
