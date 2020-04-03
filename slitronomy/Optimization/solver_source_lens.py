__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization import algorithms
from slitronomy.Optimization import proximals
from slitronomy.Util import util


class SparseSolverSourceLens(SparseSolverSource):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, lens_model_class, numerics_class, source_model_class, lens_light_model_class, 
                 likelihood_mask=None, num_iter_source=10, num_iter_lens=10, num_iter_weights=3, **base_kwargs):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param lens_model_class: lenstronomy.lens_model.LensModel instance describing the lens mass model.
        :param numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance.
        :param source_model_class: lenstronomy.light_model.LightModel instance describing the source light.
        :param lens_light_model_class: lenstronomy.light_model.LightModel instance describing the lens light.
        :param num_iter_source: number of iterations for sparse optimization of the source light. 
        :param num_iter_lens: number of iterations for sparse optimization of the lens light. 
        :param num_iter_weights: number of iterations for l1-norm re-weighting scheme.
        :param base_kwargs: keyword arguments for SparseSolverBase.
        """
        super(SparseSolverSourceLens, self).__init__(data_class, lens_model_class, numerics_class, source_model_class,
                                                     likelihood_mask=likelihood_mask, num_iter_source=num_iter_source, 
                                                     num_iter_weights=num_iter_weights, **base_kwargs)
        self.add_lens_light(lens_light_model_class)
        self._n_iter_lens = num_iter_lens

    def _ready(self):
        return not self.no_source_light and not self.no_lens_light

    def _solve(self, kwargs_lens=None, kwargs_ps=None, kwargs_special=None):
        """
        implements the SLIT_MCA algorithm
        """
        # set the gradient step
        mu_s = 1. / self.spectral_norm_source
        mu_l = 1. / self.spectral_norm_lens

        # initial guess as background random noise
        S, alpha_S = self.generate_initial_source()
        if self._init_lens_light_model is not None:
            # a guess for lens light has been provided
            HG = self._init_lens_light_model
            alpha_HG = self.Phi_T_l(HG)
        else:
            HG, alpha_HG = self.generate_initial_lens_light()
        if self._show_steps:
            self._plotter.plot_init(S)
            self._plotter.plot_init(HG)

        # initialise weights
        weights_source = 1.
        weights_lens = 1.

        # initialise tracker
        self._tracker.init()

        ######### Loop to update weights ########
        for j in range(self._n_iter_weights):

            # estimate initial threshold
            model = self.Y_eff if j == 0 else self.model_analysis(S=S)
            thresh_init = self._estimate_threshold_MOM(self.Y)  # first estimation from data itself

            # initial hidden variables
            if j == 0 and self.algorithm == 'FISTA':
                fista_xi_l = np.copy(alpha_HG)
                fista_t_l  = 1.
                fista_xi_s = np.copy(alpha_S)
                fista_t_s  = 1.


            ######### Loop over lens light at fixed weights ########

            for i_l in range(self._n_iter_lens):

                # get adaptive threshold
                DS = self.Y - self.H(self.F(S))
                DG = self.Y - HG
                thresh_MOM  = self._estimate_threshold_MOM(DS, DG)
                thresh_iter = self._threshold_at_iter(i_l, thresh_init, self._n_iter_lens)
                if thresh_MOM < thresh_iter:
                    thresh = thresh_MOM
                    #TODO: update the linear decrease as well!
                else:
                    thresh = thresh_iter

                # get the proximal operator with current weights
                prox_g_s = lambda x, y: self.proximal_sparsity_source(x, threshold=thresh, weights=weights_source)
                prox_g_l = lambda x, y: self.proximal_sparsity_lens(x, threshold=thresh, weights=weights_lens)


                ######### Loop over source light at fixed weights ########

                # subtract lens light from data
                self.subtract_lens_from_data(HG)

                # get the gradient of the cost function f = || Y - HFS - HG ||^2_2  wth respect to S
                grad_f_s = lambda x: self.gradient_loss_source(x)

                for i_s in range(self._n_iter_source):

                    if self.algorithm == 'FISTA':
                        alpha_S_next, fista_xi_s_next, fista_t_s_next \
                            = algorithms.step_FISTA(alpha_S, fista_xi_s, fista_t_s, grad_f_s, prox_g_s, mu_s)
                        S_next = self.Phi_s(alpha_S_next)

                    elif self.algorithm == 'FB':
                        S_next = algorithms.step_FB(S, grad_f_s, prox_g_s, mu_s)
                        alpha_S_next = self.Phi_T_s(S_next)

                    # save current step to track
                    self._tracker.save(S=S, S_next=S_next, 
                                       print_bool=(i_l % 30 == 0 and i_s % 30 == 0),
                                       iteration_text="*** iteration {}-{}-{} ***".format(j, i_l, i_s))
                    
                    # update current estimate of source light and local parameters
                    S = S_next
                    alpha_S = alpha_S_next
                    if self.algorithm == 'FISTA':
                        fista_xi_s, fista_t_s = fista_xi_s_next, fista_t_s_next

                ######### ######## end source light ######## ########

                # subtract source light (lensed and convolved) from data
                self.subtract_source_from_data(S)

                # get the gradient of the cost function f = || Y - HFS - HG ||^2_2  wth respect to HG
                grad_f_l = lambda x: self.gradient_loss_lens(x)

                if self.algorithm == 'FISTA':
                    alpha_HG_next, fista_xi_l_next, fista_t_l_next \
                        = algorithms.step_FISTA(alpha_HG, fista_xi_l, fista_t_l, grad_f_l, prox_g_l, mu_l)
                    HG_next = self.Phi_l(alpha_HG_next)

                elif self.algorithm == 'FB':
                    HG_next = algorithms.step_FB(HG, grad_f_l, prox_g_l, mu_l)
                    alpha_HG_next = self.Phi_T_l(HG_next)

                # save current step to track
                self._tracker.save(HG=HG, HG_next=HG_next, 
                                   print_bool=(i_l % 10 == 0 and i_s == self._n_iter_source-1),
                                   iteration_text="=== iteration {}-{}-{} ===".format(j, i_l, i_s))

                if self._show_steps and i_l % ma.ceil(self._n_iter_lens/2) == 0 and i_s == self._n_iter_source-1:
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i_l, iter_3=i_s)
                    self._plotter.plot_step(HG_next, iter_1=j, iter_2=i_l, iter_3=i_s)

                # update current estimate of source light and local parameters
                HG = HG_next
                alpha_HG = alpha_HG_next
                if self.algorithm == 'FISTA':
                    fista_xi_l, fista_t_l = fista_xi_l_next, fista_t_l_next

            ######### ######## end lens light ######## ########

            # update weights if necessary
            if self._n_iter_weights > 1:
                weights_source, weights_lens = self._update_weights(alpha_S, alpha_HG=alpha_HG)

        ######### ######## end weights ######## ########

        # reset effective data to original data
        self.reset_data()

        # store results
        self._tracker.finalize()
        self._source_model = S
        self._lens_light_model = HG

        # get wavelets coefficients
        alpha_S_final = self.Phi_T_s(self.project_on_original_grid_source(S))
        coeffs_S_1d = util.cube2array(alpha_S_final)
        coeffs_HG_1d = util.cube2array(alpha_HG)
        
        if self._show_steps:
            self._plotter.plot_final(self._source_model)
            self._plotter.plot_final(self._lens_light_model)
        
        model = self.image_model(unconvolved=False)
        return model, coeffs_S_1d, coeffs_HG_1d, []

    def _gradient_loss_analysis_lens(self, HG):
        """
        returns the gradient of f = || Y' - HG ||^2_2, where Y' = Y - HFS
        with respect to HG
        """
        model = self.model_analysis(S=None, HG=HG)
        error = self.Y_eff - model
        grad  = - error
        return grad

    def _gradient_loss_synthesis_lens(self, alpha_HG):
        """
        returns the gradient of f = || Y' - Phi_l alpha_HG ||^2_2, where Y' = Y - H F Phi_s alpha_S
        with respect to alpha_HG
        """
        model = self.model_synthesis(alpha_S=None, alpha_HG=alpha_HG)
        error = self.Y_eff - model
        grad  = - self.Phi_T_l(error)
        return grad

    def _proximal_sparsity_analysis_lens(self, HG, threshold, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T HG|_0
        or
            g = lambda * |Phi^T HG|_1
        """
        n_scales = self._n_scales_lens_light
        level_const = threshold * np.ones(n_scales)
        level_const[0] += self._increm_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise.levels_image

        alpha_HG = self.Phi_T_l(HG)

        # apply proximal operator
        step = 1  # because threshold is already expressed in data units
        alpha_HG_proxed = proximals.prox_sparsity_wavelets(alpha_HG, step=step, level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)
        HG_proxed = self.Phi_l(alpha_HG_proxed)

        if self._force_positivity:
            HG_proxed = proximals.prox_positivity(HG_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        HG_proxed = self.apply_image_plane_mask(HG_proxed)
        return HG_proxed

    def _proximal_sparsity_synthesis_lens(self, alpha_HG, threshold, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |alpha_HG|_0
        or
            g = lambda * |alpha_HG|_1
        """
        n_scales = self._n_scales_lens_light
        level_const = threshold * np.ones(n_scales)
        level_const[0] += self._increm_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise.levels_image

        # apply proximal operator
        step = 1  # because threshold is already expressed in data units
        alpha_HG_proxed = proximals.prox_sparsity_wavelets(alpha_HG, step=step, level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)

        #TODO: positivity applied in starlets space ?
        # if self._force_positivity:
        #     alpha_HG_proxed = proximals.prox_positivity(alpha_HG_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        for ns in range(n_scales):
            alpha_HG_proxed[ns, :, :] = self.apply_image_plane_mask(alpha_HG_proxed[ns, :, :])
        return alpha_HG_proxed
