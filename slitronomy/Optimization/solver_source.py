__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma
from functools import partial

from slitronomy.Optimization.solver_base import SparseSolverBase
from slitronomy.Optimization import algorithms
from slitronomy.Optimization import proximals
from slitronomy.Util import util


class SparseSolverSource(SparseSolverBase):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, lens_model_class, image_numerics_class, source_numerics_class, source_model_class, 
                 num_iter_source=10, num_iter_weights=3, **base_kwargs):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param lens_model_class: lenstronomy.lens_model.LensModel instance describing the lens mass model.
        :param image_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for image plane.
        :param source_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for source plane.        :param source_model_class: lenstronomy.light_model.LightModel instance describing the source light.
        :param num_iter_source: number of iterations for sparse optimization of the source light. 
        :param num_iter_lens: number of iterations for sparse optimization of the lens light. 
        :param num_iter_weights: number of iterations for l1-norm re-weighting scheme.
        :param base_kwargs: keyword arguments for SparseSolverBase.

        If not set or set to None, 'threshold_decrease_type' in base_kwargs defaults to 'exponential'.
        """
        # remove settings not related to this solver
        base_kwargs_c = copy.deepcopy(base_kwargs)
        _ = base_kwargs_c.pop('num_iter_lens', None)
        _ = base_kwargs_c.pop('num_iter_global', None)

        # define default threshold decrease strategy
        if base_kwargs_c.get('threshold_decrease_type', None) is None:
            base_kwargs_c['threshold_decrease_type'] = 'exponential'

        super(SparseSolverSource, self).__init__(data_class, lens_model_class, image_numerics_class, source_numerics_class,
                                                 **base_kwargs_c)
        self.add_source_light(source_model_class)
        self._n_iter_source = num_iter_source
        if self._sparsity_prior_norm == 1:
            self._n_iter_weights = num_iter_weights
        else:
            self._n_iter_weights = 1   # reweighting scheme only defined for l1-norm sparsity

    def _ready(self):
        return not self.no_source_light

    def _solve(self, kwargs_lens=None, kwargs_ps=None, kwargs_special=None):
        """
        implements the SLIT algorithm
        """
        # set the gradient step: 0 < mu < 2/spectral_norm
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
        for j in range(self._n_iter_weights):

            # estimate initial threshold
            thresh_init = self._estimate_threshold_source(self.Y)
            thresh = thresh_init

            # initial hidden variables
            if j == 0 and self.algorithm == 'FISTA':
                fista_xi = np.copy(alpha_S)
                fista_t  = 1.

            ######### Loop over iterations at fixed weights ########
            for i in range(self._n_iter_source):

                # get the proximal operator with current weights, convention is that it takes 2 arguments
                prox_g = self.proximal_source(threshold=thresh, weights=weights)

                if self.algorithm == 'FISTA':
                    alpha_S_next, fista_xi_next, fista_t_next \
                        = algorithms.step_FISTA(alpha_S, fista_xi, fista_t, grad_f, prox_g, mu)
                    S_next = self.Phi_s(alpha_S_next)

                elif self.algorithm == 'FB':
                    S_next = algorithms.step_FB(S, grad_f, prox_g, mu)

                # save current step to track
                self._tracker.save(S=S, S_next=S_next, print_bool=(i % 30 == 0),
                                   iteration_text="=== iteration {}-{} ===".format(j, i))

                if self._show_steps and (i % ma.ceil(self._n_iter_source/2) == 0):
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i)

                # update current estimate of source light and local parameters
                if self.algorithm == 'FISTA':
                    alpha_S = alpha_S_next
                    fista_xi, fista_t = fista_xi_next, fista_t_next
                elif self.algorithm == 'FB':
                    S = S_next

                # update adaptive threshold
                thresh = self._update_threshold(thresh, thresh_init, self._n_iter_source)

            # update weights if necessary
            if self._n_iter_weights > 1:
                weights, _ = self._update_weights(S, threshold=self._k_min, wavelet_index=0)

        # store results
        self._tracker.finalize()
        self._source_model = S

        # all optimized coefficients (flattened)
        alpha_S_final = self.Phi_T_s(self.project_on_original_grid_source(S), k=0)
        coeffs_S_1d = util.cube2array(alpha_S_final)

        if self._show_steps:
            self._plotter.plot_final(self._source_model)

        model = self.image_model(unconvolved=False)
        return model, coeffs_S_1d, [], []

    def _gradient_loss_analysis_source(self, S):
        """
        returns the gradient of f = || Y' - HFS ||^2_2, where Y' = Y - HG
        with respect to S
        """
        model = self.model_analysis(S, HG=None)
        error = self.Y_eff - model
        grad  = - self.F_T(self.R_T(self.H_T(error)))
        return grad

    def _gradient_loss_synthesis_source(self, alpha_S):
        """
        returns the gradient of f = || Y' - H F Phi alpha_S ||^2_2, where Y' = Y - Phi_l alpha_HG
        with respect to alpha_S
        """
        model = self.model_synthesis(alpha_S, alpha_HG=None)
        error = self.Y_eff - model
        grad  = - self.Phi_T_s(self.F_T(self.R_T(self.H_T(error))))
        return grad

    def _proximal_analysis_source(self, threshold, weights, reweight_index=0):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |Phi^T S|_0 + i_{>=0}(S)
        or
            g = lambda * |Phi^T S|_1 + i_{>=0}(S)
        """
        prox_list = []

        for k in range(self.num_wavelet_dicts_source):

            noise_levels = self.noise.levels_source(k=k)
            n_scales = noise_levels.shape[0]
            level_const = threshold * np.ones(n_scales)
            level_const[0] += self._increm_high_freq  # increment threshold for first decomposition scale
            if k == reweight_index:
                # reweighting acting only on the selected wavelet (relevant for multiple sparsity constraints) 
                level_pixels = weights * noise_levels
            else:
                level_pixels = np.ones_like(noise_levels)

            def prox_sparsity_analysis(S, step):
                # go to transformed space
                alpha_S = self.Phi_T_s(S, k=k)
                # apply proximal operator
                step_eff = 1  # because threshold is already expressed in data units
                alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step=step_eff, 
                                                                  level_const=level_const, 
                                                                  level_pixels=level_pixels,
                                                                  l_norm=self._sparsity_prior_norm)
                # back to direct space
                S_proxed = self.Phi_s(alpha_S_proxed, k=k)
                return S_proxed

            prox_list.append(prox_sparsity_analysis)
        
        # positivity constraint
        prox_list.append(proximals.prox_positivity)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        mask = self.lensingOperator.sourcePlane.effective_mask
        prox_list.append(partial(proximals.prox_mask, mask=mask))

        return prox_list

    def _proximal_synthesis_source(self, alpha_S, threshold, weights):
        """
        returns the proximal operator of the regularisation term
            g = lambda * |alpha_S|_0
        or
            g = lambda * |alpha_S|_1
        """
        n_scales = self.noise.levels_source.shape[0]
        level_const = threshold * np.ones(n_scales)
        level_const[0] += self._increm_high_freq  # possibly a stronger threshold for first decomposition levels (small scales features)
        level_pixels = weights * self.noise.levels_source

        # apply proximal operator
        step = 1  # because threshold is already expressed in data units
        alpha_S_proxed = proximals.prox_sparsity_wavelets(alpha_S, step=step, 
                                                          level_const=level_const, level_pixels=level_pixels,
                                                          l_norm=self._sparsity_prior_norm)

        #TODO: positivity applied in starlets space ?
        # if self._force_positivity:
        #     alpha_S_proxed = proximals.prox_positivity(alpha_S_proxed)

        # finally, set to 0 every pixel that is outside the 'support' in source plane
        for ns in range(n_scales):
            alpha_S_proxed[ns, :, :] = self.apply_source_plane_mask(alpha_S_proxed[ns, :, :])
        return alpha_S_proxed
