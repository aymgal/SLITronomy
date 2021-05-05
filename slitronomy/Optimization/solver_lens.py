__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_base import SparseSolverBase
from slitronomy.Optimization import algorithms
from slitronomy.Util import util


class SparseSolverLens(SparseSolverBase):

    """Solve for the lens light on a pixelated grid, regularized with starlets"""

    def __init__(self, data_class, image_numerics_class, source_numerics_class, lens_light_model_class,
                 lens_light_model_map=None, num_iter_lens=10, num_iter_weights=3, **base_kwargs):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param image_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for image plane.
        :param source_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for source plane.
        :param source_model_class: lenstronomy.light_model.LightModel instance describing the source light.
        :param num_iter_lens: number of iterations for sparse optimization in the main loop.  
        :param num_iter_weights: number of iterations for l1-norm re-weighting scheme.
        :param base_kwargs: keyword arguments for SparseSolverBase.

        If not set or set to None, 'threshold_decrease_type' in base_kwargs defaults to 'exponential'.
        """
        super(SparseSolverLens, self).__init__(data_class, image_numerics_class, source_numerics_class, 
                                               **base_kwargs)

        # define default threshold decrease strategy
        if 'threshold_decrease_type' not in base_kwargs:
            self._threshold_decrease_type = 'exponential'

        self.add_lens_light(lens_light_model_class)
        self._n_iter_lens = num_iter_lens
        self._init_lens_light_model = lens_light_model_map
        if self._sparsity_prior_norm == 1:
            self._n_iter_weights = num_iter_weights
        else:
            self._n_iter_weights = 1   # reweighting scheme only defined for l1-norm sparsity

    def _ready(self):
        return not self.no_lens_light

    def _solve(self, kwargs_lens=None, kwargs_ps=None, kwargs_special=None):
        """
        implements the SLIT algorithm
        """
        # set the gradient step: 0 < mu < 2/spectral_norm
        mu = 1. / self.spectral_norm_lens

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2
        grad_f = lambda x : self.gradient_loss_lens(x)

        # initial guess as background random noise
        HG, alpha_HG = self.generate_initial_lens_light()
        if self._show_steps:
            self._plotter.plot_init(HG)

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
            thresh_init = self._estimate_threshold_lens(self.Y_eff)
            thresh = thresh_init

            # initial hidden variables
            if j == 0 and self.algorithm == 'FISTA':
                fista_xi = np.copy(alpha_HG)
                fista_t  = 1.

            ######### Loop over iterations at fixed weights ########
            for i in range(self._n_iter_lens):

                # get the proximal operator with current weights, convention is that it takes 2 arguments
                prox_g = lambda x, y: self.proximal_sparsity_lens(x, threshold=thresh, weights=weights)

                if self.algorithm == 'FB':
                    HG_next = algorithms.step_FB(HG, grad_f, prox_g, mu)
                    alpha_HG_next = self.Phi_T_l(HG_next)

                elif self.algorithm == 'FISTA':
                    alpha_HG_next, fista_xi_next, fista_t_next \
                        = algorithms.step_FISTA(alpha_HG, fista_xi, fista_t, grad_f, prox_g, mu)
                    HG_next = self.Phi_l(alpha_HG_next)

                # save current step to track
                self._tracker.save(HG=HG, HG_next=HG_next, print_bool=(i % 10 == 0),
                                   iteration_text="=== iteration {:03}-{:03} ===".format(j, i))

                if self._show_steps and (i % ma.ceil(self._n_iter_lens/2) == 0):
                    self._plotter.plot_step(HG_next, iter_1=j, iter_2=i)

                # update current estimate of source light and local parameters
                HG = HG_next
                alpha_HG = alpha_HG_next
                if self.algorithm == 'FISTA':
                    fista_xi, fista_t = fista_xi_next, fista_t_next

                # update adaptive threshold
                thresh = self._update_threshold(thresh, thresh_init, self._n_iter_lens)

            # update weights if necessary
            if self._n_iter_weights > 1:
                _, weights = self._update_weights(alpha_HG=alpha_HG, threshold=self._k_min)

        # reset data to original data
        self.reset_partial_data()

        # store results
        self._tracker.finalize()
        self._lens_light_model = HG

        # all optimized coefficients (flattened)
        alpha_HG_final = self.Phi_T_l(HG)
        coeffs_HG_1d = util.cube2array(alpha_HG_final)

        if self._show_steps:
            self._plotter.plot_final(self._lens_light_model)

        model = self.image_model(unconvolved=False)
        return model, coeffs_HG_1d, [], []
