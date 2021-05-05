__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_base import SparseSolverBase
from slitronomy.Optimization import algorithms
from slitronomy.Util import util


class SparseSolverSourceLens(SparseSolverBase):

    """Implements an improved version of the original SLIT algorithm (https://github.com/herjy/SLIT)"""

    def __init__(self, data_class, image_numerics_class, source_numerics_class, 
                 source_model_class, lens_light_model_class, lens_model_class, 
                 num_iter_global=10, num_iter_source=10, num_iter_lens=10, num_iter_weights=2, 
                 lens_light_model_map=None, **base_kwargs):
        """
        :param data_class: lenstronomy.imaging_data.ImageData instance describing the data.
        :param lens_model_class: lenstronomy.lens_model.LensModel instance describing the lens mass model.
        :param image_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for image plane.
        :param source_numerics_class: lenstronomy.ImSim.Numerics.numerics_subframe.NumericsSubFrame instance for source plane.
        :param source_model_class: lenstronomy.light_model.LightModel instance describing the source light.
        :param lens_light_model_class: lenstronomy.light_model.LightModel instance describing the lens light.
        :param num_iter_global: number of iterations to alternate between source and lens light. 
        :param num_iter_source: number of iterations for sparse optimization of the source light. 
        :param num_iter_weights: number of iterations for l1-norm re-weighting scheme.
        :param base_kwargs: keyword arguments for SparseSolverBase.

        If not set or set to None, 'threshold_decrease_type' in base_kwargs defaults to 'linear'.
        """
        super(SparseSolverSourceLens, self).__init__(data_class, image_numerics_class, 
                                                     source_numerics_class, 
                                                     lens_model_class=lens_model_class,
                                                     **base_kwargs)

        self.add_source_light(source_model_class)
        self.add_lens_light(lens_light_model_class)

        # define default threshold decrease strategy
        if 'threshold_decrease_type' not in base_kwargs:
            self._threshold_decrease_type = 'linear'

        self._n_iter_source = num_iter_source
        self._n_iter_lens = num_iter_lens
        self._n_iter_global = num_iter_global
        if self._sparsity_prior_norm == 1:
            self._n_iter_weights = num_iter_weights
        else:
            self._n_iter_weights = 1   # reweighting scheme only defined for l1-norm sparsity
        
    def _ready(self):
        return not self.no_source_light and not self.no_lens_light

    def _solve(self, kwargs_lens=None, kwargs_ps=None, kwargs_special=None):
        """
        implements the SLIT_MCA algorithm
        """
        # set the gradient steps: 0 < mu < 2/spectral_norm
        mu_s = 1. / self.spectral_norm_source
        mu_l = 1. / self.spectral_norm_lens

        # generate initial guesses
        S, alpha_S = self.generate_initial_source()
        if self._init_lens_light_model is None:
            HG, alpha_HG = self.generate_initial_lens_light()
        else:
            # a guess for lens light has been provided
            if self._verbose:
                print("SparseSolverSourceLens: Using the provided lens light model as initial guess for sparse modelling")
            HG = self._init_lens_light_model
            alpha_HG = self.Phi_T_l(HG)

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
            model = self.Y_p if j == 0 else self.model_analysis(S=S, HG=HG)
            thresh_init = self._estimate_threshold_MOM(model)  # first estimation from data itself
            thresh = thresh_init
            # some hidden variables carried along the loop for threshold update
            thresh_init_adapt, n_iter_adapt = thresh_init, self._n_iter_global

            ######### Loop over source and lens light at *fixed weights* ########

            for i in range(self._n_iter_global):

                # get the proximal operators with current threshold and weights
                prox_g_s = lambda x, y: self.proximal_sparsity_source(x, threshold=thresh, weights=weights_source)
                prox_g_l = lambda x, y: self.proximal_sparsity_lens(x, threshold=thresh, weights=weights_lens)

                ######### Loop over source light at fixed weights ########

                # subtract lens light from data
                self.subtract_lens_from_data(HG)

                # get the gradient of the cost function f = || Y - HFS - HG ||^2_2  with respect to S
                grad_f_s = lambda x: self.gradient_loss_source(x)

                # initial hidden variables for the source
                if self.algorithm == 'FISTA':
                    fista_xi_s, fista_t_s = np.copy(alpha_S), 1.

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
                                       print_bool=(i % 10 == 0 and i_s % 10 == 0),
                                       iteration_text="{:03}-{:03}-{:03}".format(j, i, i_s))
                    
                    # update current estimate of source light and local parameters
                    S = S_next
                    alpha_S = alpha_S_next
                    if self.algorithm == 'FISTA':
                        fista_xi_s, fista_t_s = fista_xi_s_next, fista_t_s_next

                ######### ######## end source light ######## ########

                if self._show_steps: # and i % ma.ceil(self._n_iter_global/2) == 0:
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i, iter_3=i_s)
                    #self._plotter.plot_step(HG_next, iter_1=j, iter_2=i, iter_3=i_s)

                ######### Loop over lens light at fixed weights ########

                # subtract source light (lensed and convolved) from data
                self.subtract_source_from_data(S)

                # get the gradient of the cost function f = || Y - HFS - HG ||^2_2  with respect to HG
                grad_f_l = lambda x: self.gradient_loss_lens(x)

                # initial hidden variables for the lens
                if self.algorithm == 'FISTA':
                    fista_xi_l, fista_t_l = np.copy(alpha_HG), 1.

                for i_l in range(self._n_iter_lens):

                    if self.algorithm == 'FISTA':
                        alpha_HG_next, fista_xi_l_next, fista_t_l_next \
                            = algorithms.step_FISTA(alpha_HG, fista_xi_l, fista_t_l, grad_f_l, prox_g_l, mu_l)
                        HG_next = self.Phi_l(alpha_HG_next)

                    elif self.algorithm == 'FB':
                        HG_next = algorithms.step_FB(HG, grad_f_l, prox_g_l, mu_l)
                        alpha_HG_next = self.Phi_T_l(HG_next)

                    # save current step to track
                    self._tracker.save(HG=HG, HG_next=HG_next, 
                                       print_bool=(i % 10 == 0 and i_l % 10 == 0),
                                       iteration_text = "{:03}-{:03}-{:03}".format(j, i, i_l))

                    # update current estimate of source light and local parameters
                    HG = HG_next
                    alpha_HG = alpha_HG_next
                    if self.algorithm == 'FISTA':
                        fista_xi_l, fista_t_l = fista_xi_l_next, fista_t_l_next

                ######### ######## end lens light ######## ########

                if self._show_steps: # and i % ma.ceil(self._n_iter_global/2) == 0:
                    #self._plotter.plot_step(S_next, iter_1=j, iter_2=i, iter_3=i_s)
                    self._plotter.plot_step(HG_next, iter_1=j, iter_2=i, iter_3=i_l)

                # update adaptive threshold
                DS = self.Y_eff - self.H(self.R(self.F(S)))  # image with lensed convolved source subtracted
                DG = self.Y_eff - self.R(HG)                 # image with convolved lens subtracted
                thresh_MOM = self._estimate_threshold_MOM(DS, DG)
                thresh_dec = self._update_threshold(thresh, thresh_init_adapt, n_iter_adapt)
                # choose the minimum between the MOM estimation and the 'classic' decreasing one
                if thresh_MOM < thresh_dec:
                    thresh = thresh_MOM
                    # update hidden variables for next loop
                    thresh_init_adapt, n_iter_adapt = thresh_MOM, self._n_iter_global - i
                else:
                    thresh = thresh_dec

            ######### ######## end lens light ######## ########

            # update weights if necessary
            if self._n_iter_weights > 1:
                weights_source, weights_lens = self._update_weights(alpha_S, alpha_HG=alpha_HG)

        ######### ######## end weights ######## ########

        # reset data to original data
        self.reset_partial_data()

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
