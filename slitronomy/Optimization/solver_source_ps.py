__author__ = 'aymgal'

# class that implements SLIT algorithm

import copy
import numpy as np
import math as ma

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization import algorithms
from slitronomy.Util import util


class SparseSolverSourcePS(SparseSolverSource):

    """Implements the original SLIT algorithm with point source support"""

    def __init__(self, data_class, lens_model_class, image_numerics_class, source_numerics_class, source_model_class,
                 num_iter_source=10, num_iter_global=10, num_iter_weights=3, **base_kwargs):

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
        :param base_kwargs: keyword arguments for SparseSolverBase.
        
        If not set or set to None, 'threshold_decrease_type' in base_kwargs defaults to 'exponential'.
        """
        # remove settings not related to this solver
        _ = base_kwargs.pop('num_iter_lens', None)

        # define default threshold decrease strategy
        if base_kwargs.get('threshold_decrease_type', None) is None:
            base_kwargs['threshold_decrease_type'] = 'exponential'
            
        super(SparseSolverSourcePS, self).__init__(data_class, lens_model_class, image_numerics_class, source_numerics_class, source_model_class,
                                                   num_iter_source=num_iter_source, num_iter_weights=num_iter_weights, 
                                                   **base_kwargs)
        self.add_point_source()
        self._n_iter_global = num_iter_global

    def _ready(self):
        return not self.no_source_light and not self.no_point_source

    def _build_ps_mask(iter_index, kwargs_ps, kwargs_special):
        from TDLMCpipeline.Modelling.mask import ImageMask

        mask_shape = self._data_class.data.shape
        delta_pix = self._data_class.pixel_width

        max_radius = 0.5 # arcsec
        mask_radius = (self._n_iter_global - iter_index) / max_radius
        
        # translate the PS coordinates so origin is lower left
        ra_ps, dec_ps = kwargs_ps[0]['ra_image'], kwargs_ps[0]['dec_image']
        ra_ps_pix, dec_ps_pix = self._data_class.map_coord2pix(ra_ps, dec_ps)
        ra_ps, dec_ps = ra_ps_pix * delta_pix, dec_ps_pix * delta_pix

        #TODO: check implementation of this
        # if 'delta_x_image' in kwargs_special:
        #     ra_ps_pix += kwargs_special['delta_x_image']
        #     dec_ps_pix += kwargs_special['delta_y_image']

        mask_kwargs = {
            'mask_type': 'circle',
            'center_list': list(zip(ra_ps, dec_ps)),
            'radius_list': [mask_radius]*len(ra_ps_pix),
            'inverted_list': [True]*len(ra_ps_pix),
            'operation_list': ['inter']*(len(ra_ps_pix)-1),
        }
        mask_class = ImageMask(mask_shape=mask_shape, delta_pix=delta_pix, **mask_kwargs)
        return mask_class.get_mask(show_details=False)

    def _solve(self, kwargs_lens, kwargs_ps, kwargs_special):
        """
        implements the SLIT algorithm with point source support
        """
        if self._ps_solver is None:
            raise ValueError("No function has been provided for point source amplitude inversion")

        # set the gradient step
        mu = 1. / self.spectral_norm_source

        # get the gradient of the cost function, which is f = || Y - HFS ||^2_2
        grad_f = lambda x : self.gradient_loss_source(x)

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

                # subtract point sources from data
                self.subtract_point_source_from_data(P)

                # # update mask regions centered on point sources
                # ps_mask = self._build_ps_mask(i, kwargs_ps, kwargs_special)

                # estimate initial threshold after subtraction of point sources
                thresh_init = self._estimate_threshold_source(self.Y_eff)
                thresh = thresh_init

                # initial hidden variables
                if j == 0 and self.algorithm == 'FISTA':
                    fista_xi = np.copy(alpha_S)
                    fista_t  = 1.

                for i_s in range(self._n_iter_source):
                    # update adaptive threshold
                    thresh = self._update_threshold(thresh, thresh_init, self._n_iter_source)

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
                    self._tracker.save(S=S, S_next=S_next, 
                                       print_bool=(i % 10 == 0 and i_s % 10 == 0),
                                       iteration_text="{:03}-{:03}-{:03}".format(j, i, i_s))

                    if self._show_steps and (i_s % ma.ceil(self._n_iter_source/2) == 0):
                        self._plotter.plot_step(S_next, iter_1=j, iter_2=i, iter_3=i_s)

                    # update current estimate of source light and local parameters
                    S = S_next
                    alpha_S = alpha_S_next
                    if self.algorithm == 'FISTA':
                        fista_xi, fista_t = fista_xi_next, fista_t_next


                ######### ######## end source light ######## ########


                # subtract source light for point source linear amplitude optimization
                self.subtract_source_from_data(S)

                # solve for point source amplitudes
                effective_data_1d = util.image2array(self.Y_eff)[self._mask_1d]
                P, ps_error, ps_cov_param, ps_param = self._ps_solver(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps, 
                                                                      kwargs_special=kwargs_special, inv_bool=False,
                                                                      data_response_external=effective_data_1d)

                if self._show_steps and i % ma.ceil(self._n_iter_global/2) == 0 and i_s == self._n_iter_source-1:
                    self._plotter.plot_step(S_next, iter_1=j, iter_2=i, iter_3=i_s)

            ######### ######## end point source ######## ########

            # update weights if necessary
            if self._n_iter_weights > 1:
                weights, _ = self._update_weights(alpha_S)

        ######### ######## end weights ######## ########

        # reset effective data to original data
        self.reset_data()

        # store results
        self._tracker.finalize()
        self._source_model = S
        self._ps_model = P

        # all optimized coefficients (flattened)
        alpha_S_final = self.Phi_T_s(self.project_on_original_grid_source(S))
        coeffs_S_1d = util.cube2array(alpha_S_final)
        amps_P = ps_param

        if self._show_steps:
            self._plotter.plot_final(self._source_model)

        model = self.image_model(unconvolved=False)
        return model, coeffs_S_1d, [], amps_P
