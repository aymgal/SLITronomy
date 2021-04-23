__author__ = 'aymgal'


import numpy as np

from slitronomy.Util import util



class ModelManager(object):

    """Utility class for initializing model operators and managing model components"""

    def __init__(self, data_class, lensing_operator_class, numerics_class, 
                 thread_count=1, random_seed=None):
        self._lensing_op = lensing_operator_class
        self._numerics_class = numerics_class
        self._ss_factor = numerics_class.grid_supersampling_factor
        self._conv = numerics_class.convolution_class
        if self._conv is not None:
            self._conv_transpose = self._conv.copy_transpose()
        else:
            self._conv_transpose = None
        self._prepare_data(data_class, self._lensing_op.source_subgrid_resolution)
        self._no_source_light = True
        self._no_lens_light = True
        self._no_point_source = True
        self._ps_fixed = True
        self._thread_count = thread_count
        self.random_seed = random_seed

        # TEMP: for PS mask generations
        self._data_class = data_class

    def add_source_light(self, source_model_class):
        # takes the first source light profile in the model list
        self._source_light_profile = source_model_class.func_list[0]
        if hasattr(self._source_light_profile, 'thread_count'):
            self._source_light_profile.thread_count = self._thread_count
        self._no_source_light = False

    def add_lens_light(self, lens_light_model_class):
        # takes the first lens light profile in the model list
        self._lens_light_profile = lens_light_model_class.func_list[0]
        if hasattr(self._lens_light_profile, 'thread_count'):
            self._lens_light_profile.thread_count = self._thread_count
        self._no_lens_light = False

    def add_point_source(self, fix_model=False):
        self._no_point_source = False
        self._ps_solver = None
        self._ps_fixed = fix_model

    def set_source_wavelet_scales(self, n_scales_source):
        self._n_scales_source = n_scales_source

    def set_lens_wavelet_scales(self, n_scales_lens):
        self._n_scales_lens_light = n_scales_lens

    def set_point_source_solver_func(self, point_source_solver_func):
        self._ps_solver = point_source_solver_func

    def set_point_source_error_func(self, point_source_error_func):
        self._ps_error = point_source_error_func

    @property
    def n_scales_source(self):
        if not hasattr(self, '_n_scales_source'):
            return None
        return self._n_scales_source

    @property
    def n_scales_lens_light(self):
        if not hasattr(self, '_n_scales_lens_light'):
            return None
        return self._n_scales_lens_light

    def subtract_from_data(self, array_2d):
        """Update 'partial' data by subtracting the input array"""
        self._image_data_part = self._image_data_eff - array_2d

    def reset_partial_data(self):
        """cancel any previous call to self.subtract_from_data()"""
        self._image_data_part = np.copy(self._image_data_eff)

    def fill_masked_data(self, background_rms, ps_mask=None, init_ps_model=None):
        """Replace masked pixels with background noise
        This affects the ORIGINAL imaging data as well!
        """
        if not hasattr(self, '_mask'):
            raise ValueError("No likelihood mask has been setup")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        noise = background_rms * np.random.randn(*self._image_data.shape)
        masked_pixels = np.where(self._mask == 0)
        self._image_data_eff[masked_pixels] = noise[masked_pixels]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self._image_data_eff - init_ps_model, cmap='gist_stern')
        # plt.colorbar()
        # plt.show()

        # WIP
        if ps_mask is not None:
            # compute the median value of the data with point source removed
            # from pixel valus inside the point source regions
            ps_pixels = np.where(ps_mask == 0)
            data_minus_ps = self._image_data_eff - init_ps_model
            med = np.nanmedian(data_minus_ps[ps_pixels])
            self._image_data_eff[ps_pixels] = med + init_ps_model[ps_pixels]

        # plt.figure()
        # plt.imshow(self._image_data_eff - init_ps_model, cmap='gist_stern')
        # plt.colorbar()
        # plt.show()

        # raise

        self.reset_partial_data()

    @property
    def image_data(self):
        """Original input data (no pre-processed)"""
        return self._image_data

    @property
    def effective_image_data(self):
        """Input data (possibly pre-processed) data"""
        return self._image_data_eff

    @property
    def partial_image_data(self):
        """
        Most of times contain effective_image_data minus a model component (lens, source).
        After a call reset_partial_data(), this is the same effective_image_data
        """
        return self._image_data_part

    @property
    def lensingOperator(self):
        return self._lensing_op

    @property
    def no_source_light(self):
        return self._no_source_light

    @property
    def no_lens_light(self):
        return self._no_lens_light

    @property
    def no_point_source(self):
        return self._no_point_source

    @property
    def fixed_point_source_model(self):
        return self._ps_fixed

    @property
    def num_pix_image(self):
        return self._lensing_op.imagePlane.num_pix

    @property
    def num_pix_source(self):
        return self._lensing_op.sourcePlane.num_pix

    def _set_likelihood_mask(self, mask):
        self._mask = mask
        self._mask_1d = util.image2array(mask)
        self._lensing_op.set_likelihood_mask(mask)

    def _set_point_source_mask(self, mask):
        self._ps_mask = mask

    def _prepare_data(self, data_class, subgrid_res_source):
        num_pix_x, num_pix_y = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._num_pix_source = int(num_pix_x * subgrid_res_source)
        self._image_data = np.copy(data_class.data)
        self._image_data_eff = np.copy(data_class.data)
        self.reset_partial_data()
