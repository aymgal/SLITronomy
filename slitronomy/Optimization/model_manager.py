__author__ = 'aymgal'


import numpy as np

from slitronomy.Util import util



class ModelManager(object):

    """Utility class for initializing model operators and managing model components"""

    def __init__(self, data_class, lensing_operator_class, numerics_class, 
                 thread_count=1, random_seed=None):
        self._data_class = data_class
        self._lensing_op = lensing_operator_class
        self._numerics_class = numerics_class
        self._ss_factor = self._numerics_class.grid_supersampling_factor
        self._conv = self._numerics_class.convolution_class
        if self._conv is not None:
            try:
                self._conv_transpose = self._conv.copy_transpose()
            except NotImplementedError:
                raise ValueError("Convolution class does not supported copy_transpose()")
        else:
            self._conv_transpose = None
        self._no_source_light = True
        self._no_lens_light = True
        self._no_point_source = True
        self._ps_fixed = True
        self._thread_count = thread_count
        self.random_seed = random_seed
        self._prepare_data()

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

    def add_point_source(self, fix_model, filter_res, radius_regions, min_scale_regions,
                         check_filtering):
        self._no_point_source = False
        self._ps_fixed = fix_model
        self._ps_filter_residuals = filter_res
        self._ps_radius_regions = radius_regions
        self._ps_min_scale_regions = min_scale_regions
        self._ps_check = check_filtering

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

    def clean_masked_data(self, background_rms, init_ps_model=None):
        """Replace masked pixels with background noise
        This affects the ORIGINAL imaging data as well!
        """
        # fill masked pixels (excluded from the likelihood) with background noise
        mask = self.likelihood_mask
        if mask is not None:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            noise = background_rms * np.random.randn(*self._image_data_eff.shape)
            masked_pixels = np.where(mask == 0)
            self._image_data_eff[masked_pixels] = noise[masked_pixels]

        # fill point sources with a filtered version of the data
        ps_mask = self.point_source_mask(split=False)
        if ps_mask is not None:
            self._clean_masked_data_ps_residuals(ps_mask, init_ps_model)

        # make sure 'partial' is up-to-date
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
    def data_pixel_width(self):
        return self._data_class.pixel_width

    def data_coord2pix(self, ra, dec):
        return self._data_class.map_coord2pix(ra, dec)

    @property
    def num_pix_image(self):
        if self._lensing_op is None:
            return len(self.image_data)
        return self._lensing_op.imagePlane.num_pix

    @property
    def num_pix_source(self):
        if self._lensing_op is None:
            return None
        return self._lensing_op.sourcePlane.num_pix

    @property
    def likelihood_mask(self):
        if not hasattr(self, '_mask'):
            return None
        return self._mask

    def point_source_mask(self, split=True):
        if not hasattr(self, '_ps_mask_list'):
            return None
        if split is True:
            return self._ps_mask_list
        else:
            if len(self._ps_mask_list) == 1:
                return self._ps_mask_list[0]
            else:
                ps_mask_union = np.sum(self._ps_mask_list, axis=0)
                ps_mask_union[ps_mask_union > 1] = 1
                ps_mask_union[ps_mask_union < 0] = 0
                return ps_mask_union

    def _set_point_source_mask(self, mask_list):
        self._ps_mask_list = mask_list

    def _set_likelihood_mask(self, mask):
        self._mask = mask
        self._mask_1d = util.image2array(mask)
        if self._lensing_op is not None:
            self._lensing_op.set_likelihood_mask(mask)

    def _prepare_data(self):
        num_pix_x, num_pix_y = self._data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._image_data = np.copy(self._data_class.data)
        self._image_data_eff = np.copy(self._image_data)
        self.reset_partial_data()

    def _clean_masked_data_ps_residuals(self, ps_mask, init_ps_model):
        if self._ps_check is True:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            ax = axes[0]
            ax.set_title("before filtering")
            im = ax.imshow(self._image_data_eff - init_ps_model, cmap='gist_stern')
            fig.colorbar(im, ax=ax)

        ps_pixels = np.where(ps_mask > 1e-5)  # no need to consider pixels that are below this value
        data_m_ps = self._image_data_eff - init_ps_model
        n_scales = int(np.log2(min(*data_m_ps.shape)))  # maximal number of scales
        starlet_coeffs = util.starlet_transorm(data_m_ps, n_scales)
        # remove lower scales
        data_m_ps_filtered = np.sum(starlet_coeffs[self._ps_min_scale_regions:], axis=0)
        # replace pixels according to the mask values values
        data_m_ps_new = data_m_ps_filtered*ps_mask + data_m_ps*(1-ps_mask)
        self._image_data_eff[ps_pixels] = (data_m_ps_new + init_ps_model)[ps_pixels]

        if self._ps_check is True:
            ax = axes[1]
            ax.set_title("after filtering")
            im = ax.imshow(self._image_data_eff - init_ps_model, cmap='gist_stern')
            fig.colorbar(im, ax=ax)
            ax = axes[2]
            ax.set_title("regions only")
            im = ax.imshow(data_m_ps_filtered*ps_mask, cmap='gist_stern')
            fig.colorbar(im, ax=ax)
            plt.show()
