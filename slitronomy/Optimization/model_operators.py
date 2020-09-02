__author__ = 'aymgal'


import numpy as np

from slitronomy.Optimization.model_manager import ModelManager
from slitronomy.Util import util


class ModelOperators(ModelManager):

    """Utility class for access to operator as defined in formal optimization equations"""

    def __init__(self, data_class, lensing_operator_class, numerics_class,
                 fixed_spectral_norm_source=None, thread_count=1, random_seed=None):
        super(ModelOperators, self).__init__(data_class, lensing_operator_class, numerics_class,
                                             thread_count=thread_count, random_seed=random_seed)
        self._fixed_spectral_norm_source = fixed_spectral_norm_source

    @property
    def Y(self):
        """
        Original imaging data.
        """
        return self._image_data

    @property
    def Y_eff(self):
        """
        "Effective" imaging data.
        This can be the entire imaging data, or an updated version of it with a component subtracted
        """
        return self._image_data_eff

    def M(self, image_2d):
        """Apply image plane mask"""
        return self._mask * image_2d

    def M_s(self, source_2d):
        """Apply source plane mask"""
        return self._lensing_op.sourcePlane.effective_mask * source_2d

    def H(self, array_2d):
        """alias method for convolution with the PSF kernel"""
        if self._conv is None:
            return array_2d
        return self._conv.convolution2d(array_2d)

    def H_T(self, array_2d):
        """alias method for convolution with the transposed PSF kernel"""
        if self._conv_transpose is None:
            return array_2d
        return self._conv_transpose.convolution2d(array_2d)

    def F(self, source_2d):
        """alias method for lensing from source plane to image plane"""
        return self._lensing_op.source2image_2d(source_2d)

    def F_T(self, image_2d):
        """alias method for ray-tracing from image plane to source plane"""
        return self._lensing_op.image2source_2d(image_2d)

    def R(self, image_2d):
        """alias for resize to lower resolution (DOWNsampling operation), from finer grid to imaging data grid"""
        #TODO
        return util.Downsample(image_2d, factor=self._ss_factor)

    def R_T(self, image_2d):
        """alias for resize to higher resolution (UPsampling operation), from finer grid to imaging data grid"""
        #TODO
        res = util.Upsample(image_2d, factor=self._ss_factor)
        return res

    def Phi_s(self, array_2d):
        """alias method for inverse wavelet transform"""
        if not hasattr(self, '_n_scales_source'):
            raise ValueError("Wavelet scales have not been set")
        return self._source_light.function_2d(coeffs=array_2d, n_scales=self._n_scales_source,
                                              n_pixels=array_2d.size)

    def Phi_T_s(self, array_2d):
        """alias method for wavelet transform"""
        if not hasattr(self, '_n_scales_source'):
            raise ValueError("Wavelet scales have not been set")
        return self._source_light.decomposition_2d(image=array_2d, n_scales=self._n_scales_source)

    def Phi_l(self, array_2d):
        """alias method for inverse wavelet transform"""
        if self.no_lens_light:
            raise ValueError("Wavelet operator needs lens light class")
        if not hasattr(self, '_n_scales_lens_light'):
            raise ValueError("Wavelet scales have not been set")
        return self._lens_light.function_2d(coeffs=array_2d, n_scales=self._n_scales_lens_light,
                                            n_pixels=array_2d.size)

    def Phi_T_l(self, array_2d):
        """alias method for wavelet transform"""
        if self.no_lens_light:
            raise ValueError("Wavelet operator needs lens light class")
        if not hasattr(self, '_n_scales_lens_light'):
            raise ValueError("Wavelet scales have not been set")
        return self._lens_light.decomposition_2d(image=array_2d, n_scales=self._n_scales_lens_light)

    @property
    def psf_kernel(self):
        if self._conv is None:
            return None
        return self._conv.pixel_kernel()

    @property
    def spectral_norm_source_is_fixed(self):
        return self._fixed_spectral_norm_source is not None

    @property
    def spectral_norm_source(self):
        if not hasattr(self, '_spectral_norm_source'):
            self.update_spectral_norm_source()
        return self._spectral_norm_source

    @property
    def spectral_norm_lens(self):
        if not hasattr(self, '_spectral_norm_lens'):
            self.update_spectral_norm_lens()
        return self._spectral_norm_lens

    def update_spectral_norm_source(self):
        if self.spectral_norm_source_is_fixed:
            self._spectral_norm_source = self._fixed_spectral_norm_source
        else:
            self._spectral_norm_source = self.compute_spectral_norm_source()

    def update_spectral_norm_lens(self):
        self._spectral_norm_lens = self.compute_spectral_norm_lens()

    def compute_spectral_norm_source(self):
        def _operator(x):
            x = self.H_T(x)
            x = self.R_T(x)
            x = self.F_T(x)
            x = self.Phi_T_s(x)
            return x
        def _inverse_operator(x):
            x = self.Phi_s(x)
            x = self.F(x)
            x = self.R(x)
            x = self.H(x)
            return x
        return util.spectral_norm(self._num_pix, _operator, _inverse_operator, num_iter=20, tol=1e-10, seed=self.random_seed)

    def compute_spectral_norm_lens(self):
        def _operator(x):
            x = self.R_T(x)
            x = self.Phi_T_l(x)
            return x
        def _inverse_operator(x):
            x = self.Phi_l(x)
            x = self.R(x)
            return x
        return util.spectral_norm(self._num_pix, _operator, _inverse_operator, num_iter=20, tol=1e-10)
