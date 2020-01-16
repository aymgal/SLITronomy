__author__ = 'aymgal'


import numpy as np

from slitronomy.Util import util



class ModelOperators(object):

    """Utility class for access to operator as defined in formal optimization equations"""

    def __init__(self, data_class, lensing_operator_class, 
                 source_light_class, lens_light_class=None, convolution_class=None,
                 likelihood_mask=None):
        self._image_data = data_class.data
        self._image_data_eff = np.copy(data_class.data)
        if likelihood_mask is None:
            likelihood_mask = np.ones_like(self._image_data)
        self._mask = likelihood_mask
        self._mask_1d = util.image2array(likelihood_mask)
        self._source_light = source_light_class
        self._lens_light = lens_light_class
        self._lensing_op = lensing_operator_class
        self._conv = convolution_class
        if self._conv is not None:
            self._conv_transpose = convolution_class.copy_transpose()
        else:
            self._conv_transpose = None

    def set_wavelet_scales(self, n_scales_source, n_scales_lens=None):
        self._n_scales_source = n_scales_source
        self._n_scales_lens_light = n_scales_lens

    def subtract_from_data(self, array_2d):
        """Update "effective" data by subtracting the input array"""
        self._image_data_eff = self._image_data - array_2d

    def reset_data(self):
        """cancel any previous call to self.subtract_from_data()"""
        self._image_data_eff = self._image_data

    @property
    def spectral_norm_source(self):
        if not hasattr(self, '_spectral_norm_source'):
            def _operator(x):
                x = self.H_T(x)
                x = self.F_T(x)
                x = self.Phi_T_s(x)
                return x
            def _inverse_operator(x):
                x = self.Phi_s(x)
                x = self.F(x)
                x = self.H(x)
                return x
            self._spectral_norm_source = util.spectral_norm(self._num_pix, _operator, _inverse_operator,
                                                            num_iter=20, tol=1e-10)
        return self._spectral_norm_source

    @property
    def spectral_norm_lens(self):
        if not hasattr(self, '_spectral_norm_lens'):
            def _operator(x):
                x = self.Phi_T_l(x)
                return x
            def _inverse_operator(x):
                x = self.Phi_l(x)
                return x
            self._spectral_norm_lens = util.spectral_norm(self._num_pix, _operator, _inverse_operator,
                                                            num_iter=20, tol=1e-10)
        return self._spectral_norm_lens

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

    def Phi_s(self, array_2d):
        """alias method for inverse wavelet transform"""
        if not hasattr(self, '_n_scales_source'):
            raise ValueError("Wavelet scales have not been set")
        return self._source_light.function_2d(coeffs=array_2d, n_scales=self._n_scales_source,
                                              n_pixels=np.size(array_2d))

    def Phi_T_s(self, array_2d):
        """alias method for wavelet transform"""
        if not hasattr(self, '_n_scales_source'):
            raise ValueError("Wavelet scales have not been set")
        return self._source_light.decomposition_2d(image=array_2d, n_scales=self._n_scales_source)

    def Phi_l(self, array_2d):
        """alias method for inverse wavelet transform"""
        if self._lens_light is None:
            raise ValueError("Wavelet operator needs lens light class")
        if not hasattr(self, '_n_scales_lens_light'):
            raise ValueError("Wavelet scales have not been set")
        return self._lens_light.function_2d(coeffs=array_2d, n_scales=self._n_scales_lens_light,
                                            n_pixels=np.size(array_2d))

    def Phi_T_l(self, array_2d):
        """alias method for wavelet transform"""
        if self._lens_light is None:
            raise ValueError("Wavelet operator needs lens light class")
        if not hasattr(self, '_n_scales_lens_light'):
            raise ValueError("Wavelet scales have not been set")
        return self._lens_light.decomposition_2d(image=array_2d, n_scales=self._n_scales_lens_light)
