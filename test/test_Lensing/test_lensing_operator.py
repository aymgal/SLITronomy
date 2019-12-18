__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt

from slitronomy.Lensing.lensing_operator import LensingOperator, LensingOperatorInterpol
from slitronomy.Util import util

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as l_util


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        num_pix = 25  # cutout pixel size
        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        kwargs_data = {
            #'background_rms': background_rms,
            #'exposure_time': np.ones((num_pix, num_pix)) * exp_time,  # individual exposure time/weight per pixel
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((num_pix, num_pix))
        }
        self.data = ImageData(**kwargs_data)

        self.lens_model = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

        # PSF specification
        #kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm, 'pixel_size': delta_pix, 'truncation': 11}
        kwargs_psf = {'psf_type': 'NONE'}
        psf = PSF(**kwargs_psf)

        # list of source light profiles
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_sersic_ellipse_source = {'amp': 2000, 'R_sersic': 0.6, 'n_sersic': 1, 'e1': 0.1, 'e2': 0.1,
                                        'center_x': 0.3, 'center_y': 0.3}
        kwargs_source = [kwargs_sersic_ellipse_source]
        source_model = LightModel(light_model_list=source_model_list)

        # list of lens light profiles
        lens_light_model_list = []
        kwargs_lens_light = [{}]
        lens_light_model = LightModel(light_model_list=lens_light_model_list)

        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        self.image_model = ImageModel(self.data, psf, self.lens_model, source_model, 
                                 lens_light_model, point_source_class=None, kwargs_numerics=kwargs_numerics)
        
        # create simulated image
        image_sim_no_noise = self.image_model.image(self.kwargs_lens, kwargs_source, kwargs_lens_light)
        self.source_light_lensed = image_sim_no_noise
        self.data.update_data(image_sim_no_noise)

        # source only, in source plane, on same grid as data
        self.source_light_delensed = self.image_model.source_surface_brightness(kwargs_source, unconvolved=False, de_lensed=True)

        # define some auto mask for tests
        self.likelihood_mask = np.zeros_like(self.source_light_lensed)
        self.likelihood_mask[self.source_light_lensed > 0.1 * self.source_light_lensed.max()] = 1

    def test_matrix_product(self):
        lensing_op = LensingOperator(self.data, self.lens_model, matrix_prod=False)
        lensing_op.update_mapping(self.kwargs_lens)

        lensing_op_mat = LensingOperator(self.data, self.lens_model, matrix_prod=True)
        lensing_op_mat.update_mapping(self.kwargs_lens)

        source_1d = util.image2array(self.source_light_delensed)
        image_1d = util.image2array(self.source_light_lensed)

        npt.assert_equal(lensing_op.source2image(source_1d), lensing_op_mat.source2image(source_1d))
        npt.assert_equal(lensing_op.image2source(image_1d), lensing_op_mat.image2source(image_1d))

    def test_minimal_source_plane(self):
        source_1d = util.image2array(self.source_light_delensed)

        # test with no mask
        lensing_op = LensingOperator(self.data, self.lens_model, matrix_prod=True, 
                                     likelihood_mask=None, minimal_source_plane=True)
        lensing_op.update_mapping(self.kwargs_lens)
        image_1d = util.image2array(self.source_light_lensed)
        assert lensing_op.image2source(image_1d).size < source_1d.size

        # test with mask
        lensing_op = LensingOperator(self.data, self.lens_model, matrix_prod=True, 
                                     likelihood_mask=self.likelihood_mask, minimal_source_plane=True)
        lensing_op.update_mapping(self.kwargs_lens)
        image_1d = util.image2array(self.source_light_lensed)
        assert lensing_op.image2source(image_1d).size < source_1d.size

        # for Interpol operator, only works with no mask (for now)
        lensing_op = LensingOperatorInterpol(self.data, self.lens_model,  
                                             likelihood_mask=None, minimal_source_plane=True)
        lensing_op.update_mapping(self.kwargs_lens)
        image_1d = util.image2array(self.source_light_lensed)
        assert lensing_op.image2source(image_1d).size < source_1d.size

    def test_interpol_mapping(self):
        lensing_op = LensingOperatorInterpol(self.data, self.lens_model)
        lensing_op.update_mapping(self.kwargs_lens)

        source_1d = util.image2array(self.source_light_delensed)
        image_1d = util.image2array(self.source_light_lensed)

        source_1d_lensed = lensing_op.source2image(source_1d)
        image_1d_delensed = lensing_op.image2source(image_1d)
        assert source_1d_lensed.shape == image_1d.shape
        assert image_1d_delensed.shape == source_1d.shape

        npt.assert_almost_equal(source_1d_lensed/source_1d_lensed.max(), image_1d/image_1d.max(), decimal=0.8)
        npt.assert_almost_equal(image_1d_delensed/image_1d_delensed.max(), source_1d/source_1d.max(), decimal=0.8)
    
    # def test_plot_neighbors_map(self):
    #     lensing_op = LensingOperatorInterpol(self.data, self.lens_model)
    #     fig = lensing_op.plot_neighbors_map(self.kwargs_lens, num_image_pixels=31)
        # import matplotlib.pyplot as plt
        # plt.show()

if __name__ == '__main__':
    pytest.main()
