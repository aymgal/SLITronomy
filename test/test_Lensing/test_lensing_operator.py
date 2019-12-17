__author__ = 'aymgal'

import numpy as np

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as lenstro_util

from slitronomy.Lensing.lensing_operator import LensingOperator


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        num_pix = 25  # cutout pixel size
        delta_pix = 0.08
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = lenstro_util.make_grid_with_coordtransform(numPix=num_pix, deltapix=delta_pix, subgrid_res=1, 
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
        self.kwargs_lens = [{'theta_E': 0.5, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

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
        image_model = ImageModel(self.data, psf, self.lens_model, source_model, 
                                 lens_light_model, point_source_class=None, kwargs_numerics=kwargs_numerics)
        
        # create simulated image
        image_sim_no_noise = image_model.image(self.kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps=None)
        self.data.update_data(image_sim_no_noise)

    def test_init(self):
        self.lensing_op = LensingOperator(self.data, self.lens_model, subgrid_res_source=1, 
                                     minimal_source_plane=False, min_num_pix_source=10, 
                                     matrix_prod=False)
        self.lensing_op_matrix = LensingOperator(self.data, self.lens_model, subgrid_res_source=1, 
                                            minimal_source_plane=False, min_num_pix_source=10, 
                                            matrix_prod=True)
        self.lensing_op_subgrid2 = LensingOperator(self.data, self.lens_model, subgrid_res_source=1, 
                                              minimal_source_plane=False, min_num_pix_source=10, 
                                              matrix_prod=True)

    def test_mapping(self):
        

if __name__ == '__main__':
    pytest.main()
