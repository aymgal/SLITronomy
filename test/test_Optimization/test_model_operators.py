__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Optimization.model_operators import ModelOperators
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Util import util

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
import lenstronomy.Util.util as l_util


class TestModelOperators(object):
    """
    tests the Lensing Operator classes
    """
    def setup(self):
        self.num_pix = 25  # cutout pixel size
        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        kwargs_data = {
            #'background_rms': background_rms,
            #'exposure_time': np.ones((self.num_pix, self.num_pix)) * exp_time,  # individual exposure time/weight per pixel
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix))
        }
        self.data = ImageData(**kwargs_data)

        self.lens_model = LensModel(['SPEP'])
        kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

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

        # define some mask
        self.likelihood_mask = np.ones((self.num_pix, self.num_pix))

        # get a lensing operator
        lensing_op = LensingOperator(self.data, self.lens_model, matrix_prod=True)
        lensing_op.update_mapping(kwargs_lens)

    def test_(self):
        #TODO
        pass

# class TestRaise(unittest.TestCase):
#     def test_raise(self):
#         with self.assertRaises(ValueError):
#             #TODO
#             pass

if __name__ == '__main__':
    pytest.main()
