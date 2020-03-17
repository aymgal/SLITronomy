__author__ = 'aymgal'

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import copy

from slitronomy.Lensing.lensing_planes import AbstractPlaneGrid, ImagePlaneGrid, SourcePlaneGrid
from slitronomy.Lensing.lensing_operator import LensingOperator
from slitronomy.Util import util

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.util as l_util


class TestAbstractPlaneGrid(object):
    """
    tests the Lensing Operator class
    """
    def setup(self):
        self.num_pix = 25  # cutout pixel size
        self.delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=self.delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix))
        }
        data_class = ImageData(**kwargs_data)
        self.grid = AbstractPlaneGrid(data_class)

    def test_properties(self):
        assert self.grid.delta_pix == self.delta_pix
        assert self.grid.num_pix == self.num_pix
        assert self.grid.grid_size == self.num_pix**2
        assert self.grid.grid_shape == (self.num_pix, self.num_pix)
        assert self.grid.shrinked is False  # by default


class TestImagePlaneGrid(object):
    """
    tests the Lensing Operator class
    """
    def setup(self):
        self.num_pix = 25  # cutout pixel size
        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix))
        }
        data_class = ImageData(**kwargs_data)
        self.image_plane = ImagePlaneGrid(data_class)


    def test_grid(self):
        grid_x_1d, grid_y_1d = self.image_plane.grid(two_dim=False)
        assert len(grid_x_1d.shape) == 1 and len(grid_y_1d.shape) == 1
        grid_x_2d, grid_y_2d = self.image_plane.grid(two_dim=True)
        assert len(grid_x_2d.shape) == 2 and len(grid_y_2d.shape) == 2

    def test_grid_pixels(self):
        grid_x_1d, grid_y_1d = self.image_plane.grid_pixels(two_dim=False)
        assert len(grid_x_1d.shape) == 1 and len(grid_y_1d.shape) == 1
        grid_x_2d, grid_y_2d = self.image_plane.grid_pixels(two_dim=True)
        assert len(grid_x_2d.shape) == 2 and len(grid_y_2d.shape) == 2

class TestSourcePlaneGrid(object):
    """
    tests the Lensing Operator class
    """
    def setup(self):
        self.num_pix = 25  # cutout pixel size
        self.subgrid_res_source = 2
        delta_pix = 0.32
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix))
        }
        data_class = ImageData(**kwargs_data)
        self.source_plane = SourcePlaneGrid(data_class, self.subgrid_res_source)

        # create a mask mimicking the real case of lensing operation
        lens_model_class = LensModel(['SIE'])
        kwargs_lens = [{'theta_E': 1.5, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1}]
        lensing_op = LensingOperator(data_class, lens_model_class, subgrid_res_source=self.subgrid_res_source)
        lensing_op.update_mapping(kwargs_lens)
        unit_image = np.ones((self.num_pix, self.num_pix))
        mask_image = np.zeros((self.num_pix, self.num_pix))
        mask_image[2:-2, 2:-2] = 1  # some binary image that mask out borders
        self.unit_image_mapped = lensing_op.image2source_2d(unit_image, no_flux_norm=False)
        self.mask_mapped = lensing_op.image2source_2d(mask_image)

    def test_grid(self):
        grid_x_1d, grid_y_1d = self.source_plane.grid(two_dim=False)
        assert len(grid_x_1d.shape) == 1 and len(grid_y_1d.shape) == 1
        grid_x_2d, grid_y_2d = self.source_plane.grid(two_dim=True)
        assert len(grid_x_2d.shape) == 2 and len(grid_y_2d.shape) == 2

    def test_grid_pixels(self):
        grid_x_1d, grid_y_1d = self.source_plane.grid_pixels(two_dim=False)
        assert len(grid_x_1d.shape) == 1 and len(grid_y_1d.shape) == 1
        grid_x_2d, grid_y_2d = self.source_plane.grid_pixels(two_dim=True)
        assert len(grid_x_2d.shape) == 2 and len(grid_y_2d.shape) == 2

    def test_effective_mask_default(self):
        mask = self.source_plane.effective_mask
        unit = np.ones((self.num_pix*self.subgrid_res_source, self.num_pix*self.subgrid_res_source))
        npt.assert_equal(mask, unit)
    
    def test_effective_mask_with_lensing(self):
        # add mapped mask to setup the 'effective_mask'
        self.source_plane.add_delensed_masks(self.unit_image_mapped, mapped_mask=self.mask_mapped)
        mask = self.source_plane.effective_mask
        unit = np.ones((self.num_pix*self.subgrid_res_source, self.num_pix*self.subgrid_res_source))
        npt.assert_raises(AssertionError, npt.assert_equal, mask, unit)  # test non-equality

    def test_shrink_grid_and_reset(self):
        self.source_plane.add_delensed_masks(self.unit_image_mapped)
        self.source_plane.shrink_grid_to_mask(min_num_pix=None)
        assert self.source_plane.num_pix < self.num_pix*self.subgrid_res_source

        self.source_plane.reset_grid()
        self.source_plane.add_delensed_masks(self.unit_image_mapped)
        min_num_pix = 45
        self.source_plane.shrink_grid_to_mask(min_num_pix=min_num_pix)
        assert self.source_plane.num_pix in [min_num_pix-1, min_num_pix, min_num_pix+1] 

    def test_update_subgrid_resolution(self):
        num_pix = 25  # cutout pixel size
        delta_pix = 0.32
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                         inverse=False, left_lower=False)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((num_pix, num_pix))
        }
        data_class = ImageData(**kwargs_data)
        source_plane = SourcePlaneGrid(data_class, subgrid_res=1)
        assert source_plane.subgrid_resolution == 1
        assert source_plane.grid_shape == (num_pix, num_pix)
        source_plane.subgrid_resolution = 3
        assert source_plane.subgrid_resolution == 3
        assert source_plane.grid_shape == (num_pix*3, num_pix*3)


class TestRaise(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRaise, self).__init__(*args, **kwargs)
        self.num_pix = 25
        self.num_pix_x, self.num_pix_y = 25, 30
        delta_pix = 0.24
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix_x, deltapix=delta_pix, subgrid_res=1, 
                                                   inverse=False, left_lower=False)
        kwargs_data_nonsquare = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix_x, self.num_pix_y))
        }
        self.data_class_nonsquare = ImageData(**kwargs_data_nonsquare)
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix))
        }
        self.data_class = ImageData(**kwargs_data)

    def test_raise(self):
        with self.assertRaises(ValueError):
            # testing non-square image -> should raise an error
            grid = AbstractPlaneGrid(self.data_class_nonsquare)
        with self.assertRaises(ValueError):
            grid = AbstractPlaneGrid(self.data_class)
            # only inheriting classes have access to grid coordinates -> raise an error
            theta_x = grid.theta_x
        with self.assertRaises(ValueError):
            grid = AbstractPlaneGrid(self.data_class)
            # only inheriting classes have access to grid coordinates -> raise an error
            theta_y = grid.theta_y

if __name__ == '__main__':
    pytest.main()
