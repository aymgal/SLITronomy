__author__ = 'aymgal'

from slitronomy.Util import mask_util

import numpy as np
import numpy.testing as npt
import pytest
import unittest

np.random.seed(18)


def test_get_point_source_mask():
    mask_shape = (10, 10)
    delta_pix = 1
    ra_list = [2, 5]
    dec_list = [4, 8]
    radius = 3

    # mask as a unique array
    mask = mask_util.get_point_source_mask(mask_shape, delta_pix, dec_list, ra_list, radius, 
                                           split_masks=False)
    assert isinstance(mask, np.ndarray)
    npt.assert_equal(mask, mask.astype(bool))  # test it's only 0s and 1s

    mask_list = mask_util.get_point_source_mask(mask_shape, delta_pix, dec_list, ra_list, radius)
    assert isinstance(mask_list, list)
    assert len(mask_list) == len(ra_list)
    npt.assert_equal(mask_list[0], mask_list[0].astype(bool))  # test it's only 0s and 1s
    npt.assert_equal(mask_list[1], mask_list[1].astype(bool))  # test it's only 0s and 1s

    mask_list_smoothed = mask_util.get_point_source_mask(mask_shape, delta_pix, dec_list, ra_list, radius, 
                                                         smoothed=True)
    assert isinstance(mask_list_smoothed, list)
    assert len(mask_list_smoothed) == len(ra_list)
    npt.assert_raises(AssertionError, npt.assert_equal, mask_list_smoothed[0], mask_list_smoothed[0].astype(bool))
    npt.assert_raises(AssertionError, npt.assert_equal, mask_list_smoothed[1], mask_list_smoothed[1].astype(bool))


def test_get_point_source_mask():
    mask_shape = (10, 10)
    delta_pix = 1
    center_list = [None, (4, 8), (1, 1)]
    margin = 3
    radius_list = [3, 2, 3.5]
    axis_ratio_list = [0.8, 0.9, 0.4]
    angle_list = [0.8, 0.9, 0.66]
    operation_list = ['union', 'inter', 'subtract']
    inverted_list = [True, False, False]
    kwargs_square = {
        'mask_type': 'square',
        'margin': margin,
    }
    kwargs_circle = {
        'mask_type': 'circle',
        'radius_list': radius_list,
        'center_list': center_list,
        'operation_list': operation_list,
        'inverted_list': inverted_list,
    }
    kwargs_ellipse = {
        'mask_type': 'ellipse',
        'radius_list': radius_list,
        'center_list': center_list,
        'axis_ratio_list': axis_ratio_list,
        'angle_list': angle_list,
    }
    mask_cls_sq = mask_util.ImageMask(mask_shape, delta_pix, **kwargs_square)
    mask_sq = mask_cls_sq.get_mask()
    npt.assert_equal(mask_sq, mask_sq.astype(bool))  # test it's only 0s and 1s
    assert mask_sq[0, 0] == 0
    assert mask_sq[5, 5] == 1

    mask_cls_c  = mask_util.ImageMask(mask_shape, delta_pix, **kwargs_circle)
    mask_c = mask_cls_c.get_mask()
    mask_c_inv = mask_cls_c.get_mask(inverted=True)
    npt.assert_equal(mask_c, 1 - mask_c_inv)

    mask_cls_e  = mask_util.ImageMask(mask_shape, delta_pix, **kwargs_ellipse)
    mask_e_smo = mask_cls_e.get_mask(smoothed=True)
    mask_e_bool = mask_cls_e.get_mask(convert_to_bool=True)
    assert mask_e_bool.dtype == bool

    # extreme case
    mask_cls_big_sq = mask_util.ImageMask(mask_shape, delta_pix, mask_type='square', margin=100)
    mask = mask_cls_big_sq.get_mask()
    npt.assert_equal(mask, np.zeros(mask_shape))
    mask_cls_big_c = mask_util.ImageMask(mask_shape, delta_pix, mask_type='circle', radius_list=[100], center_list=[None])
    mask = mask_cls_big_c.get_mask(inverted=True)
    npt.assert_equal(mask, np.zeros(mask_shape))



class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            mask_shape = (10,)
            delta_pix = 1
            mask_class = mask_util.ImageMask(mask_shape, delta_pix)
        

if __name__ == '__main__':
    pytest.main()
