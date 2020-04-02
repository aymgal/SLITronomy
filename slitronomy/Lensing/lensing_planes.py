__author__ = 'aymgal'

import numpy as np
from scipy.ndimage import morphology

from slitronomy.Util import util


class AbstractPlaneGrid(object):

    """
    Base class for image and source plane grids

    TODO : use the lenstronomy's PixelGrid class instead
    """

    def __init__(self, data_class):
        self.data = data_class
        num_pix_x, num_pix_y = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._delta_pix = data_class.pixel_width
        self._shrinked = False

    @property
    def num_pix(self):
        return self._num_pix

    @property
    def grid_size(self):
        return self._num_pix**2

    @property
    def grid_shape(self):
        return (self._num_pix, self._num_pix)

    @property
    def delta_pix(self):
        return self._delta_pix

    @property
    def theta_x(self):
        if not hasattr(self, '_x_grid_1d'):
            raise ValueError("theta coordinates are not defined")
        return self._x_grid_1d

    @property
    def theta_y(self):
        if not hasattr(self, '_y_grid_1d'):
            raise ValueError("theta coordinates are not defined")
        return self._y_grid_1d

    @property
    def unit_image(self):
        return np.ones(self.grid_shape)

    def grid(self, two_dim=False):
        if two_dim:
            return util.array2image(self._x_grid_1d), util.array2image(self._y_grid_1d)
        return self._x_grid_1d, self._y_grid_1d

    def grid_pixels(self, two_dim=False):
        theta_x_pix, theta_y_pix = self.data.map_coord2pix(self.theta_x, self.theta_y)
        if two_dim:
            return util.array2image(theta_x_pix), util.array2image(theta_y_pix)
        return theta_x_pix, theta_y_pix

    @property
    def shrinked(self):
        return self._shrinked


class ImagePlaneGrid(AbstractPlaneGrid):

    """Class that defines the grid on which lens galaxy is projected"""

    def __init__(self, data_class):
        super(ImagePlaneGrid, self).__init__(data_class)
        # get the coordinates arrays of image plane
        x_grid, y_grid = data_class.pixel_coordinates
        self._x_grid_1d = util.image2array(x_grid)
        self._y_grid_1d = util.image2array(y_grid)


class SourcePlaneGrid(AbstractPlaneGrid):

    """Class that defines the grid on which source galaxy is projected"""

    # TODO : use lenstronomy's util.make_subgrid(), it will automatically align the center of source plane

    def __init__(self, data_class, subgrid_res=1, verbose=False):
        super(SourcePlaneGrid, self).__init__(data_class)
        self._subgrid_res = subgrid_res

        # adapt grid size and resolution
        self._num_pix *= int(subgrid_res)
        self._delta_pix /= float(subgrid_res)

        # get the coordinates arrays of source plane, with aligned origin
        self._x_grid_1d, self._y_grid_1d = util.make_grid(numPix=self._num_pix, deltapix=self._delta_pix)

        # WARNING : we assume that center of coordinates is at the center of the image !!
        # TODO : make sure that center is consistent > use RegularGrid class in lenstronomy, like in Numerics ??

        self._first_print = True  # for printing messages only once
        self._verbose = verbose

    @property
    def effective_mask(self):
        """
        Returns the intersection between the likelihood mask and the area on source plane
        that has corresponding pixels in image plane
        """
        if not hasattr(self, '_effective_mask'):
            print("Warning : lensed unit image in source plane has not been set, effective mask filled with 1s")
            self._effective_mask = np.ones(self.grid_shape)
        return self._effective_mask.astype(float)

    @property
    def reduction_mask(self):
        if not hasattr(self, '_reduc_mask_1d'):
            print("Warning : no reduction mask has been computed for grid shrinking")
            self._reduc_mask_1d = np.ones(self.grid_size, dtype=bool)
        return util.array2image(self._reduc_mask_1d.astype(float))

    def add_delensed_masks(self, mapped_image, mapped_mask=None):
        """input mapped_image and mask must be non-boolean 2d arrays"""
        image_refined = self._fill_mapping_holes(mapped_image).astype(bool)
        if mapped_mask is not None:
            mask_refined = self._fill_mapping_holes(mapped_mask).astype(bool)
            self._effective_mask = np.zeros(self.grid_shape, dtype=bool)
            # union of the two masks to get an "effective" mask
            self._effective_mask[image_refined & mask_refined] = True
        else:
            self._effective_mask = image_refined

    def shrink_grid_to_mask(self, min_num_pix=None):
        if min_num_pix is None:
            # kind of arbitrary as a default
            min_num_pix = int(self.num_pix / 10)
        if (self.effective_mask is None) or (self.num_pix <= min_num_pix):
            # if no mask to shrink to, or already shrunk, or already smaller than minimal allowed size
            return
        reduc_mask, reduced_num_pix = self.shrink_plane_iterative(self.effective_mask, min_num_pix=min_num_pix)
        self._update_grid_after_shrink(reduc_mask, reduced_num_pix)
        if self._first_print and self._verbose:
            print("INFO : source grid has been reduced from {} to {} side pixels".format(self._num_pix_large, self._num_pix))
            self._first_print = False

    def project_on_original_grid(self, image):
        if hasattr(self, '_num_pix_large'):
            input_is_1d = (len(image.shape) == 1)
            array_large = np.zeros(self._num_pix_large**2)
            if input_is_1d:
                array_large[self._reduc_mask_1d] = image[:]
                return array_large
            else:
                array_large[self._reduc_mask_1d] = util.image2array(image)[:]
                return util.array2image(array_large)
        else:
            return image

    def reset_grid(self):
        if self.shrinked:
            self._num_pix = self._num_pix_large
            self._x_grid_1d = self._x_grid_1d_large
            self._y_grid_1d = self._y_grid_1d_large
            self._effective_mask = self._effective_mask_large
            delattr(self, '_num_pix_large')
            delattr(self, '_x_grid_1d_large')
            delattr(self, '_y_grid_1d_large')
            delattr(self, '_effective_mask_large')
            self._shrinked = False

    @property
    def subgrid_resolution(self):
        return self._subgrid_res

    @subgrid_resolution.setter
    def subgrid_resolution(self, new_subgrid_res):
        """Update all required fields when setting a new subgrid resolution"""
        self.reset_grid()
        if hasattr(self, '_effective_mask'):
            print("Warning : reset effective_mask to only 1s")
            self._effective_mask = np.ones(self.grid_shape)
        self._subgrid_res = new_subgrid_res
        self._num_pix = self.data.num_pixel_axes[0] * self._subgrid_res
        self._delta_pix = self.data.pixel_width / self._subgrid_res
        self._x_grid_1d, self._y_grid_1d = util.make_grid(numPix=self._num_pix, deltapix=self._delta_pix)

    def _fill_mapping_holes(self, image):
        """
        erosion operation for filling holes that may be introduced by pixelated lensing operations

        The higher the subgrid resolution of the source, the highest the number of holes.
        Hence the 'strength' of the erosion is set to the subgrid resolution (or round up integer) of the source plane
        """
        strength = np.ceil(self._subgrid_res).astype(int)
        # invert 0s and 1s
        image = 1 - image
        # apply morphological erosion operation
        image = morphology.binary_erosion(image, iterations=strength).astype(int)
        # invert 1s and 0s
        image = 1 - image
        # re-apply erosion to remove the "dilation" effect of the previous erosion
        image = morphology.binary_erosion(image, iterations=strength).astype(int)
        return image

    def _update_grid_after_shrink(self, reduc_mask, reduced_num_pix):
        self._reduc_mask_1d = util.image2array(reduc_mask).astype(bool)
        # backup the original 'large' grid
        self._num_pix_large = self._num_pix
        self._x_grid_1d_large = np.copy(self._x_grid_1d)
        self._y_grid_1d_large = np.copy(self._y_grid_1d)
        self._effective_mask_large = np.copy(self.effective_mask)
        # update coordinates array
        self._num_pix = reduced_num_pix
        self._x_grid_1d = self._x_grid_1d[self._reduc_mask_1d]
        self._y_grid_1d = self._y_grid_1d[self._reduc_mask_1d]
        # don't know why, but can apply reduc_mask_1d only on 1D arrays
        effective_mask_1d = util.image2array(self._effective_mask)
        self._effective_mask = util.array2image(effective_mask_1d[self._reduc_mask_1d])
        self._shrinked = True

    @staticmethod
    def shrink_plane_iterative(effective_mask, min_num_pix=10):
        """
        :param min_num_pix: minimal allowed number of pixels in source plane
        """
        num_pix_origin = len(effective_mask)
        num_pix = num_pix_origin  # start at original size
        n_rm = 1
        test_mask = np.ones((num_pix, num_pix))
        reduc_mask = test_mask
        while num_pix > min_num_pix:
            # array full of zeros
            test_mask_next = np.zeros_like(effective_mask)
            # fill with ones to create a centered square with size reduced by 2*n_rm
            test_mask_next[n_rm:-n_rm, n_rm:-n_rm] = 1
            # update number side length of the non-zero
            num_pix_next = num_pix_origin - 2 * n_rm
            # test if all ones in test_mask are also ones in target mask
            intersection_mask = np.zeros_like(test_mask_next)
            intersection_mask[(test_mask_next == 1) & (effective_mask == 1)] = 1
            is_too_large_mask = np.all(intersection_mask == effective_mask)
            if is_too_large_mask:
                # if the intersection is equal to the original array mask, this means that we can try a smaller mask
                num_pix = num_pix_next
                test_mask = test_mask_next
                n_rm += 1
            else:
                # if not, then the mask at previous iteration was the correct one
                break
        reduc_mask = test_mask
        red_num_pix = num_pix
        return reduc_mask.astype(bool), red_num_pix
