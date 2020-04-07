__author__ = 'aymgal'

import numpy as np
from scipy.ndimage import morphology

from slitronomy.Util import util


class PlaneGrid(object):

    """
    Base class for image and source plane grids, designed for pixelated lensing operator.
    """

    def __init__(self, num_pix, grid_class):
        """Initialise the grid.
        
        Parameters
        ----------
        image_grid_class : [lenstronomy.ImSim.Numerics.grid].RegularGrid or .AdaptiveGrid
            RegularGrid or .AdaptiveGrid instance
        """
        self._num_pix = num_pix
        self._grid = grid_class
        self._x_grid_1d, self._y_grid_1d = self._grid.coordinates_evaluate
        self._delta_pix = self._grid.pixel_width

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
        return self._x_grid_1d

    @property
    def theta_y(self):
        return self._y_grid_1d

    @property
    def unit_image(self):
        return np.ones(self.grid_shape)

    def grid(self, two_dim=False):
        if two_dim:
            return util.array2image(self.theta_x), util.array2image(self.theta_y)
        return self.theta_x, self.theta_y

    def grid_pixels(self, two_dim=False):
        theta_x_pix, theta_y_pix = self._grid.map_coord2pix(self.theta_x, self.theta_y)
        if two_dim:
            return util.array2image(theta_x_pix), util.array2image(theta_y_pix)
        return theta_x_pix, theta_y_pix


class SizeablePlaneGrid(PlaneGrid):

    """
    Class that defines the typical grid on which source galaxy is projected,
    whose size can be adapted with respect to image masks projected by a LensingOperator.
    """

    def __init__(self, num_pix, grid_class, subgrid_res=1, verbose=False):
        """Initialise SizeablePlaneGrid instance. 
        
        Parameters
        ----------
        grid_class : [lenstronomy.ImSim.Numerics.grid].RegularGrid or .AdaptiveGrid
            RegularGrid or .AdaptiveGrid instance
        subgrid_res : int, optional
            Source pixel size to image pixel size ratio
        verbose : bool, optional
            If False, print statements are shut down (e.g. when reducing iteratively grid size)
        """
        if not isinstance(subgrid_res, int):
            raise TypeError("'subgrid_res' must be an integer")
        super(SizeablePlaneGrid, self).__init__(num_pix, grid_class)
        self._num_pix *= subgrid_res  # update number of side pixels
        self._subgrid_res = subgrid_res
        self._first_print = True  # for printing messages only once
        self._verbose = verbose
        self._shrinked = False

    @property
    def shrinked(self):
        return self._shrinked

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

    def _fill_mapping_holes(self, image):
        """
        erosion operation for filling holes that may be introduced by pixelated lensing operations

        The higher the subgrid resolution of the source, the highest the number of holes.
        Hence the 'strength' of the erosion is set to the subgrid resolution (or round up integer) of the source plane
        """
        strength = int(np.ceil(self.subgrid_resolution))
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
