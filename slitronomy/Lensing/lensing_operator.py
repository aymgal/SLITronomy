__author__ = 'aymgal'

import numpy as np
from scipy import sparse

from slitronomy.Lensing.lensing_planes import ImagePlaneGrid, SourcePlaneGrid
from slitronomy.Util import util



class LensingOperator(object):

    """Defines the mapping of pixelated light profiles between image and source planes"""

    def __init__(self, data_class, lens_model_class, subgrid_res_source=1, 
                 likelihood_mask=None, minimal_source_plane=False, min_num_pix_source=10,
                 fix_minimal_source_plane=True, use_mask_for_minimal_source_plane=True,
                 matrix_prod=True):
        """

        :param min_num_pix_source: minimal number of pixels in the 
        """
        self.lensModel = lens_model_class
        self.imagePlane  = ImagePlaneGrid(data_class)
        self.sourcePlane = SourcePlaneGrid(data_class, subgrid_res=subgrid_res_source)
        self._likelihood_mask = likelihood_mask
        self._minimal_source_plane = minimal_source_plane
        self._fix_minimal_source_plane = fix_minimal_source_plane
        self._use_mask_for_minimal_source_plane = use_mask_for_minimal_source_plane
        self._min_num_pix_source = min_num_pix_source
        self._matrix_prod = matrix_prod

    def source2image(self, source_1d, kwargs_lens=None, update=False):
        if (not hasattr(self, '_lens_mapping') or update) and kwargs_lens is not None:
            self.update_mapping(kwargs_lens)
        if self._matrix_prod:
            return self._source2image_matrix(source_1d)
        else:
            return self._source2image_list(source_1d)

    def source2image_2d(self, source, **kwargs):
        source_1d = util.image2array(source)
        return util.array2image(self.source2image(source_1d, **kwargs))

    def _source2image_matrix(self, source_1d):
        image_1d = self._lens_mapping.dot(source_1d)
        return image_1d

    def _source2image_list(self, source_1d):
        image_1d = np.ones(self.imagePlane.grid_size)
        # loop over source plane pixels
        for j in range(source_1d.size):
            indices_i = np.where(self._lens_mapping == j)
            image_1d[indices_i] = source_1d[j]
        return image_1d

    def image2source(self, image_1d, kwargs_lens=None, update=False, test_unit_image=False):
        """if test_unit_image is True, do not normalize light flux to better visualize the mapping"""
        if (not hasattr(self, '_lens_mapping') or update) and kwargs_lens is not None:
            self.update_mapping(kwargs_lens)
        if self._matrix_prod:
            return self._image2source_matrix(image_1d, update=update, test_unit_image=test_unit_image)
        else:
            return self._image2source_list(image_1d, update=update, test_unit_image=test_unit_image)

    def image2source_2d(self, image, **kwargs):
        image_1d = util.image2array(image)
        return util.array2image(self.image2source(image_1d, **kwargs))

    def _image2source_matrix(self, image_1d, update=False, test_unit_image=False):
        if not hasattr(self, '_norm_image2source') or update:
            # avoid computing it each time, save normalisation factors
            # normalization is sum of weights for each source pixel
            self._norm_image2source = np.squeeze(np.maximum(1, self._lens_mapping.sum(axis=0)).A)
        source_1d = self._lens_mapping.T.dot(image_1d)
        if not test_unit_image:
            # normalization
            source_1d /= self._norm_image2source
        return source_1d

    def _image2source_list(self, image_1d, update=False, test_unit_image=False):
        source_1d = np.zeros(self.sourcePlane.grid_size)
        # loop over source plane pixels
        for j in range(source_1d.size):
            # retieve corresponding pixels in image plane
            indices_i = np.where(self._lens_mapping == j)
            flux_i = image_1d[indices_i]
            flux_j = np.sum(flux_i)
            if not test_unit_image:
                # normalization
                flux_j /= max(1, flux_i.size)
            source_1d[j] = flux_j
        return source_1d

    @property
    def source_plane_coordinates(self):
        return self.sourcePlane.theta_x, self.sourcePlane.theta_y

    @property
    def image_plane_coordinates(self):
        return self.imagePlane.theta_x, self.imagePlane.theta_y

    def update_mapping(self, kwargs_lens, kwargs_special=None):
        # reset source plane grid if it was altered by previous mass model
        if not self._fix_minimal_source_plane and self.sourcePlane.shrinked:
            self._reset_source_plane_grid()

        # compute mapping between image and source plances due to lensing
        self._compute_mapping(kwargs_lens, kwargs_special)

        # compute areas on source plane where you have no constrains
        self._compute_source_mask()
        
        if self._minimal_source_plane and not self.sourcePlane.shrinked:
            # for source plane to be reduced to minimal size
            # we compute effective source mask and shrink the grid to match it
            self._shrink_source_plane_grid()
        
            # recompute the mapping with updated grid
            self._compute_mapping(kwargs_lens, kwargs_special=kwargs_special)

        return (self.imagePlane.grid_size, self.imagePlane.delta_pix, 
                self.sourcePlane.grid_size, self.sourcePlane.delta_pix)

    def delete_mapping(self):
        if hasattr(self, '_lens_mapping'): delattr(self, '_lens_mapping')
        if hasattr(self, '_norm_image2source'): delattr(self, '_norm_image2source')

    def _compute_mapping(self, kwargs_lens, kwargs_special):
        """
        Core method that computes the mapping between image and source planes pixels
        from ray-tracing performed by the input parametric mass model
        """
        # delete previous mapping and init the new one
        self.delete_mapping()

        # initialize matrix
        if self._matrix_prod:
            lens_mapping_matrix = np.zeros((self.imagePlane.grid_size, self.sourcePlane.grid_size))
        else:
            lens_mapping_list = []

        # backward ray-tracing to get source coordinates in image plane (the 'betas')
        beta_x, beta_y = self.lensModel.ray_shooting(self.imagePlane.theta_x, self.imagePlane.theta_y, kwargs_lens)

        # get source plane offsets
        grid_offset_x, grid_offset_y = self._source_grid_offsets(kwargs_special)

        # iterate through indices of image plane (indices 'i')
        for i in range(self.imagePlane.grid_size):
            # find source pixel that corresponds to ray traced image pixel
            j = self._find_source_pixel(i, beta_x, beta_y, grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y)
            
            # fill the mapping array
            if self._matrix_prod:
                lens_mapping_matrix[i, j] = 1
            else:
                lens_mapping_list.append(j)
        
        if self._matrix_prod:
            # convert numpy array to sparse matrix, using Compressed Sparse Row (CSR) format for fast vector products
            self._lens_mapping = sparse.csr_matrix(lens_mapping_matrix)
        else:
            # convert the list to array 
            self._lens_mapping = np.array(lens_mapping_list)

    def _compute_source_mask(self):
        # de-lens a unit image it to get non-zero source plane pixel
        unit_image_mapped = self.image2source_2d(self.imagePlane.unit_image)
        unit_image_mapped[unit_image_mapped > 0] = 1
        if self._likelihood_mask is not None and self._use_mask_for_minimal_source_plane:
            # de-lens a unit image it to get non-zero source plane pixel
            mask_mapped = self.image2source_2d(self._likelihood_mask)
            mask_mapped[mask_mapped > 0] = 1
        else:
            mask_mapped = None
        # setup the image to source plane for filling holes due to pixelisation of the lensing operation
        self.sourcePlane.add_delensed_masks(unit_image_mapped, mapped_mask=mask_mapped)

    def _reset_source_plane_grid(self):
        self.sourcePlane.reset_grid()

    def _shrink_source_plane_grid(self):
        self.sourcePlane.shrink_grid_to_mask(min_num_pix=self._min_num_pix_source)

    def _source_grid_offsets(self, kwargs_special):
        if kwargs_special is None: return 0, 0
        grid_offset_x = kwargs_special.get('delta_x_source_grid', 0)
        grid_offset_y = kwargs_special.get('delta_y_source_grid', 0)
        return grid_offset_x, grid_offset_y

    def _find_source_pixel(self, i, beta_x, beta_y, grid_offset_x=0, grid_offset_y=0):
        dist2_map = self._distance_to_source_grid(i, beta_x, beta_y, grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y, squared=True)
        # find the index that corresponds to the minimal distance (closest pixel)
        j = np.argmin(dist2_map)
        return j

    def _distance_to_source_grid(self, i, beta_x, beta_y, grid_offset_x=0, grid_offset_y=0, squared=False, pixel_conversion=False):
        # coordinate grid of source plane
        diff_x, diff_y = self._difference_on_source_grid_axis(i, beta_x, beta_y, grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y,
                                                              pixel_conversion=pixel_conversion)
        # compute the distance between ray-traced coordinate and source plane grid
        # (square of the distance, not required to apply sqrt operation)
        dist_squared = diff_x**2 + diff_y**2
        if squared:
            return dist_squared
        return np.sqrt(dist_squared)

    def _difference_on_source_grid_axis(self, i, beta_x_image, beta_y_image, grid_offset_x=0, grid_offset_y=0,
                                        absolute=False, pixel_conversion=False):
        # coordinate grid of source plane
        theta_x_source = self.sourcePlane.theta_x + grid_offset_x
        theta_y_source = self.sourcePlane.theta_y + grid_offset_y
        if pixel_conversion:
            num_pix = self.sourcePlane.num_pix
            delta_pix = self.sourcePlane.delta_pix
            theta_x_source = (theta_x_source + delta_pix*num_pix/2.) / delta_pix
            theta_y_source = (theta_y_source + delta_pix*num_pix/2.) / delta_pix
            beta_x_image_i = (beta_x_image[i] + delta_pix*num_pix/2.) / delta_pix
            beta_y_image_i = (beta_y_image[i] + delta_pix*num_pix/2.) / delta_pix
        else:
            beta_x_image_i = beta_x_image[i]
            beta_y_image_i = beta_y_image[i]
        # compute the difference between ray-traced coordinate and source plane grid
        dist_x = beta_x_image_i - theta_x_source
        dist_y = beta_y_image_i - theta_y_source
        if absolute:
            return np.abs(dist_x), np.abs(dist_y)
        return dist_x, dist_y

    def _index_1d_to_2d_source(self, j):
        if j is None:
            return (None, None)
        return util.index_1d_to_2d(j, self.sourcePlane.num_pix)

    def _index_2d_to_1d_source(self, x, y):
        if x is None or y is None:
            return None
        return util.index_2d_to_1d(x, y, self.sourcePlane.num_pix)


class LensingOperatorInterpol(LensingOperator):

    """
    Defines the mapping of pixelated light profiles between image and source planes

    Contrarily to LensingOperator, follows Treu & Koopmans 2004 to interpolate flux on source plane.
    """

    def __init__(self, data_class, lens_model_class, subgrid_res_source=1, 
                 likelihood_mask=None, minimal_source_plane=False, fix_minimal_source_plane=True, 
                 use_mask_for_minimal_source_plane=True, min_num_pix_source=10):
        _matrix_prod = True
        (super(LensingOperatorInterpol, self).__init__(data_class, lens_model_class, 
            subgrid_res_source=subgrid_res_source, likelihood_mask=likelihood_mask, 
            minimal_source_plane=minimal_source_plane, fix_minimal_source_plane=fix_minimal_source_plane, 
            min_num_pix_source=min_num_pix_source, use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
            matrix_prod=_matrix_prod))

    def _compute_mapping(self, kwargs_lens, kwargs_special):
        """
        Core method that computes the mapping between image and source planes pixels
        from ray-tracing performed by the input parametric mass model
        """
        # delete previous mapping and init the new one
        self.delete_mapping()
        lens_mapping_matrix = np.zeros((self.imagePlane.grid_size, self.sourcePlane.grid_size))

        # backward ray-tracing to get source coordinates in image plane (the 'betas')
        beta_x, beta_y = self.lensModel.ray_shooting(self.imagePlane.theta_x, self.imagePlane.theta_y, kwargs_lens)

        # get source plane offsets
        grid_offset_x, grid_offset_y = self._source_grid_offsets(kwargs_special)

        # iterate through indices of image plane (indices 'i')
        for i in range(self.imagePlane.grid_size):
            # find source pixel that corresponds to ray traced image pixel
            j_list, _, dist_A_x, dist_A_y = self._find_surrounding_source_pixels(i, beta_x, beta_y, grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y)

            # get interpolation weights
            weight_list = self._bilinear_weights(dist_A_x, dist_A_y)

            # remove pixels and weights that are outside source plane grid
            if self._minimal_source_plane:
                j_list, weight_list = self._check_inside_grid(j_list, weight_list)

            # fill the mapping arrays
            lens_mapping_matrix[i, j_list] = weight_list

        # convert to optimized sparse matrix, using Compressed Sparse Row (CSR) format for fast vector products
        self._lens_mapping = sparse.csr_matrix(lens_mapping_matrix)

    def _find_surrounding_source_pixels(self, i, beta_x, beta_y, grid_offset_x=0, grid_offset_y=0):
        # map of the distance to the ray-traced pixel, along each axis, in pixel units
        diff_map_x_pix, diff_map_y_pix = self._difference_on_source_grid_axis(i, beta_x, beta_y, pixel_conversion=True,
                                                                              grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y)

        # index of source pixel that is the closest to the ray-traced pixel
        j = np.argmin(diff_map_x_pix**2 + diff_map_y_pix**2)

        # find the 4 neighboring pixels
        nb_list, nb_list_2d, idx_closest = self._neighboring_pixels(j, diff_map_x_pix, diff_map_y_pix)

        # compute distance (in pixel units) to lower-left "A" neighboring pixel
        dist_to_A_x, dist_to_A_y = self._distance_lower_left_neighbor(i, beta_x, beta_y, nb_list, pixel_conversion=True,
                                                                      grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y)

        return nb_list, idx_closest, dist_to_A_x, dist_to_A_y

    def _neighboring_pixels(self, j, difference_x, difference_y):
        """
        returns the 4 surrounding pixels in the following order [A, B, C, D]
          C:(0, 1) .____. D:(1, 1)
                   | o  |
                   |    |
          A:(0, 0) '----' B:(1, 0)

        difference_x, difference_y should be in *pixel units*
        """
        diff_x_j, diff_y_j = difference_x[j], difference_y[j]
        # convert to 2D indices
        r, s = self._index_1d_to_2d_source(j)
        if diff_x_j >= 0 and diff_y_j >= 0:
            # closest pixel is A (if the pixel distance to grid is defined as "pixel - coordinates")
            nb_list_2d = [(r, s), (r, s+1), (r+1, s), (r+1, s+1)]
            idx_closest = 0
        elif diff_x_j < 0 and diff_y_j >= 0:
            # closest pixel is B
            nb_list_2d = [(r, s-1), (r, s), (r+1, s-1), (r+1, s)]
            idx_closest = 1
        elif diff_x_j >= 0 and diff_y_j < 0:
            # closest pixel is C
            nb_list_2d = [(r-1, s), (r-1, s+1), (r, s), (r, s+1)]
            idx_closest = 2
        elif diff_x_j < 0 and diff_y_j < 0:
            # closest pixel is D
            nb_list_2d = [(r-1, s-1), (r-1, s), (r, s-1), (r, s)]
            idx_closest = 3
        else:
            raise ValueError("Could not find 4 neighboring pixels for pixel {} ({},{})".format(j, r, s))
        # check if indices are not outside of the image, put None if it is
        max_index_value = self.sourcePlane.num_pix - 1
        for idx, (r, s) in enumerate(nb_list_2d):
            if r >= max_index_value or s >= max_index_value:
                nb_list_2d[idx] = (None, None)
        # convert indices to 1D index
        nb_list = [self._index_2d_to_1d_source(r, s) for (r, s) in nb_list_2d]
        return nb_list, nb_list_2d, idx_closest

    def _distance_lower_left_neighbor(self, i, beta_x, beta_y, neighbor_list, pixel_conversion=False,
                                      grid_offset_x=0, grid_offset_y=0):
        nb_idx_A = 0  # following conventions of self._neighboring_pixels()
        i_A = neighbor_list[nb_idx_A]
        theta_x_A = self.sourcePlane.theta_x[i_A] + grid_offset_x
        theta_y_A = self.sourcePlane.theta_y[i_A] + grid_offset_y
        dist_to_A_x = abs(theta_x_A - beta_x[i])
        dist_to_A_y = abs(theta_y_A - beta_y[i])
        if pixel_conversion:
            dist_to_A_x /= self.sourcePlane.delta_pix
            dist_to_A_y /= self.sourcePlane.delta_pix
        return dist_to_A_x, dist_to_A_y

    def _bilinear_weights(self, distance_to_A_x, distance_to_A_y):
        """
        returns bilinear weights following order defined in self._neighboring_pixels()
        similar to Eq. (B2) of Treu & Koopmans 2004
        """
        t, u = distance_to_A_x, distance_to_A_y
        wA = (1. - t) * (1. - u)
        wB = t * (1. - u)
        wC = (1. - t) * u
        wD = t * u
        return [wA, wB, wC, wD]

    def _check_inside_grid(self, pixel_list, weight_list):
        """
        remove from pixel and weights list 
        """
        pixel_list_clean, weight_list_clean = [], []
        for p, w in zip(pixel_list, weight_list):
            if p is not None:
                pixel_list_clean.append(p)
                weight_list_clean.append(w)
        return pixel_list_clean, weight_list_clean

    def plot_neighbors_map(self, kwargs_lens, num_image_pixels=100):
        """utility for debug only : visualize mapping and interpolation""" 
        import matplotlib.pyplot as plt

        theta_x_source, theta_y_source = self.source_plane_coordinates

        # backward ray-tracing to get source coordinates in image plane (the 'betas')
        beta_x, beta_y = self.lensModel.ray_shooting(self.imagePlane.theta_x, self.imagePlane.theta_y, kwargs_lens)

        zeros = np.zeros(self.sourcePlane.grid_shape)
        fig = plt.figure(figsize=(15,15))
        zeros[0, 0] = 6
        plt.plot([0], [0], '+k', zorder=2)
        plt.text(0, 0, "pixel center", fontsize=8, color='black')

        # iterate through indices of image plane (indices 'i')
        for i in range(self.imagePlane.grid_size):
            # find source pixel that corresponds to ray traced image pixel
            j_1d_list, closest, dist_A_x, dist_A_y = self._find_surrounding_source_pixels(i, beta_x, beta_y)
            j_2d_list = [self._index_1d_to_2d_source(j) for j in j_1d_list]

            # get interpolation weights
            weight_list = self._bilinear_weights(dist_A_x, dist_A_y)
            weight_list_print = ",".join(["{:.2f}".format(w) for w in weight_list])

            j = j_1d_list[closest]

            if i % num_image_pixels == 0:
                ray_traced_pixel_x = -0.5+(beta_x[i]+self.sourcePlane.num_pix*self.sourcePlane.delta_pix/2.)/self.sourcePlane.delta_pix
                ray_traced_pixel_y = -0.5+(beta_y[i]+self.sourcePlane.num_pix*self.sourcePlane.delta_pix/2.)/self.sourcePlane.delta_pix
                plt.plot([ray_traced_pixel_x], [ray_traced_pixel_y], 'xw', zorder=2)
                # plt.text(ray_traced_pixel_x, ray_traced_pixel_y, "({:.3f},{:.3f})".format(dist_A_x, dist_A_y), fontsize=8, color='white')
                plt.text(ray_traced_pixel_x, ray_traced_pixel_y, weight_list_print, fontsize=8, color='white')

                for l, (jx, jy) in enumerate(j_2d_list):
                    zeros[jx, jy] += l+1
                neighbor_pixel_x = -0.5+(theta_x_source[j]+self.sourcePlane.num_pix*self.sourcePlane.delta_pix/2.)/self.sourcePlane.delta_pix
                neighbor_pixel_y = -0.5+(theta_y_source[j]+self.sourcePlane.num_pix*self.sourcePlane.delta_pix/2.)/self.sourcePlane.delta_pix
                plt.plot([neighbor_pixel_x], [neighbor_pixel_y], '.w', zorder=1)
                plt.imshow(zeros, origin='lower', zorder=0)

        return fig
