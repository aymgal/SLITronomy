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
    """Map pixelated light profiles between image and source planes."""
    def __init__(self, data_class, lens_model_class, subgrid_res_source=1,
                 likelihood_mask=None, minimal_source_plane=False, use_mask_for_minimal_source_plane=True,
                 fix_minimal_source_plane=True, min_num_pix_source=10):
        _matrix_prod = True
        super(LensingOperatorInterpol, self).__init__(data_class,
            lens_model_class, subgrid_res_source=subgrid_res_source,
            likelihood_mask=likelihood_mask,
            minimal_source_plane=minimal_source_plane,
            fix_minimal_source_plane=fix_minimal_source_plane,
            min_num_pix_source=min_num_pix_source,
            use_mask_for_minimal_source_plane=use_mask_for_minimal_source_plane,
            matrix_prod=_matrix_prod)

    def _compute_mapping(self, kwargs_lens, kwargs_special):
        """Compute mapping between image and source plane pixels.

        This method uses lenstronomy to ray-trace the image plane pixel
        coordinates back to the source plane and regularize the resultig
        positions to a grid. In contrast to LensingOperator, the mapping
        incorporates a bilinear weighting scheme to interpolate flux on the
        source plane following Treu & Koopmans (2004).

        """
        # Remove previous mapping
        self.delete_mapping()

        # Compute lens mapping from image to source coordinates
        beta_x, beta_y = self.lensModel.ray_shooting(self.imagePlane.theta_x,
                                                     self.imagePlane.theta_y,
                                                     kwargs_lens)

        # Get source plane offsets
        grid_offset_x, grid_offset_y = self._source_grid_offsets(kwargs_special)

        # Determine source pixels and their appropriate weights
        indices, weights = self._find_source_pixels(beta_x, beta_y,
                                                    grid_offset_x,
                                                    grid_offset_y)

        # Build lensing matrix as a csr (sparse) matrix for fast operations
        dense_shape = (self.imagePlane.grid_size, self.sourcePlane.grid_size)
        self._lens_mapping = sparse.csr_matrix((weights, indices),
                                               shape=dense_shape)

    def _find_source_pixels(self, beta_x, beta_y, grid_offset_x, grid_offset_y):
        """Fast binning of ray-traced coordinates and weight calculation.

        NOTE : We probably still need to verify what happens when ray-traced
               coordinates fall outside of the source plane grid.

        """
        # Standardize inputs for vectorization
        beta_x = np.atleast_1d(beta_x)
        beta_y = np.atleast_1d(beta_y)

        # Shift source grid if necessary
        source_theta_x = self.sourcePlane.theta_x + grid_offset_x
        source_theta_y = self.sourcePlane.theta_y + grid_offset_y

        # Compute bin edges so that (theta_x, theta_y) lie at the centers
        num_pix = self.sourcePlane.num_pix
        delta_pix = self.sourcePlane.delta_pix
        half_pix = delta_pix / 2

        theta_x = source_theta_x[:num_pix]
        xbins = np.linspace(theta_x[0] - half_pix, theta_x[-1] + half_pix,
                            num_pix + 1)
        index_x = np.digitize(beta_x, xbins) - 1

        theta_y = source_theta_y[::num_pix]
        ybins = np.linspace(theta_y[0] - half_pix, theta_y[-1] + half_pix,
                            num_pix + 1)
        index_y = np.digitize(beta_y, ybins) - 1

        # Find the (1D) source plane pixel that (beta_x, beta_y) falls in
        index_1 = index_x + index_y * num_pix

        # Compute distances between ray-traced betas and source grid points
        dx = beta_x - source_theta_x[index_1]
        dy = beta_y - source_theta_y[index_1]

        # Find the three other nearest pixels
        index_2 = index_1 + np.sign(dx).astype(int)
        index_3 = index_1 + np.sign(dy).astype(int) * num_pix
        index_4 = index_2 + np.sign(dy).astype(int) * num_pix

        # Gather indices (sorted for eventual correct SparseTensor ordering)
        cols = np.sort([index_1, index_2, index_3, index_4], axis=0).T.flat

        # Compute bilinear weights like in Treu & Koopmans (2004)
        dist_x = (np.repeat(beta_x, 4) - source_theta_x[cols]) / delta_pix
        dist_y = (np.repeat(beta_y, 4) - source_theta_y[cols]) / delta_pix
        weights = (1 - np.abs(dist_x)) * (1 - np.abs(dist_y))

        # Check for duplicates and ignore them. This is important for the
        # sparse csr_matrix to be generated correctly
        # WARNING : This only works if there is 1 set of sequential duplicates !
        mask = np.ones(len(weights), dtype=bool)
        if np.any(weights == 1):
            inds = np.where(weights == 1)[0]
            mask[inds[1:]] = False  # Keep only the first instance

        # Construct 2D indices for use in _compute_mapping
        rows = np.repeat(np.arange(self.imagePlane.grid_size), 4)
        indices = (rows[mask], cols[mask])

        return indices, weights[mask]
