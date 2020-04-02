__author__ = 'aymgal', 'austinpeel'

import numpy as np
from scipy import sparse

from slitronomy.Lensing.lensing_planes import ImagePlaneGrid, SourcePlaneGrid
from slitronomy.Util import util



class LensingOperator(object):

    """Defines the mapping of pixelated light profiles between image and source planes"""

    def __init__(self, data_class, lens_model_class, subgrid_res_source=1,
                 likelihood_mask=None, minimal_source_plane=False, min_num_pix_source=10,
                 fix_minimal_source_plane=True, use_mask_for_minimal_source_plane=True,
                 source_interpolation='bilinear', matrix_prod=True, verbose=False):
        """

        :param min_num_pix_source: minimal number of pixels in the
        """
        self.lensModel = lens_model_class
        self.imagePlane  = ImagePlaneGrid(data_class)
        self.sourcePlane = SourcePlaneGrid(data_class, subgrid_res=subgrid_res_source, verbose=verbose)
        self._subgrid_res_source = subgrid_res_source
        self._likelihood_mask = likelihood_mask
        self._minimal_source_plane = minimal_source_plane
        self._fix_minimal_source_plane = fix_minimal_source_plane
        self._use_mask_for_minimal_source_plane = use_mask_for_minimal_source_plane
        self._min_num_pix_source = min_num_pix_source
        if source_interpolation not in ['nearest', 'bilinear']:
            raise ValueError("source interpolation '{}' not supported in LensingOperator (only 'nearest', 'bilinear')")
        self._interpolation = source_interpolation
        self._matrix_prod = matrix_prod

    def source2image(self, source_1d, kwargs_lens=None, kwargs_special=None, update_lens=False):
        if not hasattr(self, '_lens_mapping') or update_lens:
            if kwargs_lens is None:
                raise ValueError("'kwargs_lens' is required to update lensing operator")
            self.update_mapping(kwargs_lens, kwargs_special=kwargs_special)
        if not self._matrix_prod and self._interpolation == 'nearest':
            image = self._source2image_list(source_1d)
        else:
            image = self._source2image_matrix(source_1d)
        return image

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

    def image2source(self, image_1d, kwargs_lens=None, kwargs_special=None, update_lens=False,
                     no_flux_norm=False):
        """if no_flux_norm is True, do not normalize light flux to better visualize the mapping"""
        if not hasattr(self, '_lens_mapping') or update_lens:
            if kwargs_lens is None:
                raise ValueError("'kwargs_lens' is required to update lensing operator")
            self.update_mapping(kwargs_lens, kwargs_special=kwargs_special)
        if not self._matrix_prod and self._interpolation == 'nearest':
            source = self._image2source_list(image_1d, update_lens=update_lens, no_flux_norm=no_flux_norm)
        else:
            source = self._image2source_matrix(image_1d, update_lens=update_lens, no_flux_norm=no_flux_norm)
        return source

    def image2source_2d(self, image, **kwargs):
        image_1d = util.image2array(image)
        return util.array2image(self.image2source(image_1d, **kwargs))

    def _image2source_matrix(self, image_1d, update_lens=False, no_flux_norm=False):
        if not hasattr(self, '_norm_image2source') or update_lens:
            # avoid computing it each time, save normalisation factors
            # normalization is sum of weights for each source pixel
            self._norm_image2source = np.squeeze(np.maximum(1, self._lens_mapping.sum(axis=0)).A)
        source_1d = self._lens_mapping.T.dot(image_1d)
        if not no_flux_norm:
            # normalization
            source_1d /= self._norm_image2source
        return source_1d

    def _image2source_list(self, image_1d, update_lens=False, no_flux_norm=False):
        source_1d = np.zeros(self.sourcePlane.grid_size)
        # loop over source plane pixels
        for j in range(source_1d.size):
            # retieve corresponding pixels in image plane
            indices_i = np.where(self._lens_mapping == j)
            flux_i = image_1d[indices_i]
            flux_j = np.sum(flux_i)
            if not no_flux_norm:
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

    @property
    def pixel_area_ratio(self):
        """source pixel area divide by image pixel area"""
        return self._subgrid_res_source**2

    def magnification_map(self, kwargs_lens):
        mag_map_1d = self.lensModel.magnification(self.imagePlane.theta_x, self.imagePlane.theta_y, kwargs_lens)
        return util.array2image(mag_map_1d)

    def update_mapping(self, kwargs_lens, kwargs_special=None):
        # reset source plane grid if it was altered by previous mass model
        if not self._fix_minimal_source_plane:
            self._reset_source_plane_grid()

        # compute mapping between image and source plances due to lensing
        self._compute_mapping(kwargs_lens, kwargs_special=kwargs_special)

        # compute areas on source plane where you have no constrains
        self._compute_source_mask()

        if self._minimal_source_plane and not self.sourcePlane.shrinked:
            # for source plane to be reduced to minimal size
            # we compute effective source mask and shrink the grid to match it
            self._shrink_source_plane_grid()

            # recompute the mapping with updated grid
            self._compute_mapping(kwargs_lens, kwargs_special=kwargs_special)

        return (self.sourcePlane.grid_size, self.sourcePlane.delta_pix)

    def delete_mapping(self):
        if hasattr(self, '_lens_mapping'): delattr(self, '_lens_mapping')
        if hasattr(self, '_norm_image2source'): delattr(self, '_norm_image2source')

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

    def _compute_mapping(self, kwargs_lens, kwargs_special=None):
        if self._interpolation == 'nearest':
            self._compute_mapping_nearest(kwargs_lens, kwargs_special=kwargs_special)
        elif self._interpolation == 'bilinear':
            self._compute_mapping_bilinear(kwargs_lens, kwargs_special=kwargs_special)

    def _compute_mapping_bilinear(self, kwargs_lens, kwargs_special):
        """Compute the mapping between image and source plane pixels.

        This method uses lenstronomy to ray-trace the image plane pixel
        coordinates back to the source plane and regularize the resultig
        positions to a grid. In contrast to the 'nearest' interpolation scheme,
        this mapping incorporates a bilinear weighting to interpolate flux on
        the source plane following Treu & Koopmans (2004).

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
        indices, weights = self._find_source_pixels_bilinear(beta_x, beta_y,
                                                             grid_offset_x,
                                                             grid_offset_y)

        # Build lensing matrix as a csr (sparse) matrix for fast operations
        dense_shape = (self.imagePlane.grid_size, self.sourcePlane.grid_size)
        self._lens_mapping = sparse.csr_matrix((weights, indices),
                                               shape=dense_shape)

    def _find_source_pixels_bilinear(self, beta_x, beta_y, grid_offset_x,
                                     grid_offset_y, warning=False):
        """Fast binning of ray-traced coordinates and weight calculation.

        Parameters
        ----------
        beta_x, beta_y : array-like
            Coordinates in the source plane of ray-traced points from the
            image plane (obtained from lenstronomy).
        grid_offset_x, grid_offset_y : float
            Amount by which to shift the source plane grid in each direction.
        warning : bool
            Print a warning if any returned weights are negative.

        Returns
        -------
        (row, col), weight
            Weights are the source grid interpolation values, which belong
            at position (row, col) in the sparse lensing matrix. There are at
            most 4 weights corresponding to each row.

        Notes
        -----
        Ray-traced coordinates from the image plane are simply removed if they
        fall outside of the source plane grid, as is done in Treu & Koopmans
        (2004). Although this should only rarely occur in practice, e.g. for
        extreme parameters of the lens model, a better approach might still be
        to expand the source plane instead.

        """
        # Standardize inputs for vectorization
        beta_x = np.atleast_1d(beta_x)
        beta_y = np.atleast_1d(beta_y)
        assert len(beta_x) == len(beta_y), "Input arrays must be the same size."
        num_beta = len(beta_x)

        # Shift source plane grid if necessary
        source_theta_x = self.sourcePlane.theta_x + grid_offset_x
        source_theta_y = self.sourcePlane.theta_y + grid_offset_y

        # Compute bin edges so that (theta_x, theta_y) lie at the grid centers
        num_pix = self.sourcePlane.num_pix
        delta_pix = self.sourcePlane.delta_pix
        half_pix = delta_pix / 2

        theta_x = source_theta_x[:num_pix]
        xbins = np.linspace(theta_x[0] - half_pix, theta_x[-1] + half_pix,
                            num_pix + 1)

        theta_y = source_theta_y[::num_pix]
        ybins = np.linspace(theta_y[0] - half_pix, theta_y[-1] + half_pix,
                            num_pix + 1)

        # Keep only betas that fall within the source plane grid
        selection = ((beta_x > xbins[0]) & (beta_x < xbins[-1]) &
                     (beta_y > ybins[0]) & (beta_y < ybins[-1]))
        if np.any(1 - selection.astype(int)):
            beta_x = beta_x[selection]
            beta_y = beta_y[selection]
            num_beta = len(beta_x)

        # Find the (1D) source plane pixel that (beta_x, beta_y) falls in
        index_x = np.digitize(beta_x, xbins) - 1
        index_y = np.digitize(beta_y, ybins) - 1
        index_1 = index_x + index_y * num_pix

        # Compute distances between ray-traced betas and source grid points
        dx = beta_x - source_theta_x[index_1]
        dy = beta_y - source_theta_y[index_1]

        # Find the three other nearest pixels (may end up out of bounds)
        index_2 = index_1 + np.sign(dx).astype(int)
        index_3 = index_1 + np.sign(dy).astype(int) * num_pix
        index_4 = index_2 + np.sign(dy).astype(int) * num_pix

        # Treat these index arrays as four sets stacked vertically
        # Prepare to mask out out-of-bounds pixels as well as repeats
        # The former is important for the csr_matrix to be generated correctly
        max_index = self.sourcePlane.grid_size - 1  # Upper index bound
        mask = np.ones((4, num_beta), dtype=bool)  # Mask for the betas

        # Mask out any neighboring pixels that end up out of bounds
        mask[1, np.where((index_2 < 0) | (index_2 > max_index))[0]] = False
        mask[2, np.where((index_3 < 0) | (index_3 > max_index))[0]] = False
        mask[3, np.where((index_4 < 0) | (index_4 > max_index))[0]] = False

        # Mask any repeated pixels (2 or 3x) arising from unlucky grid alignment
        zero_dx = list(np.where(dx == 0)[0])
        zero_dy = list(np.where(dy == 0)[0])
        unique, counts = np.unique(zero_dx + zero_dy, return_counts=True)
        repeat_row = [ii + 1 for c in counts for ii in range(0, 3, 3 - c)]
        repeat_col = [u for (u, c) in zip(unique, counts) for _ in range(c + 1)]
        mask[(repeat_row, repeat_col)] = False

        # Generate 2D indices of non-zero elements for the sparse matrix
        row = np.tile(np.nonzero(selection)[0], (4, 1))
        col = np.array([index_1, index_2, index_3, index_4])

        # Compute bilinear weights like in Treu & Koopmans (2004)
        col[~mask] = 0  # Avoid accessing source_thetas out of bounds
        dist_x = (np.tile(beta_x, (4, 1)) - source_theta_x[col]) / delta_pix
        dist_y = (np.tile(beta_y, (4, 1)) - source_theta_y[col]) / delta_pix
        weight = (1 - np.abs(dist_x)) * (1 - np.abs(dist_y))

        # Make sure the weights are properly normalized
        # This step is only necessary where the mask has excluded source pixels
        norm = np.expand_dims(np.sum(weight, axis=0, where=mask), 0)
        weight = weight / norm

        if warning:
            if np.any(weight[mask] < 0):
                num_neg = np.sum((weight[mask] < 0).astype(int))
                print("Warning : {} weights are negative.".format(num_neg))

        return (row[mask], col[mask]), weight[mask]

    def _compute_mapping_nearest(self, kwargs_lens, kwargs_special):
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
            j = self._find_source_pixel_nearest(i, beta_x, beta_y, grid_offset_x=grid_offset_x, grid_offset_y=grid_offset_y)

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

    def _find_source_pixel_nearest(self, i, beta_x, beta_y, grid_offset_x=0, grid_offset_y=0):
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
