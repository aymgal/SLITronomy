import numpy as np
from scipy import sparse

import lenstronomy.Util.util as lenstro_util


def pixelated_planes_mapping(data_class, lens_model_class, kwargs_lens, num_pix_source_plane, delta_pix_source_plane):
    image_sim = data_class.data
    # get the coordinates in image plane (the usual 'thetas')
    x_grid, y_grid = data_class.pixel_coordinates
    theta_x_1d = lenstro_util.image2array(x_grid)
    theta_y_1d = lenstro_util.image2array(y_grid)
    
    # get the coordinates of source plane (those are 'thetas' but in source plane !)
    x_grid_src_1d, y_grid_src_1d = lenstro_util.make_grid(numPix=num_pix_source_plane, deltapix=delta_pix_source_plane, 
                                                          subgrid_res=1)
    theta_x_src_1d = x_grid_src_1d
    theta_y_src_1d = y_grid_src_1d
    
    # backward ray-tracing to get source coordinates in image plane (the usual 'betas')
    beta_x_1d, beta_y_1d = lens_model_class.ray_shooting(theta_x_1d, theta_y_1d, kwargs_lens)
    
    # declare array to store initially the mapping
    mass_mapping_array = np.zeros((image_sim.size, num_pix_source_plane**2))

    # 1. iterate through indices of image plane (indices 'i')
    for i in range(image_sim.size):    
        # 2. get the coordinates of ray traced pixels (in image plane coordinates)
        beta_x_i = beta_x_1d[i]
        beta_y_i = beta_y_1d[i]

        # 3. compute the closest coordinates in source plane (indices 'j')
        distance_on_source_grid_x = np.abs(beta_x_i - theta_x_src_1d)
        distance_on_source_grid_y = np.abs(beta_y_i - theta_y_src_1d)
        j_x = np.argmin(distance_on_source_grid_x)
        j_y = np.argmin(distance_on_source_grid_y)
        if isinstance(j_x, list):
            j_x = j_x[0]
            print("Warning : found more than one possible x coordinates in source plane for index i={}".format(i))
        if isinstance(j_y, list):
            j_y = j_y[0]
            print("Warning : found more than one possible y coordinates in source plane for index i={}".format(i))
    
        # coordinates in source plane
        #theta_x_j = theta_x_src_1d[j_x]
        #theta_y_j = theta_y_src_1d[j_y]

        # 4. find the 1D index that corresponds to these coordinates
        j = j_x + j_y

        # 5. fill the mapping array
        mass_mapping_array[i, j] = 1

    # convert numpy array to sparse matrix, using Compressed Sparse Row (CSR) format for fast vector products
    mass_mapping_sparse = sparse.csr_matrix(mass_mapping_array)
    # NOTE : Compressed Sparse Column (CSC) made almost no difference in arithemtics operations, and takes more memory

    # delete original array from memory
    del mass_mapping_array
    
    return mass_mapping_sparse