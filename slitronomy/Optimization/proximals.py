__author__ = 'aymgal'

# implementations of proximal operators adapted to sparsity

import numpy as np
from slitronomy.Util import util


def prox_sparsity_wavelets(coeffs_input, step, level_const=None, level_pixels=None, l_norm=1):
    """
    Apply soft or hard threshold on all wavelets scales excepts the last one (the coarse scale)  
    """
    if l_norm not in [0, 1]:
        raise ValueError("Sparsity proximal operator only defined with l0- and l1-norms")
    if step == 0:
        return coeffs_input

    coeffs = np.copy(coeffs_input)
    n_scales = coeffs.shape[0]

    # apply threshold operation to all starlet scales except the coarsest
    for s in range(n_scales-1):
        thresh = step
        if level_const is not None:
            thresh *= level_const[s]
        if level_pixels is not None: 
            thresh *= level_pixels[s, :, :]
        if l_norm == 0:
            coeffs[s, :, :] = util.hard_threshold(coeffs[s, :, :], thresh)
        else:
            coeffs[s, :, :] = util.soft_threshold(coeffs[s, :, :], thresh)

    return coeffs


def prox_positivity(image_input):
    image = np.copy(image_input)
    image[image < 0] = 0.
    return image
