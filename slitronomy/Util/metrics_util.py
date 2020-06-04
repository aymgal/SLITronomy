__author__ = 'aymgal'

import numpy as np


def SDR(truth, model):
    """computes Source Distortion Ratio"""
    diff = truth - model
    return 10 * np.log10( np.linalg.norm(truth.flatten(), 2) / np.linalg.norm(diff.flatten(), 2) )


def SSIM(truth, model):
    """computes Structure Similarity Index"""
    import tensorflow as tf
    dyn_range = truth.max() - truth.min()
    t = tf.convert_to_tensor(truth[:, :, np.newaxis])
    m = tf.convert_to_tensor(model[:, :, np.newaxis])
    ssim = tf.image.ssim(t, m, max_val=dyn_range)
    return ssim.numpy()


def QOR(truth, model, noise_map):
    """computes Quality Of Reconstruction"""
    return np.std((truth-model)/noise_map)
    

def chi2_nu(normalized_residuals, num_data_points):
    """computes the reduced chi2 from a normalized residual map"""
    return np.sum(normalized_residuals**2) / num_data_points


def total_mag(flux_map, zero_point):
    return -2.5 * np.log10(np.sum(flux_map)) + zero_point
