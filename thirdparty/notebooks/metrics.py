import numpy as np


def SDR(truth, model):
    """computes Source Distortion Ratio"""
    diff = truth - model
    return 10 * np.log10( np.linalg.norm(truth.flatten(), 2) / np.linalg.norm(diff.flatten(), 2) )

def QOR(truth, model, noise_map):
    """computes Quality Of Reconstruction"""
    return np.std((truth-model)/noise_map)