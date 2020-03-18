__author__ = 'aymgal'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from slitronomy.Util import plot_util


class SolverTracker(object):


    def __init__(self, solver_class, verbose=False):
        self._solver = solver_class
        self._verbose = verbose

    @property
    def track(self):
        if not hasattr(self, '_track'):
            return None
        return self._track

    def init(self):
        self._track = {
            'reg': [[], []],
            'loss': [[], []],
            'red_chi2': [[], []],
            'step_diff': [[], []],
        }

    def save(self, S=None, S_next=None, HG=None, HG_next=None, print_bool=False, iteration_text=None):
        if not hasattr(self, '_track'):
            raise ValueError("Tracker has not been initialized")
        if S is not None:
            loss_S = self._solver.loss(S=S_next)
            reg_S = self._solver.regularization(S=S_next)
            red_chi2_S = self._solver.reduced_chi2(S=S_next)
            step_diff_S = self._solver.norm_diff(S, S_next)
        else:
            loss_S = np.nan
            reg_S = np.nan
            red_chi2_S = np.nan
            step_diff_S = np.nan
        if HG is not None:
            loss_HG = self._solver.loss(HG=HG_next)
            reg_HG = self._solver.regularization(HG=HG_next)
            red_chi2_HG = self._solver.reduced_chi2(HG=HG_next)
            step_diff_HG = self._solver.norm_diff(HG, HG_next)
        else:
            loss_HG = np.nan
            reg_HG = np.nan
            red_chi2_HG = np.nan
            step_diff_HG = np.nan
        # print info
        if self._verbose and print_bool:
            if iteration_text is None:
                iteration_text = "iteration ?"
            print("{} : loss+reg = {:.4f}|{:.4f}, red-chi2 = {:.4f}|{:.4f}, step_diff = {:.4f}|{:.4f}"
                  .format(iteration_text, loss_S+reg_S, loss_HG+reg_HG, red_chi2_S, red_chi2_HG, step_diff_S, step_diff_HG))
        # save in track
        self._track['loss'][0].append(loss_S)
        self._track['loss'][1].append(loss_HG)
        self._track['reg'][0].append(reg_S)
        self._track['reg'][1].append(reg_HG)
        self._track['red_chi2'][0].append(red_chi2_S)
        self._track['red_chi2'][1].append(red_chi2_HG)
        self._track['step_diff'][0].append(step_diff_S)
        self._track['step_diff'][1].append(step_diff_HG)

    def finalize(self):
        """convert to numpy array for practicality / plots"""
        self._track['loss'] = np.array(self._track['loss'])
        self._track['reg'] = np.array(self._track['reg'])
        self._track['red_chi2'] = np.array(self._track['red_chi2'])
        self._track['step_diff'] = np.array(self._track['step_diff'])
