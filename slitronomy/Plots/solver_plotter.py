__author__ = 'aymgal'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from slitronomy.Util import plot_util


class SolverPlotter(object):

    _cmap_1 = 'cubehelix'
    _cmap_2 = 'gist_stern'

    def __init__(self, solver_class):
        self._solver = solver_class

    def plot_init(self, image, show_now=True):
        title = "initial guess"
        return self.quick_imshow(image, title=title, show_now=show_now, cmap=self._cmap_2)

    def plot_step(self, image, iter_1, iter_2=None, show_now=True):
        if iter_2 is not None:
            title = "iteration {}-{}".format(iter_1, iter_2)
        else:
            title = "iteration {}".format(iter_1)
        return self.quick_imshow(image, title=title, show_now=show_now, cmap=self._cmap_2)

    def plot_final(self, image, show_now=True):
        title = "final reconstruction"
        return self.quick_imshow(image, title=title, show_now=show_now, cmap=self._cmap_2)

    def plot_results(self, model_log_scale=False, res_vmin=None, res_vmax=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        ax.set_title("source model")
        src_model = self._solver.source_model
        print("Negative source pixels ?", np.any(src_model < 0))
        if model_log_scale:
            vmin = max(src_model.min(), 1e-3)
            vmax = min(src_model.max(), 1e10)
            src_model[src_model <= 0.] = 1e-10
            im = ax.imshow(src_model, origin='lower', cmap=self._cmap_1, 
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.imshow(src_model, origin='lower', cmap=self._cmap_1)
        # ax.imshow(self.lensingOperator.sourcePlane.reduction_mask, origin='lower', cmap='gray', alpha=0.1)
        plot_util.nice_colorbar(im)
        ax = axes[0, 1]
        ax.set_title("image model")
        img_model = self._solver.image_model(unconvolved=False)
        print("Negative image pixels ?", np.any(img_model < 0))
        if model_log_scale:
            vmin = max(img_model.min(), 1e-3)
            vmax = min(img_model.max(), 1e10)
            img_model[img_model <= 0.] = 1e-10
            im = ax.imshow(img_model, origin='lower', cmap=self._cmap_1,
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.imshow(img_model, origin='lower', cmap=self._cmap_1)
        plot_util.nice_colorbar(im)
        ax = axes[0, 2]
        ax.set_title(r"(data - model)$/\sigma$")
        im = ax.imshow(self._solver.reduced_residuals(self._solver.source_model), origin='lower',
                       cmap='bwr', vmin=res_vmin, vmax=res_vmax)
        text = r"$\chi^2={:.2f}$".format(self._solver.best_fit_reduced_chi2)
        ax.text(0.05, 0.05, text, color='black', fontsize=15, 
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        plot_util.nice_colorbar(im)
        ax = axes[1, 0]
        ax.set_title("loss function")
        ax.plot(self._solver.solve_track['loss'])
        ax.set_xlabel("iterations")
        ax = axes[1, 1]
        ax.set_title("reduced chi2")
        ax.plot(self._solver.solve_track['red_chi2'])
        ax.set_xlabel("iterations")
        ax = axes[1, 2]
        ax.set_title("step-to-step difference")
        ax.semilogy(self._solver.solve_track['step_diff'])
        ax.set_xlabel("iterations")
        plt.show()

    @staticmethod
    def quick_imshow(image, title=None, show_now=False, **kwargs):
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        ax = axes
        if title is not None:
            ax.set_title(title)
        im = ax.imshow(image, origin='lower', **kwargs)
        plot_util.nice_colorbar(im)
        if show_now:
            plt.show()