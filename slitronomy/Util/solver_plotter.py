__author__ = 'aymgal'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from slitronomy.Util import plot_util


class SolverPlotter(object):

    _cmap_1 = 'cubehelix'
    _cmap_2 = 'gist_stern'
    _cmap_3 = 'bwr'

    def __init__(self, solver_class, show_now=True):
        self._solver = solver_class
        self._show_now = show_now

    def plot_init(self, image):
        title = "initial guess"
        return self.quick_imshow(image, title=title, show_now=self._show_now, cmap=self._cmap_2)

    def plot_step(self, image, iter_1, iter_2=None, iter_3=None):
        if iter_3 is not None:
            title = "iteration {}-{}-{}".format(iter_1, iter_2, iter_3)
        elif iter_2 is not None:
            title = "iteration {}-{}".format(iter_1, iter_2)
        else:
            title = "iteration {}".format(iter_1)
        return self.quick_imshow(image, title=title, show_now=self._show_now, cmap=self._cmap_2)

    def plot_final(self, image):
        title = "final reconstruction"
        return self.quick_imshow(image, title=title, show_now=self._show_now, cmap=self._cmap_2)

    def plot_results(self, model_log_scale=False, res_vmin=None, res_vmax=None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        ax = axes[0, 0]
        ax.set_title("source model")
        src_model = self._solver.source_model
        print("Negative source pixels ? {} (min = {:.2e})".format(np.any(src_model < 0), src_model.min()))
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
        if self._solver.lens_light_model is not None:
            ax.set_title("lens light model")
            img_model = self._solver.lens_light_model
            print("Negative lens pixels ? {} (min = {:.2e})".format(np.any(img_model < 0), img_model.min()))
        else:
            ax.set_title("image model")
            img_model = self._solver.image_model(unconvolved=False)
            print("Negative image pixels ? {} (min = {:.2e})".format(np.any(img_model < 0), img_model.min()))
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
        im = ax.imshow(self._solver.reduced_residuals_model, 
                       origin='lower', cmap=self._cmap_3, vmin=res_vmin, vmax=res_vmax)
        text = r"$\chi^2={:.2f}$".format(self._solver.best_fit_reduced_chi2)
        ax.text(0.2, 0.1, text, color='black', fontsize=15, 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8})
        plot_util.nice_colorbar(im)
        ax = axes[1, 0]
        ax.set_title("loss function")
        ax.plot(self._solver.track['loss'].T, '.')
        ax.set_xlabel("iterations")
        ax = axes[1, 1]
        ax.set_title("reduced chi2")
        ax.plot(self._solver.track['red_chi2'].T, '.')
        ax.set_xlabel("iterations")
        ax = axes[1, 2]
        ax.set_title("step-to-step difference")
        ax.semilogy(self._solver.track['step_diff'].T, '.')
        ax.set_xlabel("iterations")
        return fig

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