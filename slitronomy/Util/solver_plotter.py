__author__ = 'aymgal'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from slitronomy.Util import plot_util


class SolverPlotter(object):

    _cmap_1 = 'cubehelix'
    _cmap_2 = 'gist_stern'
    _cmap_3 = 'bwr'
    _color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

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

    def plot_results(self, model_log_scale=False, res_vmin=-6, res_vmax=6, cmap_1=None, cmap_2=None):
        n_comp = self._solver.track['loss'].shape[0]
        names = self._solver.component_names
        fig, axes = plt.subplots(2, 4, figsize=(22, 9))
        ax = axes[0, 0]
        ax.set_title("imaging data")
        data = self._solver.M(self._solver.Y)
        if model_log_scale:
            vmin = max(data.min(), 1e-3)
            vmax = min(data.max(), 1e10)
            data[data <= 0.] = 1e-10
            im = ax.imshow(data, origin='lower', cmap=self._cmap_1, 
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            im = ax.imshow(data, origin='lower', cmap=self._cmap_1)
        plot_util.nice_colorbar(im)
        ax = axes[0, 1]
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
            if cmap_1 is None:
                cmap_1 = self._cmap_1
            im = ax.imshow(src_model, origin='lower', cmap=cmap_1)
        # ax.imshow(self.lensingOperator.sourcePlane.reduction_mask, origin='lower', cmap='gray', alpha=0.1)
        plot_util.nice_colorbar(im)
        ax = axes[0, 2]
        if not self._solver.no_lens_light:
            ax.set_title("lens light model")
            img_model = self._solver.lens_light_model
        # elif not self._solver.no_point_source:
        #     ax.set_title("point source model")
        #     img_model = self._solver.point_source_model
        else:
            ax.set_title("image model")
            img_model = self._solver.image_model(unconvolved=False)
            print("Negative image pixels ? {} (min = {:.2e})".format(np.any(img_model < 0), img_model.min()))
        if model_log_scale:
            vmin = max(img_model.min(), 1e-3)
            vmax = min(img_model.max(), 1e10)
            img_model[img_model <= 0.] = 1e-10
            im = ax.imshow(img_model, origin='lower', cmap=self._cmap_2,
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        else:
            if cmap_2 is None:
                cmap_2 = self._cmap_2
            im = ax.imshow(img_model, origin='lower', cmap=cmap_2)
        plot_util.nice_colorbar(im)
        ax = axes[0, 3]
        ax.set_title(r"(data - model)$/\sigma$")
        im = ax.imshow(self._solver.reduced_residuals_model, 
                       origin='lower', cmap=self._cmap_3, vmin=res_vmin, vmax=res_vmax)
        text = r"$\chi^2={:.2f}$".format(self._solver.best_fit_reduced_chi2)
        ax.text(0.2, 0.1, text, color='black', fontsize=15, 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8})
        plot_util.nice_colorbar(im)
        ax = axes[1, 0]
        ax.set_title("loss | regularization")
        for i in range(n_comp):
            data = self._solver.track['loss'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i], 
                    label='loss({})'.format(names[i]))
        ax.set_xlabel("iterations")
        # ax.set_ylabel("loss")
        ax.legend(loc='upper right')
        ax = axes[1, 1]
        for i in range(n_comp):
            data = self._solver.track['reg'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i+n_comp], 
                     label='reg({})'.format(names[i]))
        # ax.set_ylabel("regularization")
        ax.legend(loc='upper right')
        ax = axes[1, 2]
        ax.set_title("reduced chi2")
        for i in range(n_comp):
            data = self._solver.track['red_chi2'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i])
        ax.set_xlabel("iterations")
        ax.set_ylabel(r"$\chi^2_{\nu}$")
        ax = axes[1, 3]
        ax.set_title("step-to-step difference")
        for i in range(n_comp):
            data = self._solver.track['step_diff'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i])
        ax.set_xlabel("iterations")
        ax.set_ylabel(r"$||x_{i+1}-x_{i}||_2$")
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