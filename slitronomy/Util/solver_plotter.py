__author__ = 'aymgal'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from slitronomy.Util import plot_util
from slitronomy.Util import metrics_util

class SolverPlotter(object):

    _cmap_1 = 'cubehelix'
    _cmap_2 = 'RdBu_r'
    _cmap_misc = 'gist_stern'
    _color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    _vmin_log = 1e-5
    _vmax_log = 1e10

    def __init__(self, solver_class, show_now=True):
        self._solver = solver_class
        self._show_now = show_now

    def plot_init(self, image):
        title = "initial guess"
        return self.quick_imshow(image, title=title, show_now=self._show_now, cmap=self._cmap_misc)

    def plot_step(self, image, iter_1, iter_2=None, iter_3=None):
        if iter_3 is not None:
            title = "iteration {}-{}-{}".format(iter_1, iter_2, iter_3)
        elif iter_2 is not None:
            title = "iteration {}-{}".format(iter_1, iter_2)
        else:
            title = "iteration {}".format(iter_1)
        return self.quick_imshow(image, title=title, show_now=self._show_now, cmap=self._cmap_misc)

    def plot_final(self, image):
        title = "final reconstruction"
        return self.quick_imshow(image, title=title, show_now=self._show_now, cmap=self._cmap_misc)

    def plot_results(self, log_scale=False, vmin_image=None, vmax_image=None, 
                     vmin_source=None, vmax_source=None, vmin_res=-6, vmax_res=6,
                     cmap_image=None, cmap_source=None, fontsize=12, with_history=True, unconvolved=False):
        if cmap_image is None:
            cmap_image = self._cmap_1
        if cmap_source is None:
            cmap_source = self._cmap_1

        n_comp = self._solver.track['loss'].shape[0]
        names = self._solver.component_names
        n_rows, n_cols = 2, 4
        if with_history:
            n_rows, n_cols, figsize = 2, 4, (22, 9)
        else:
            n_rows, n_cols, figsize = 1, 4, (22, 4)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        # ====== IMAGING DATA ====== #
        ax = axes[0, 0]
        ax.set_title("imaging data", fontsize=fontsize)
        data = self._solver.M(self._solver.Y)
        norm = self._prepare_color_norm(data, log_scale, vmin_image, vmax_image)
        im = ax.imshow(data, origin='lower', cmap=cmap_image, norm=norm)
        plot_util.nice_colorbar(im, label="flux", fontsize=fontsize)

        # ====== IMAGE MODEL ====== #
        ax = axes[0, 1]
        if not self._solver.no_lens_light:
            ax.set_title("lens light model", fontsize=fontsize)
            img_model = self._solver.lens_light_model
        # elif not self._solver.no_point_source:
        #     ax.set_title("point source model")
        #     img_model = self._solver.point_source_model
        else:
            ax.set_title("image model", fontsize=fontsize)
            img_model = self._solver.image_model(unconvolved=unconvolved)
            print("Negative image pixels ? {} (min = {:.2e})".format(np.any(img_model < 0), img_model.min()))
        norm = self._prepare_color_norm(img_model, log_scale, vmin_image, vmax_image)
        im = ax.imshow(img_model, origin='lower', cmap=cmap_image, norm=norm)
        plot_util.nice_colorbar(im, label="flux", fontsize=fontsize)

        # ====== SOURCE MODEL ====== #
        ax = axes[0, 2]
        ax.set_title("source model", fontsize=fontsize)
        src_model = self._solver.source_model
        print("Negative source pixels ? {} (min = {:.2e})".format(np.any(src_model < 0), src_model.min()))
        norm = self._prepare_color_norm(src_model, log_scale, vmin_source, vmax_source)
        im = ax.imshow(src_model, origin='lower', cmap=cmap_source, norm=norm)
        plot_util.nice_colorbar(im, label="flux", fontsize=fontsize)

        # ====== NORMALIZED RESIDUALS ====== #
        ax = axes[0, 3]
        ax.set_title(r"norm. residuals", fontsize=fontsize)
        residuals_map = self._solver.normalized_residuals_model
        residuals_map_min, residuals_map_max = residuals_map.min(), residuals_map.max()
        im = ax.imshow(residuals_map, 
                       origin='lower', cmap=self._cmap_2, vmin=vmin_res, vmax=vmax_res)
        text = r"$\chi^2={:.2f}$".format(self._solver.best_fit_reduced_chi2)
        ax.text(0.2, 0.1, text, color='black', fontsize=15, 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, bbox={'color': 'white', 'alpha': 0.8})
        plot_util.nice_colorbar_residuals(im, residuals_map, vmin_res, vmax_res, 
                                          label=r"(f${}_{\rm model}$ - f${}_{\rm data}$)/$\sigma$", 
                                          fontsize=fontsize)

        if not with_history:
            return fig

        # ====== CONVERGENCE HISTORY PLOTS ====== #
        ax = axes[1, 0]
        ax.set_title("loss", fontsize=fontsize)
        for i in range(n_comp):
            data = self._solver.track['loss'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i], 
                    label='loss({})'.format(names[i]))
        ax.set_xlabel("iterations", fontsize=fontsize)
        ax.set_ylabel(r"$||{\rm Y} - {\rm HFS}||_2^2\ /\ 2$", fontsize=fontsize)
        if n_comp > 1:
            ax.set_ylabel(r"$||{\rm Y} - {\rm G_H} - {\rm HFS}||_2^2\ /\ 2$", fontsize=fontsize)
            ax.legend(loc='upper right')

        ax = axes[1, 1]
        ax.set_title("regularization", fontsize=fontsize)
        for i in range(n_comp):
            data = self._solver.track['reg'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i+n_comp], 
                     label='reg({})'.format(names[i]))
        ax.set_xlabel("iterations", fontsize=fontsize)
        ax.set_ylabel(r"$||\Phi^\top{\rm S}||_"+str(self._solver.prior_l_norm)+r"$", fontsize=fontsize)
        if n_comp > 1:
            ax.legend(loc='upper right')

        ax = axes[1, 2]
        ax.set_title(r"reduced $\chi^2$", fontsize=fontsize)
        for i in range(n_comp):
            data = self._solver.track['red_chi2'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i])
        ax.set_xlabel("iterations", fontsize=fontsize)
        ax.set_ylabel(r"$\chi^2_{\nu}$", fontsize=fontsize)

        ax = axes[1, 3]
        ax.set_title("step-to-step difference", fontsize=fontsize)
        for i in range(n_comp):
            data = self._solver.track['step_diff'][i, :]
            if np.all(np.isnan(data)): continue
            ax.plot(data, linestyle='none', marker='.', color=self._color_cycle[i])
        ax.set_xlabel("iterations", fontsize=fontsize)
        ax.set_ylabel(r"$||x_{i+1}-x_{i}||_2$", fontsize=fontsize)
        return fig

    @staticmethod
    def plot_source_residuals_comparison(source_truth, source_model_list, name_list, 
                                         vmin_res=-0.5, vmax_res=-0.5, cmap='cubehelix',
                                         fontsize=12):
        """given a true source, plot residuals of a list of source model"""
        n_model = len(source_model_list)
        fig, axes = plt.subplots(1, 1+2*n_model, figsize=((1+2*n_model)*4.5, 3))
        ax = axes[0]
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        ax.set_title("true source", fontsize=fontsize)
        im = ax.imshow(source_truth, origin='lower', cmap=cmap, vmin=0)
        lims = (len(source_truth)/4, 3*len(source_truth)/4)  # zoom a bit on the image
        #ax.set_xlim(*lims)
        #ax.set_ylim(*lims)
        plot_util.nice_colorbar(im, label="flux", fontsize=fontsize)
        
        i = 1
        for source_model, name in zip(source_model_list, name_list):
            print("min/max for source model '{}': {}/{}".format(name, source_model.min(), source_model.max()))

            residuals_source = source_truth - source_model
            residuals_map_min, residuals_map_max = residuals_source.min(), residuals_source.max()

            ax = axes[i]
            ax.set_title("model '{}'".format(name), fontsize=fontsize)
            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            im = ax.imshow(source_model, origin='lower', cmap=cmap)
            #ax.set_xlim(*lims)
            #ax.set_ylim(*lims)
            plot_util.nice_colorbar(im, label="flux", fontsize=fontsize)
            i += 1
            
            ax = axes[i]
            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            ax.set_title("difference", fontsize=fontsize)
            #ax.set_xlim(*lims)
            #ax.set_ylim(*lims)
            im = ax.imshow(residuals_source, origin='lower', cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res)
            plot_util.nice_colorbar_residuals(im, residuals_source, vmin_res, vmax_res,
                                                label=r"f${}_{\rm model}$ - f${}_{\rm truth}$", fontsize=fontsize)
            i += 1
            
            print("SDR for model '{}' = {:.3f}".format(name, metrics_util.SDR(source_truth, source_model)))
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

    def _prepare_color_norm(self, image, log_scale, vmin_user, vmax_user):
        if vmin_user is None:
            if log_scale:
                vmin = max(image.min(), self._vmin_log)
            else:
                vmin = image.min()
        else:
            vmin = vmin_user
        if vmax_user is None:
            if log_scale:
                vmax = min(image.max(), self._vmax_log)
            else:
                vmax = image.max()
        else:
            vmax = vmax_user
        if log_scale:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        return norm