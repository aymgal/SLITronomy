__author__ = 'aymgal'

import matplotlib
matplotlib.use("Agg")

from slitronomy.Util.solver_plotter import SolverPlotter
from slitronomy.Optimization.solver_source import SparseSolverSource

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as l_util

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest


class TestSolverPlotter(object):
    """
    tests the Solver Plotter class
    """
    def setup(self):
        self.num_pix = 25  # cutout pixel size
        delta_pix = 0.2
        background_rms = 0.05
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                   inverse=False, left_lower=False)
        # imaging data class
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': np.zeros((self.num_pix, self.num_pix)),
            'background_rms': background_rms,
            'noise_map': background_rms*np.ones((self.num_pix, self.num_pix)),
        }
        data_class = ImageData(**kwargs_data)

        # lens mass class
        lens_model_class = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

        # source light class
        source_model_class = LightModel(['STARLETS'])
        self.kwargs_source = [{'coeffs': 0, 'n_scales': 3, 'n_pixels': self.num_pix**2}]

        # init sparse solver
        self.solver = SparseSolverSource(data_class, lens_model_class, source_model_class,
                                         num_iter=10)

        # init the plotter
        self.plotter = SolverPlotter(self.solver, show_now=False)

    def test_plot_init(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        self.plotter.plot_init(image)

    def test_plot_step(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        iter_1, iter_2, iter_3 = 1, 2, 3
        self.plotter.plot_step(image, iter_1, iter_2=iter_2, iter_3=iter_3)

    def test_plot_final(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        self.plotter.plot_final(image)

    def test_plot_results(self):
        # launch solver
        _, _, _, _, _ = self.solver.solve(self.kwargs_lens, self.kwargs_source)
        self.plotter.plot_results(model_log_scale=True)
        # plt.show()
        plt.close()

    def test_quick_imshow(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        kwargs = {'cmap': 'bwr'}
        self.plotter.quick_imshow(image, title="test image", show_now=False, **kwargs)


if __name__ == '__main__':
    pytest.main()
