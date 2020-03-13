__author__ = 'aymgal'

import matplotlib
matplotlib.use('agg')

import pytest
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt

from slitronomy.Util import util
from slitronomy.Util.solver_plotter import SolverPlotter
from slitronomy.Optimization.solver_source import SparseSolverSource

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import lenstronomy.Util.util as l_util


np.random.seed(18)

class TestSolverPlotter(object):
    """
    tests the Solver Plotter class
    """
    def setup(self):
        self.num_pix = 20  # cutout pixel size
        delta_pix = 0.2
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ \
            = l_util.make_grid_with_coordtransform(numPix=self.num_pix, deltapix=delta_pix, subgrid_res=1, 
                                                   inverse=False, left_lower=False)
        # imaging data class
        gaussian = Gaussian()
        x, y = l_util.make_grid(self.num_pix, 1)
        gaussian1 = gaussian.function(x, y, amp=5, sigma=1, center_x=-7, center_y=-7)
        gaussian2 = gaussian.function(x, y, amp=20, sigma=2, center_x=-3, center_y=-3)
        gaussian3 = gaussian.function(x, y, amp=60, sigma=4, center_x=+5, center_y=+5)
        image_data = util.array2image(gaussian1 + gaussian2 + gaussian3)
        background_rms = 0.1
        image_data += background_rms * np.random.randn(self.num_pix, self.num_pix) 
        kwargs_data = {
            'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 
            'transform_pix2angle': Mpix2coord,
            'image_data': image_data,
            'background_rms': background_rms,
            'noise_map': background_rms*np.ones_like(image_data),
        }
        data_class = ImageData(**kwargs_data)

        # lens mass class
        lens_model_class = LensModel(['SPEP'])
        self.kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0, 'e1': -0.05, 'e2': 0.05}]

        # source light class
        source_model_class = LightModel(['STARLETS'])
        self.kwargs_source = [{'coeffs': 0, 'n_scales': 3, 'n_pixels': self.num_pix**2}]

        numerics_class = NumericsSubFrame(pixel_grid=data_class, psf=PSF(psf_type='NONE'))

        # init sparse solver
        self.solver = SparseSolverSource(data_class, lens_model_class, source_model_class, numerics_class,
                                         num_iter_source=10)

        # init the plotter
        self.plotter = SolverPlotter(self.solver, show_now=False)

    def test_plot_init(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        self.plotter.plot_init(image)
        plt.close()

    def test_plot_step(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        iter_1, iter_2, iter_3 = 1, 2, 3
        self.plotter.plot_step(image, iter_1, iter_2=iter_2, iter_3=iter_3)
        plt.close()

    def test_plot_final(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        self.plotter.plot_final(image)
        plt.close()

    def test_plot_results(self):
        # launch solver first
        _, _, _ = self.solver.solve(self.kwargs_lens, self.kwargs_source)
        self.plotter.plot_results(model_log_scale=True)
        plt.close()

    def test_quick_imshow(self):
        image = np.random.rand(self.num_pix, self.num_pix)
        kwargs = {'cmap': 'bwr'}
        self.plotter.quick_imshow(image, title="test image", show_now=False, **kwargs)
        plt.close()

if __name__ == '__main__':
    pytest.main()
