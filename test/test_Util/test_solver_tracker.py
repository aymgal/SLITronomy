__author__ = 'aymgal'

from slitronomy.Util.solver_tracker import SolverTracker
from slitronomy.Optimization.solver_source import SparseSolverSource

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
import lenstronomy.Util.util as l_util

import numpy as np
import numpy.testing as npt
import pytest


class TestSolverTracker(object):
    """
    tests the Solver Tracker class
    """
    def setup(self):
        self.num_pix = 20  # cutout pixel size
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
        source_model_class = LightModel(['SLIT_STARLETS'])
        self.kwargs_source = [{'coeffs': 0, 'n_scales': 3, 'n_pixels': self.num_pix**2}]

        # define numerics classes
        image_numerics_class = NumericsSubFrame(pixel_grid=data_class, psf=PSF(psf_type='NONE'))
        source_numerics_class = NumericsSubFrame(pixel_grid=data_class, psf=PSF(psf_type='NONE'), supersampling_factor=1)

        # init sparse solver
        self.solver = SparseSolverSource(data_class, lens_model_class, image_numerics_class, source_numerics_class,
                                         source_model_class, num_iter_source=10)

        # init the tracker
        self.tracker_alone = SolverTracker(self.solver, verbose=True)

    def test_track_with_solver(self):
        track_before = self.solver.track
        assert track_before is None  # before solver has been ran
        # launch solver
        _ = self.solver.solve(self.kwargs_lens, self.kwargs_source)
        track_after = self.solver.track
        assert isinstance(track_after, dict)
        assert len(track_after['loss'][0]) > 1
        assert len(track_after['red_chi2'][0]) > 1
        assert len(track_after['step_diff'][0]) > 1

    def test_init2finalize(self):
        assert self.tracker_alone.track is None
        self.tracker_alone.init()
        assert isinstance(self.tracker_alone.track, dict)
        self.tracker_alone.save(S=None, S_next=None, HG=None, HG_next=None)
        for key in self.tracker_alone.track:
            assert np.isnan(self.tracker_alone.track[key][0][-1])
            assert np.isnan(self.tracker_alone.track[key][1][-1])
        self.tracker_alone.finalize()
        for key in self.tracker_alone.track:
            assert isinstance(self.tracker_alone.track[key], np.ndarray)

if __name__ == '__main__':
    pytest.main()
