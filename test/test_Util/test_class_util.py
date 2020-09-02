__author__ = 'aymgal'

from slitronomy.Util import class_util

import numpy as np
import numpy.testing as npt
import pytest
import unittest

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.differential_extinction import DifferentialExtinction
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame
from lenstronomy.Util import util

np.random.seed(18)


class TestClassUtil(object):

    def setup(self):
        # data specifics
        sigma_bkg = .05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg, inverse=True)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)
        kernel = psf_class.kernel_point_source
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel, 'psf_error_map': np.ones_like(kernel) * 0.001}
        psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        kwargs_ps = [{'ra_source': 0.01, 'dec_source': 0.0,
                       'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'supersampling_factor': 2}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, 
                                lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, kwargs_lens, kwargs_source,
                                       kwargs_lens_light, kwargs_ps)
        data_class.update_data(image_sim)
    
        self.imageModel = imageModel
        self.kwargs_sparse_solver = {}

    def test_create_solver_class(self):
        from slitronomy.Optimization.solver_source import SparseSolverSource
        source_model_class = LightModel(['SLIT_STARLETS'])
        lens_light_model_class = LightModel([])
        point_source_class = PointSource(point_source_type_list=[])
        source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
        solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                      self.imageModel.ImageNumerics, source_numerics_class, 
                                                      self.imageModel.LensModel, 
                                                      source_model_class, lens_light_model_class, point_source_class,
                                                      self.imageModel._extinction, self.kwargs_sparse_solver)
        assert isinstance(solver_class, SparseSolverSource)

        from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens
        source_model_class = LightModel(['SLIT_STARLETS'])
        lens_light_model_class = LightModel(['SLIT_STARLETS'])
        point_source_class = PointSource(point_source_type_list=[])
        source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
        solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                      self.imageModel.ImageNumerics, source_numerics_class, 
                                                      self.imageModel.LensModel, 
                                                      source_model_class, lens_light_model_class, point_source_class,
                                                      self.imageModel._extinction, self.kwargs_sparse_solver)
        assert isinstance(solver_class, SparseSolverSourceLens)

        from slitronomy.Optimization.solver_source_ps import SparseSolverSourcePS
        source_model_class = LightModel(['SLIT_STARLETS'])
        lens_light_model_class = LightModel([])
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
        source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
        solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                      self.imageModel.ImageNumerics, source_numerics_class, 
                                                      self.imageModel.LensModel, 
                                                      source_model_class, lens_light_model_class, point_source_class,
                                                      self.imageModel._extinction, self.kwargs_sparse_solver)
        assert isinstance(solver_class, SparseSolverSourcePS)


class TestRaise(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRaise, self).__init__(*args, **kwargs)
        # data specifics
        sigma_bkg = .05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg, inverse=True)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)
        kernel = psf_class.kernel_point_source
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel, 'psf_error_map': np.ones_like(kernel) * 0.001}
        psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        kwargs_ps = [{'ra_source': 0.01, 'dec_source': 0.0,
                       'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'supersampling_factor': 2}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, 
                                lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, kwargs_lens, kwargs_source,
                                       kwargs_lens_light, kwargs_ps)
        data_class.update_data(image_sim)
    
        self.imageModel = imageModel
        self.kwargs_sparse_solver = {}

    def test_raise(self):
        with self.assertRaises(ValueError):
            source_model_class = LightModel(['SERSIC'])  # not supported
            lens_light_model_class = LightModel(['SLIT_STARLETS'])
            point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
            source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
            solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                          self.imageModel.ImageNumerics, source_numerics_class, 
                                                          self.imageModel.LensModel, 
                                                          source_model_class, lens_light_model_class, point_source_class,
                                                          self.imageModel._extinction, self.kwargs_sparse_solver)
        with self.assertRaises(ValueError):
            source_model_class = LightModel(['SLIT_STARLETS'])  # not supported
            lens_light_model_class = LightModel(['SERSIC'])
            point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
            source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
            solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                          self.imageModel.ImageNumerics, source_numerics_class, 
                                                          self.imageModel.LensModel, 
                                                          source_model_class, lens_light_model_class, point_source_class,
                                                          self.imageModel._extinction, self.kwargs_sparse_solver)
        with self.assertRaises(ValueError):
            source_model_class = LightModel(['SLIT_STARLETS', 'SLIT_STARLETS'])  # not supported
            lens_light_model_class = LightModel(['SLIT_STARLETS'])  # not supported
            point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
            source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
            solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                          self.imageModel.ImageNumerics, source_numerics_class, 
                                                          self.imageModel.LensModel, 
                                                          source_model_class, lens_light_model_class, point_source_class,
                                                          self.imageModel._extinction, self.kwargs_sparse_solver)
        with self.assertRaises(ValueError):
            source_model_class = LightModel(['SLIT_STARLETS'])
            lens_light_model_class = LightModel(['SLIT_STARLETS', 'SLIT_STARLETS'])  # not supported
            point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
            source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
            solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                          self.imageModel.ImageNumerics, source_numerics_class, 
                                                          self.imageModel.LensModel, 
                                                          source_model_class, lens_light_model_class, point_source_class,
                                                          self.imageModel._extinction, self.kwargs_sparse_solver)
        with self.assertRaises(ValueError):
            source_model_class = LightModel([])  # not supported
            lens_light_model_class = LightModel(['SLIT_STARLETS'])
            point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
            source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
            solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                          self.imageModel.ImageNumerics, source_numerics_class, 
                                                          self.imageModel.LensModel, 
                                                          source_model_class, lens_light_model_class, point_source_class,
                                                          self.imageModel._extinction, self.kwargs_sparse_solver)
        with self.assertRaises(NotImplementedError):
            # ask for source + lens light + point sources: not supported
            source_model_class = LightModel(['SLIT_STARLETS'])
            lens_light_model_class = LightModel(['SLIT_STARLETS'])
            point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'])
            source_numerics_class = NumericsSubFrame(self.imageModel.Data, self.imageModel.PSF, supersampling_factor=2)
            solver_class = class_util.create_solver_class(self.imageModel.Data, self.imageModel.PSF, 
                                                          self.imageModel.ImageNumerics, source_numerics_class, 
                                                          self.imageModel.LensModel, 
                                                          source_model_class, lens_light_model_class, point_source_class,
                                                          self.imageModel._extinction, self.kwargs_sparse_solver)

if __name__ == '__main__':
    pytest.main()
