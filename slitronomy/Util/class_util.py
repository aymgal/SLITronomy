__author__ = 'aymgal'

import numpy as np

from slitronomy.Optimization.solver_source import SparseSolverSource
from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens
from slitronomy.Optimization.solver_source_ps import SparseSolverSourcePS


def create_solver_class(data_class, psf_class, image_numerics_class, source_numerics_class, 
                        lens_model_class, source_model_class, lens_light_model_class, point_source_class, 
                        kwargs_sparse_solver):
    # detect which solver should be initialised, depending on model components
    lens_light_bool = (lens_light_model_class is not None and len(lens_light_model_class.profile_type_list) > 0)
    point_source_bool = (point_source_class is not None and len(point_source_class.point_source_type_list) > 0)

    # depending on the case, check if light profiles are supported and create the solver instance
    if lens_light_bool is False and point_source_bool is False:
        model_list = source_model_class.profile_type_list
        if len(model_list) != 1 or model_list[0] not in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2']:
            raise ValueError("'SLIT_STARLETS' or 'SLIT_STARLETS_GEN2' must be the only source model list for pixel-based modelling")
        solver_class = SparseSolverSource(data_class, lens_model_class, image_numerics_class, source_numerics_class,
                                          source_model_class, **kwargs_sparse_solver)
    elif point_source_bool is False:
        model_list = lens_light_model_class.profile_type_list
        if len(model_list) != 1 or model_list[0] not in ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2']:
            raise ValueError("'SLIT_STARLETS' or 'SLIT_STARLETS_GEN2' must be the only lens light model list for pixel-based modelling")
        solver_class = SparseSolverSourceLens(data_class, lens_model_class, image_numerics_class, source_numerics_class, 
                                              source_model_class, lens_light_model_class, **kwargs_sparse_solver)
    elif lens_light_bool is False:
        if not np.all(psf_class.psf_error_map == 0):
            print("WARNING : SparseSolverSourcePS does not support PSF error map for now !")
        solver_class = SparseSolverSourcePS(data_class, lens_model_class, image_numerics_class, source_numerics_class, 
                                            source_model_class, **kwargs_sparse_solver)
    return solver_class
