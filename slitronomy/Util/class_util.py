__author__ = 'aymgal'

import numpy as np


_support_profiles = ['SLIT_STARLETS', 'SLIT_STARLETS_GEN2']


def create_solver_class(data_class, psf_class, image_numerics_class, source_numerics_class, 
                        lens_model_class, source_model_class, lens_light_model_class, point_source_class, 
                        extinction_class, kwargs_sparse_solver):
    """
    From lenstronomy model classes, returns the appropriate SLITronomy solver class.
    Raises errors if the required configuration is not supported by SLITronomy.
    """
    # check source model
    model_list = source_model_class.profile_type_list
    if len(model_list) == 0:
        source_light_bool = False
    elif len(model_list) != 1:
        raise ValueError("There must be the a single source profile in the list for pixel-based modelling")
    elif model_list[0] not in _support_profiles:
        raise ValueError("Only {} are supported pixel-based light profiles".format(_support_profiles))
    else:
        source_light_bool = True

    # check lens light model
    model_list = lens_light_model_class.profile_type_list
    if len(model_list) == 0:
        lens_light_bool = False
    elif len(model_list) != 1:
        raise ValueError("There must be the a single lens light profile in the list for pixel-based modelling")
    elif model_list[0] not in _support_profiles:
        raise ValueError("Only {} are supported pixel-based light profiles".format(_support_profiles))
    else:
        lens_light_bool = True

    # check point source model
    model_list = point_source_class.point_source_type_list
    point_source_bool = (len(model_list) > 0)

    # check extinction model
    if extinction_class.compute_bool is True:
        raise ValueError("Differential extinction is not yet supported by the SLITronomy solver")

    # depending on the configuration, create the right solver instance
    if source_light_bool is True and point_source_bool is False and lens_light_bool is False:
        from slitronomy.Optimization.solver_source import SparseSolverSource
        solver_class = SparseSolverSource(data_class, lens_model_class, image_numerics_class, source_numerics_class,
                                          source_model_class, **kwargs_sparse_solver)
    
    elif source_light_bool is True and point_source_bool is False and lens_light_bool is True:
        from slitronomy.Optimization.solver_source_lens import SparseSolverSourceLens
        solver_class = SparseSolverSourceLens(data_class, lens_model_class, image_numerics_class, source_numerics_class, 
                                              source_model_class, lens_light_model_class, **kwargs_sparse_solver)
    
    elif source_light_bool is True and point_source_bool is True and lens_light_bool is False:
        from slitronomy.Optimization.solver_source_ps import SparseSolverSourcePS
        solver_class = SparseSolverSourcePS(data_class, lens_model_class, image_numerics_class, source_numerics_class, 
                                            source_model_class, **kwargs_sparse_solver)

    elif source_light_bool is False and point_source_bool is False and lens_light_bool is True:
        # Warning: in SLITronomy, modelling the lens light only uses the 'unlensed source' solver
        # which is boils down to simple deconvolution + denoising problem
        from slitronomy.Optimization.solver_lens import SparseSolverLens
        solver_class = SparseSolverLens(data_class, image_numerics_class, lens_light_model_class,
                                        lens_model_class, source_model_class, source_numerics_class,
                                        **kwargs_sparse_solver)
    
    else:
        raise NotImplementedError("SLITronomy solver for pixel-based modelling of source + lens light + point sources has not been implemented")

    return solver_class
