.. :changelog:

History
-------

0.1.0 (2020-02-19)
++++++++++++++++++

* First release on PyPI.

0.1.1 (2020-03-03)
++++++++++++++++++

* Fix issue when using non-zero grid offset in LensingOperatorInterpol
* Fix loss function definition
* Minor changes
  
0.2.0 (2020-03-13)
++++++++++++++++++

* Faster LensingOperatorInterpol thanks to a new optimized implementation (credits to @austinpeel, thanks!)
* Add new SparseSolverSourcePS that supports point source amplitude optimization (through lenstronomy's method)
* Add new example notebooks (see thirdparty/notebooks/)
* Fix L1-norm reweighting scheme in solvers
* Vast design improvements
* Various bug fixes

0.3.0 (2020-08-30)
++++++++++++++++++

LOTS of improvements on various aspects:
* Unified LensingOperator, that supports 'legacy' implementation of nearest-neighbors interpolation, and faster implementations of nearest-neighbors and bilinear interpolations (credits to @austinpeel for fast versions)
* Improved adaptive initialisation and refinement of regularisation strength
* Improved noise propagation to source plane and starlet space, with gaussian filtering for better estimation of unconstrained source pixels
* Basic support of image plane supersampling (only nearest-neighbors interpolation for upsampling)
* User can now provide a mask to select pixel that belongs only to lens light, for improving masking of source plane region. Use it with care!
* Better integration with lenstronomy classes

0.3.1 (2020-09-07)
++++++++++++++++++

* Stable version release
* Finalised integration to lenstronomy, as an optional package for pixel-based modelling
