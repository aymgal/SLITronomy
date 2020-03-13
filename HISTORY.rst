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