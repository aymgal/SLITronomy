**********
SLITronomy
**********

.. image:: https://travis-ci.org/aymgal/SLITronomy.svg?branch=master
    :target: https://travis-ci.org/aymgal/SLITronomy

.. image:: https://coveralls.io/repos/github/aymgal/SLITronomy/badge.svg
    :target: https://coveralls.io/github/aymgal/SLITronomy

.. .. image:: https://codecov.io/gh/aymgal/SLITronomy/branch/master/graph/badge.svg
..   :target: https://codecov.io/gh/aymgal/SLITronomy

.. image:: https://badge.fury.io/py/slitronomy.svg
    :target: https://badge.fury.io/py/slitronomy

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://badge.fury.io/py/slitronomy

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://badge.fury.io/py/slitronomy

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/aymgal/slitronomy/blob/master/LICENSE


Updated and improved version of the Sparse Lens Inversion Technique (SLIT), developed within the framework of lens modelling software lenstronomy.

IMPORTANT : the current version of SLItronomy only works with the branch ``dev_algo-slit`` on `this fork <https://github.com/aymgal/lenstronomy/tree/dev_slit-algo>`_  of lenstronomy. In some rare cases (e.g. travis-ci builds), it may also requires the following `fork of pySAP <https://github.com/aymgal/pysap/tree/dev-aym>`_, switched to branch ``dev-aym``.


Installation
++++++++++++

This package is available through PyPi.

.. code-block:: bash

    $ pip install slitronomy


Links
+++++

Original lenstronomy : `lenstronomy <https://github.com/sibirrer/lenstronomy>`_

Original SLIT : `SLIT <https://github.com/herjy/SLIT>`_

Example notebooks
+++++++++++++++++

An ensemble of example notebooks are located at `thirdparty/notebooks <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks>`_ :

* `Pixelated image-to-source plane mapping (nearest-neighbor interpolation) <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/01_lensing_mapping.ipynb>`_
* `Pixelated image-to-source plane mapping (bilinear interpolation) <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/01bis_lensing_mapping_interpol.ipynb>`_
* `Starlets decomposition and reconstruction <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/02_starlets_decomposition.ipynb>`_
* `Sparse source reconstruction <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/03_complex_source_reconstruction.ipynb>`_
* `Lens model optimization <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/04_source_mass_reconstruction.ipynb>`_
* `Multi-band source reconstruction <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/05_multiband_source_reconstruction.ipynb>`_ (requires MuSCADeT)
* `Sparse source and lens light reconstructions <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/06_complex_sourcelens_reconstruction.ipynb>`_
* `Sparse source reconstruction with quasar images <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/07_complex_quasar_source.ipynb>`_

If required, install the fork of Multi-band morpho-Spectral Component Analysis Deblending Tool : `MuSCADeT <https://github.com/aymgal/MuSCADeT>`_

