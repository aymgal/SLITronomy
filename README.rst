**********************************************
SLITronomy - Sparse Linear Inversion Technique
**********************************************

.. image:: https://travis-ci.org/aymgal/SLITronomy.svg?branch=master
    :target: https://travis-ci.org/aymgal/SLITronomy
    :alt: Build status

.. image:: https://coveralls.io/repos/github/aymgal/SLITronomy/badge.svg
    :target: https://coveralls.io/github/aymgal/SLITronomy
    :alt: Coverage status

.. .. image:: https://codecov.io/gh/aymgal/SLITronomy/branch/master/graph/badge.svg
..   :target: https://codecov.io/gh/aymgal/SLITronomy

.. image:: https://badge.fury.io/py/slitronomy.svg
    :target: https://badge.fury.io/py/slitronomy
    :alt: PyPi status

.. image:: https://readthedocs.org/projects/slitronomy/badge/?version=latest
    :target: https://slitronomy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://badge.fury.io/py/slitronomy
    :alt: Python 3.6 support

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://badge.fury.io/py/slitronomy
    :alt: Python 3.7 support

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/aymgal/slitronomy/blob/master/LICENSE
    :alt: MIT license


Updated and improved version of the Sparse Lens Inversion Technique (`SLIT <https://github.com/herjy/SLIT>`_), developed within the framework of lens modelling software `lenstronomy <https://github.com/sibirrer/lenstronomy>`_.

The documentation is available on `readthedocs.org <http://slitronomy.readthedocs.org/>`_ (currently in development).

IMPORTANT : the current (beta) version of SLItronomy only works with the branch ``dev_slit-bis`` on `this fork <https://github.com/aymgal/lenstronomy/tree/dev_slit-bis>`_  of lenstronomy. In some rare cases (e.g. travis-ci builds) you may have issues with the latest stable release of PySAP. If this is the case, please use this `fork of pySAP <https://github.com/aymgal/pysap/tree/dev-aym>`_ switched to branch ``dev-aym``, instead.


Installation
============

This package is available through PyPi.

.. code-block:: bash

    $ pip install slitronomy


Example notebooks
=================

An ensemble of example notebooks are located at `thirdparty/notebooks <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks>`_ :

* `Pixelated image-to-source plane mapping <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/01_lensing_mapping.ipynb>`_
* `Starlets decomposition and reconstruction <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/02_starlets_decomposition.ipynb>`_
* `Sparse source reconstruction <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/03_complex_source_reconstruction.ipynb>`_
* `Lens model optimization <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/04_source_mass_reconstruction.ipynb>`_
* `Multi-band source reconstruction <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/05_multiband_source_reconstruction.ipynb>`_ (requires `MuSCADeT <https://github.com/aymgal/MuSCADeT>`_)
* `Sparse source and lens light reconstructions <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/06_complex_sourcelens_reconstruction.ipynb>`_
* `Sparse source reconstruction with quasar images <https://github.com/aymgal/SLITronomy/tree/master/thirdparty/notebooks/07_complex_quasar_source.ipynb>`_


References
==========

* SLITronomy paper: in prep.
* Sparse Linear Inversion Technique: `Joseph et al. 2019 <https://arxiv.org/abs/1809.09121>`_.
* lenstronomy modelling software: `Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v2>`_.
