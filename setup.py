#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


readme = open('README.rst').read()

history = open('HISTORY.rst').read().replace('.. :changelog:', '')

desc = open("README.rst").read()
requires = ['numpy>=1.13', 'scipy>=0.14.0', "configparser"]
tests_require=['pytest>=2.3', "mock"]

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name='slitronomy',
    version='0.0.1',
    description='Sparse Linear Inversion Technique for lenstronomy',
    long_description=desc,
    author='Aymeric Galan',
    author_email='aymeric.galan@gmail.com',
    url='https://github.com/aymgal/SLITronomy',
    download_url='https://github.com/aymgal/slitronomy/archive/master.zip',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'slitronomy': 'slitronomy'},
    include_package_data=True,
    #setup_requires=requires,
    install_requires=requires,
    license='MIT',
    zip_safe=False,
    keywords='slitronomy',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
    ],
    tests_require=tests_require,
    cmdclass={'test': PyTest},#'build_ext':build_ext,
)
