#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

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


#readme = open('README.rst').read()

#history = open('HISTORY.rst').read().replace('.. :changelog:', '')

desc = open("README.rst").read()
requires = ['configparser']
tests_require=['pytest>=2.3', 'mock']

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))


setup(
    name='slitronomy',
    version='0.3.2',
    description='Sparse Linear Inversion Technique for lenstronomy',
    long_description=desc,
    author='Aymeric Galan',
    author_email='aymeric.galan@gmail.com',
    url='https://github.com/aymgal/SLITronomy',
    download_url='https://github.com/aymgal/slitronomy/archive/0.3.2.tar.gz.zip',
    packages=find_packages(PACKAGE_PATH, exclude=['thirdparty', 'test']),
    package_dir={'slitronomy': 'slitronomy'},
    include_package_data=True,
    install_requires=requires,
    license='MIT',
    zip_safe=False,
    keywords='slitronomy',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    tests_require=tests_require,
    python_requires='>=3.6',
    cmdclass={'test': PyTest},
)
