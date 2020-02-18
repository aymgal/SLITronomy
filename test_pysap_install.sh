#!/bin/bash


if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
    pip install PySide;
    # pip install nibabel;
    # pip install pyqtgraph;
    # pip install progressbar2;
    # pip install modopt;
    # pip install pybind11;
    pip install python-pysap;
fi

if [ "$TRAVIS_PYTHON_VERSION" == "3.6" ]; then
    mkdir -p $HOME/.local/share/pysap
    git clone https://github.com/CEA-COSMIC/pysap-data.git $HOME/.local/share/pysap/pysap-data
    ln -s $HOME/.local/share/pysap/pysap-data/pysap-data/* $HOME/.local/share/pysap
    rm $HOME/.local/share/pysap/__init__.py
    ls -l $HOME/.local/share/pysap
    pip install -b $TRAVIS_BUILD_DIR/build -t $TRAVIS_BUILD_DIR/install --no-clean --no-deps .
    ls $TRAVIS_BUILD_DIR/install
    ldd $TRAVIS_BUILD_DIR/install/pysparse.so
    export PYTHONPATH=$TRAVIS_BUILD_DIR/install:$PYTHONPATH
    export PATH=$PATH:$TRAVIS_BUILD_DIR/build/temp.linux-x86_64-3.6/extern/bin
fi