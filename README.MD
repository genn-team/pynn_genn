[![Build Status](https://gen-ci.inf.sussex.ac.uk/buildStatus/icon?job=GeNN/pynn_genn/master)](https://gen-ci.inf.sussex.ac.uk/job/GeNN/job/pynn_genn/job/master/) [![codecov](https://codecov.io/gh/genn-team/pynn_genn/branch/master/graph/badge.svg)](https://codecov.io/gh/genn-team/pynn_genn)
# A PyNN interface to GeNN
This project bridges [GeNN](http://genn-team.github.io/genn/), a GPU enhanced Neuronal Network simulation environment, with [PyNN](https://pypi.org/project/PyNN/), a simulator-independent language for building neuronal network models.

## Getting Started:
### Installing PyGeNN from source (Windows is currently not supported)
 - Clone the ``master`` branch of GeNN (https://github.com/genn-team/genn)
 - Ensure that the ``GENN_PATH`` environment variable to point to the GeNN directory.
 - Ffrom the GeNN directory, build GeNN as a dynamic library using ``make -f lib/GNUMakefileLibGeNN DYNAMIC=1 LIBGENN_PATH=pygenn/genn_wrapper/`` (you will need to add ``CPU_ONLY=1`` if you do not have an NVIDIA GPU)
 - On Mac OS X only, set your newly created library's name with ``install_name_tool -id "@loader_path/libgenn_DYNAMIC.dylib" pygenn/genn_wrapper/libgenn_DYNAMIC.dylib`` (you will need to replace ``libgenn_DYNAMIC`` with ``libgenn_CPU_ONLY_DYNAMIC`` if you do not have an NVIDIA GPU)
 - Install with setup tools using ``python setup.py develop`` command
 
### Installing the PyNN backend for GeNN
 - Clone this project
 - Install with setuptools using ``python setup.py develop`` command

## Bugs
- PyNN GeNN is still under development - current results of running PyNN system tests are outlined in [#8](https://github.com/genn-team/pynn_genn/issues/8) with links to outstanding issues causing test failures.
