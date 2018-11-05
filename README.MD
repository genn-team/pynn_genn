# A PyNN interface to GeNN
This project bridges [GeNN](http://genn-team.github.io/genn/), a GPU enhanced Neuronal Network simulation environment, with [PyNN](https://pypi.org/project/PyNN/), a simulator-independent language for building neuronal network models.

## Getting Started:
### Installing PyGeNN
 - clone GeNN (https://github.com/genn-team/genn)
 - **TEMPORARILY** checkout the ``python_wrapper_update_take_2`` branch
 - set the ``GENN_PATH`` environment variable to point to the GeNN directory.
 - from the GeNN directory, build GeNN as a dynamic library using ``make -f lib/GNUMakefileLibGeNN DYNAMIC=1 LIBGENN_PATH=pygenn/genn_wrapper/``
 - install with setup tools using ``python setup.py install`` command
 
### Installing the PyNN backend for GeNN
 - clone this project
 - install with setuptools using ``python setup.py install`` command
