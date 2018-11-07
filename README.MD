# A PyNN interface to GeNN
This project bridges [GeNN](http://genn-team.github.io/genn/), a GPU enhanced Neuronal Network simulation environment, with [PyNN](https://pypi.org/project/PyNN/), a simulator-independent language for building neuronal network models.

## Getting Started:
### Installing PyGeNN from source
 - clone GeNN (https://github.com/genn-team/genn)
 - **TEMPORARILY** checkout the ``python_wrapper`` branch
 - set the ``GENN_PATH`` environment variable to point to the GeNN directory.
 - from the GeNN directory, build GeNN as a dynamic library using ``make -f lib/GNUMakefileLibGeNN DYNAMIC=1 LIBGENN_PATH=pygenn/genn_wrapper/`` (you will need to add ``CPU_ONLY=1`` if you do not have an NVIDIA GPU)
 - on Mac OS X only, set your newly created library's name with ``install_name_tool -id "@loader_path/libgenn_DYNAMIC.dylib" pygenn/genn_wrapper/libgenn_DYNAMIC.dylib`` (you will need to replace ``libgenn_DYNAMIC`` with ``libgenn_CPU_ONLY_DYNAMIC`` if you do not have an NVIDIA GPU)
 - install with setup tools using ``python setup.py install`` command
 
### Installing the PyNN backend for GeNN
 - clone this project
 - install with setuptools using ``python setup.py install`` command
