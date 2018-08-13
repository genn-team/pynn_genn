#A PyNN interface to GeNN
This project bridges [GeNN](http://genn-team.github.io/genn/), a GPU enhanced Neuronal Network simulation environment, with [PyNN], a simulator-independent language for building neuronal network models.

## Getting Started:

 - install [PyNN](https://pypi.org/project/PyNN/)
 - get GeNN from [python_wrapper](https://github.com/genn-team/genn/tree/python_wrapper) branch
 - install [SWIG](http://www.swig.org/)
 - `cd genn/lib`
 - compile using `make -f GNUMakefile pygenn` command
 - add `genn/lib/pygenn` folder to the `$PYTHONPATH` variable
 - clone this project
