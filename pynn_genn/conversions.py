
"""
Conversion functions to GeNN-compatible data types.
"""

import numpy
from pyNN.parameters import Sequence
from pyNN.core import iteritems


def convert_to_single(value):
    if isinstance(value, Sequence):
        return_value = value.value
    elif isinstance(value, numpy.ndarray):
        if value.dtype == object and isinstance(value[0], Sequence):
            assert value.shape == (1,), "GeNN expects 1 dimensional arrays"
            return_value = value[0].value
        elif value.shape == (1,):
            return_value = value[0]
        elif numpy.allclose(value, value[0]):
            return_value = value[0]
        else:
            raise ValueError('Expected a sigle value, but got {}'.format(value))
    elif isinstance(value, list):
        assert numpy.allclose(value, value[0])
        return_value = convert_to_single(value[0])
    else:
        return_value = value

    return return_value


def convert_to_array(container):
    """
    Makes sure container only contains datatypes understood by GeNN.

    container can be scalar, a list or a dict.
    """

    compatible = None
    if isinstance(container, list):
        compatible = []
        for value in container:
            compatible.append(convert_to_single(value))

    elif isinstance(container, dict):
        compatible = {}

        for k, v in iteritems(container):
            compatible[k] = convert_to_single(v)
    
    elif isinstance(container, numpy.ndarray):
        compatible = []
        for value in container:
            compatible.append(convert_to_single(value))

    else:
        compatible = convert_to_single(container)

    return compatible


def convert_init_values(init_values_dict):
    """Makes initial values dict to GeNN compatible"""

    compatible = {}
    for k, v in init_values_dict.items():
        compatible[k] = convert_to_array(v)
    return compatible
