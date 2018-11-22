from nose.plugins.skip import SkipTest
from .scenarios.registry import registry
from nose.tools import assert_equal, assert_not_equal
from pyNN.utility import init_logging, assert_arrays_equal
import numpy

try:
    import pynn_genn
    have_genn = True
except ImportError:
    have_genn = False

try:
    import unittest2 as unittest
except ImportError:
    import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_scenarios():
    for scenario in registry:
        if "genn" not in scenario.exclude:
            scenario.description = "{}(genn)".format(scenario.__name__)

            # **HACK** work around bug in nose where names of tests don't get cached
            test_scenarios.compat_func_name = scenario.description

            if have_genn:
                yield scenario, pynn_genn
            else:
                raise SkipTest

if __name__ == '__main__':
    test_scenarios()