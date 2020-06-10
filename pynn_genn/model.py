import sys
import string
from collections import namedtuple
from functools import partial
from itertools import groupby
from copy import deepcopy
from string import Template
from lazyarray import larray
import numpy as np
from pygenn.genn_model import (create_custom_neuron_class,
                               create_custom_postsynaptic_class,
                               create_custom_current_source_class,
                               create_custom_weight_update_class,
                               init_var, init_connectivity)

from pygenn import genn_wrapper
from pyNN.standardmodels import (StandardModelType,
                                 StandardCellType,
                                 StandardCurrentSource,
                                 build_translations)
from pynn_genn.random import NativeRNG, RandomDistribution
from pyNN.parameters import LazyArray

from six import iteritems, iterkeys
import copy

# Mapping from GeNN to numpy types
genn_to_numpy_types = {
    "scalar": np.float32,
    "float": np.float32,
    "double": np.float64,
    "unsigned char": np.uint8,
    "int": np.int32,
    "unsigned int": np.uint32}


class DDTemplate(Template):
    """Template string class with the delimiter overridden with double $"""
    delimiter = "$$"


# Tuple type used to store GeNN model defintions
GeNNDefinitions = namedtuple("GeNNDefinitions",
                             ["definitions", "translations",
                              "extra_param_values"])

def sanitize_label(label):
    # Strip out any non-alphanumerical characters
    clean_label = "".join(c for c in label if c.isalnum() or c == '_')
    clean_label = clean_label.lstrip(string.digits)

    # If this is Python 2, convert unicode-encoded label to ASCII
    if sys.version_info < (3, 0):
        clean_label = clean_label.encode("ascii", "ignore")

    # Strip out any non-alphanumerical characters
    return clean_label


class GeNNStandardModelType(StandardModelType):

    genn_extra_parameters = {}

    def build_genn_model(self, defs, native_params, init_vals,
                         create_custom_model, prefix=""):

        # Take a deep copy of the definitions
        genn_defs = deepcopy(defs.definitions)

        # Check that it doesn't already have its
        # variables and parameters seperated out
        assert "param_names" not in genn_defs

        # Start with variables that definitions say MUST be variables
        var_name_types = genn_defs["var_name_types"]

        # **NOTE** all parameters are by default GeNN parameters
        param_name_types = genn_defs["param_name_types"]
        del genn_defs["param_name_types"]

        # Loop through native parameters
        prefix_len = len(prefix)
        for field_name, param in native_params.items():
            field_name_no_prefix = field_name[prefix_len:]

            # If parameter is NOT homogeneous, has correct prefix
            # and, without the prefix, it is in parameters
            if (not param.is_homogeneous and field_name.startswith(prefix) and
                field_name_no_prefix in param_name_types):
                # Add it to variables
                var_name_types.append((field_name_no_prefix,
                                       param_name_types[field_name_no_prefix]))

                # Remove from parameters
                del param_name_types[field_name_no_prefix]

        # Loop through any extra parameters provided by model
        extra_param_values = {}
        for param_name, param_val in iteritems(defs.extra_param_values):
            param_name_no_prefix = param_name[prefix_len:]

            # If prefix matches
            if param_name.startswith(prefix):
                # If extra parameter is callable
                if callable(param_val):
                    # Evaluate it
                    # **NOTE** this is basically a way of providing
                    # more GeNN parameters than there are PyNN parameters
                    ini = deepcopy(param_val(**self.parameter_space))
                # Otherwise create larray directly from value
                else:
                    ini = larray(deepcopy(param_val))

                # Add value to dictionary
                ini.shape = native_params.shape
                extra_param_values[param_name] = ini

                # If parameter is not homogeneous
                if not ini.is_homogeneous:
                    # Add it to variables
                    var_name_types.append(
                        (param_name_no_prefix,
                         param_name_types[param_name_no_prefix]))

                    # Remove from parameters
                    del param_name_types[param_name_no_prefix]

        # Set parameter names in defs
        genn_defs["param_names"] = list(param_name_types.keys())

        # Create custom model
        genn_model = create_custom_model(**genn_defs)

        # Get set of native parameter names
        native_param_keys = set(native_params.keys())

        # Build dictionary mapping from the GeNN name
        # of state variables to their PyNN name
        init_val_lookup = {self.translations[n]["translated_name"]: n
                           for n, v in init_vals.items()
                           if n in self.translations}

        # Loop through GeNN parameters
        neuron_params = {}
        for n in genn_defs["param_names"]:
            pref_n = prefix + n

            # If this variable is a native parameter,
            # evaluate it into neuron_params and simplify
            if pref_n in native_param_keys:
                neuron_params[n] = self._init_variable(False, native_params[pref_n])
            elif pref_n in extra_param_values:
                neuron_params[n] = self._init_variable(False, extra_param_values[pref_n])
            else:
                raise Exception("Property '{}' not "
                                "correctly initialised".format(n))

        # Loop through GeNN variables
        neuron_ini = {}
        for n, t in var_name_types:
            pref_n = prefix + n

            # If this variable is a native parameter,
            # evaluate it into neuron_ini
            if pref_n in native_param_keys:
                neuron_ini[n] = self._init_variable(True, native_params[pref_n])
            # Otherwise if there is an initial value associated with it
            elif pref_n in init_val_lookup:
                # Get its PyNN name from the lookup
                pynn_n = init_val_lookup[pref_n]
                neuron_ini[n] = self._init_variable(True, init_vals[pynn_n])
            # Otherwise if this variable is manually initialised
            elif pref_n in extra_param_values:
                # Set data type
                extra_param_values[pref_n].dtype = (
                    genn_to_numpy_types[t] if t in genn_to_numpy_types else np.float32)
                # Evaluate values into neuron initialiser
                neuron_ini[n] = self._init_variable(True, extra_param_values[pref_n])
            else:
                raise Exception("Variable '{}' not "
                                "correctly initialised".format(n))

        return genn_model, neuron_params, neuron_ini

    def _init_variable(self, is_var_name_type, param):
        if (isinstance(param.base_value, RandomDistribution) and
           isinstance(param.base_value.rng, NativeRNG) and
           not len(param.operations)):
            # if we need (random) on-device initialization we should use
            # PyNN GeNN NativeRNG
            # NOTE: if this parameter needs to be transformed (operations > 0),
            #       initialize on host
            params = copy.copy(param.base_value.parameters)

            rng = param.base_value.rng
            dist_name = param.base_value.name
            param_init = rng.init_var_snippet(dist_name, params)
            return init_var(param_init, params)
        elif param.is_homogeneous:
            # if param is a constant send a scalar to device
            if param.shape is None:
                param.shape = (1,)
            simplify = not is_var_name_type
            return param.evaluate(simplify=simplify)
        else:
            return param.evaluate(simplify=False)


class GeNNStandardCellType(GeNNStandardModelType, StandardCellType):

    def __init__(self, **parameters):
        super(GeNNStandardCellType, self).__init__(**parameters)

        # If cell has a postsynaptic model i.e. it's not a spike source
        if hasattr(self, "postsyn_defs"):
            # Build translations from both translation tuples
            self.translations = build_translations(
                *(self.neuron_defs.translations + self.postsyn_defs.translations))
        # Otherwise, build translations from neuron translations
        else:
            self.translations =\
                build_translations(*self.neuron_defs.translations)

    def get_extra_global_neuron_params(self, native_params, init_vals):
        return {}

    def build_genn_psm(self, native_params, init_vals, prefix):
        # Build callable to create a custom PSM from defs
        creator = partial(create_custom_postsynaptic_class,
                          self.genn_postsyn_name)

        # Build model
        return self.build_genn_model(self.postsyn_defs, native_params,
                                     init_vals, creator, prefix)

    def build_genn_neuron(self, native_params, init_vals):
        # Build callable to create a custom neuron model from defs
        creator = partial(create_custom_neuron_class, self.genn_neuron_name)

        # Build model
        return self.build_genn_model(self.neuron_defs, native_params,
                                     init_vals, creator)


class GeNNStandardSynapseType(GeNNStandardModelType):
    def build_genn_wum(self, conn_params, init_vals):
        # Take a deep copy of the definitions
        genn_defs = deepcopy(self.wum_defs)

        # Check that it doesn't already have its
        # variables and parameters separated out
        assert "param_names" not in genn_defs
        assert "var_name_types" not in genn_defs

        # Extract variables from copy of defs and remove
        # **NOTE** all vars are by default GeNN variables
        vars = genn_defs["vars"]
        del genn_defs["vars"]

        # extract the connector from the connectivity parameters
        conn = conn_params.pop('connector')

        # Start with an empty list of parameters
        param_names = []

        # Get set of forcibly mutable vars if synapse type has one
        mutable_vars = (self.mutable_vars
                        if hasattr(self, "mutable_vars")
                        else set())

        # Loop through connection parameters
        for n, p in iteritems(conn_params):
            # If this parameter is in the variable dictionary,
            # but it is homogenous
            if (not isinstance(p, LazyArray) and n in vars and
                    n not in mutable_vars and np.allclose(p, p[0])):
                # Remove from vars
                del vars[n]

                # Add it to params
                param_names.append(n)

        # Copy updated vars and parameters back into defs
        genn_defs["var_name_types"] = vars.items()
        genn_defs["param_names"] = param_names

        # Create custom model
        genn_model = create_custom_weight_update_class(self.__class__.__name__,
                                                       **genn_defs)

        # Use first entry in conn param for parameters
        wum_params = {n: conn_params[n][0]
                      for n in genn_defs["param_names"]}

        # Loop through GeNN variables
        wum_init = {}
        for n, t in iteritems(vars):
            # Get type to use for variable
            var_type = (genn_to_numpy_types[t] if t in genn_to_numpy_types
                        else np.float32)

            # If this variable is set by connection parameters,
            # use these as initial values
            if n in conn_params:
                wum_init[n] = conn_params[n].astype(var_type, copy=False)
            # Otherwise, if there is a default in the model, use that
            elif n in self.default_initial_values:
                wum_init[n] = self.default_initial_values[n]
            elif n in conn.on_device_init_params and conn.on_device_init:
                # if the parameter is to be initialized on device
                wum_init[n] = self._init_variable(n, conn.on_device_init_params[n])
            else:
                raise Exception("Variable '{}' not "
                                "correctly initialised".format(n))
        # Zero all presynaptic variables
        # **TODO** other means of initialisation
        wum_pre_init = (None
                        if "pre_var_name_types" not in genn_defs
                        else {n[0]: 0.0
                              for n in genn_defs["pre_var_name_types"]})

        # Zero all postsynaptic variables
        # **TODO** other means of initialisation
        wum_post_init = (None
                         if "post_var_name_types" not in genn_defs
                         else {n[0]: 0.0
                               for n in genn_defs["post_var_name_types"]})

        return genn_model, wum_params, wum_init, wum_pre_init, wum_post_init


class GeNNStandardCurrentSource(GeNNStandardModelType, StandardCurrentSource):
    def __init__(self, **parameters):
        super(GeNNStandardCurrentSource, self).__init__(**parameters)
        self.translations = build_translations(
            *self.currentsource_defs.translations)

    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        for pop, cs in groupby(cells, key=lambda c: c.parent):
            pop._injected_currents.append(
                    ("%s_%s_%u" % (pop._genn_label, self.__class__.__name__,
                                   pop._simulator.state.num_current_sources),
                     self,
                     list(cs)))
            pop._simulator.state.num_current_sources += 1

    def get_extra_global_params(self, native_params):
        return {}

    def build_genn_current_source(self, native_params):
        # Build callable to create a custom current source from defs
        creator = partial(create_custom_current_source_class,
                          self.genn_currentsource_name)

        # Build model
        return self.build_genn_model(self.currentsource_defs, native_params,
                                     {}, creator)

