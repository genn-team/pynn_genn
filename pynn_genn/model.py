import sys
import string
from collections import namedtuple
from functools import partial
from itertools import groupby
from copy import deepcopy
from string import Template
from lazyarray import larray
import numpy as np
from pygenn import (create_neuron_model,
                    create_postsynaptic_model,
                    create_current_source_model,
                    create_weight_update_model,
                    init_var, init_sparse_connectivity)

from pyNN.standardmodels import (StandardModelType,
                                 StandardCellType,
                                 StandardCurrentSource,
                                 build_translations)
from pynn_genn.random import NativeRNG, RandomDistribution
from pyNN.parameters import LazyArray

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
    
    def __init__(self, **parameters):
        super().__init__(**parameters)
        self._var_init_snippets = []
    
    def build_genn_model(self, defs, native_params, init_vals,
                         create_custom_model, prefix=""):

        # Take a deep copy of the definitions
        genn_defs = deepcopy(defs.definitions)

        # Start with variables that definitions say MUST be variables
        vars = genn_defs["vars"]

        # **NOTE** all parameters are by default GeNN parameters
        params = genn_defs["params"]
        del genn_defs["params"]

        # Loop through native parameters
        prefix_len = len(prefix)
        for field_name, param in native_params.items():
            field_name_no_prefix = field_name[prefix_len:]

            # If parameter is NOT homogeneous, has correct prefix
            # and, without the prefix, it is in parameters
            if (not param.is_homogeneous and field_name.startswith(prefix) and
                field_name_no_prefix in params):
                # Add it to variables
                vars.append((field_name_no_prefix,
                             params[field_name_no_prefix]))

                # Remove from parameters
                del params[field_name_no_prefix]

        # Loop through any extra parameters provided by model
        extra_param_values = {}
        for param_name, param_val in defs.extra_param_values.items():
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
                    vars.append(
                        (param_name_no_prefix,
                         params[param_name_no_prefix]))

                    # Remove from parameters
                    del params[param_name_no_prefix]

        # Set parameter names in defs
        genn_defs["params"] = params

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
        for n in genn_defs["params"].keys():
            pref_n = prefix + n

            # If this variable is a native parameter,
            # evaluate it into neuron_params and simplify
            if pref_n in native_param_keys:
                neuron_params[n] = self._init_variable(True,
                                                       native_params[pref_n])
            elif pref_n in extra_param_values:
                neuron_params[n] = self._init_variable(True, extra_param_values[
                    pref_n])
            else:
                raise Exception("Property '{}' not "
                                "correctly initialised".format(n))

        # Loop through GeNN variables
        neuron_ini = {}
        for n, t in vars:
            pref_n = prefix + n

            # If this variable is a native parameter,
            # evaluate it into neuron_ini
            if pref_n in native_param_keys:
                neuron_ini[n] = self._init_variable(False,
                                                    native_params[pref_n])
            # Otherwise if there is an initial value associated with it
            elif pref_n in init_val_lookup:
                # Get its PyNN name from the lookup
                pynn_n = init_val_lookup[pref_n]
                neuron_ini[n] = self._init_variable(False, init_vals[pynn_n])
            # Otherwise if this variable is manually initialised
            elif pref_n in extra_param_values:
                # Set data type
                extra_param_values[pref_n].dtype = (
                    genn_to_numpy_types[t] if t in genn_to_numpy_types else np.float32)
                # Evaluate values into neuron initialiser
                neuron_ini[n] = self._init_variable(False,
                                                    extra_param_values[pref_n])
            else:
                raise Exception("Variable '{}' not "
                                "correctly initialised".format(n))

        return genn_model, neuron_params, neuron_ini

    def _init_variable(self, simplify_homogeneous, param):
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
            
            # **YUCK** add reference
            self._var_init_snippets.append(param_init)
            return init_var(param_init, params)
        elif param.is_homogeneous:
            return param.evaluate(simplify=simplify_homogeneous)
        else:
            return param.evaluate(simplify=False)


class GeNNStandardCellType(GeNNStandardModelType, StandardCellType):

    def __init__(self, **parameters):
        super().__init__(**parameters)

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
        creator = partial(create_postsynaptic_model,
                          self.genn_postsyn_name)

        # Build model
        return self.build_genn_model(self.postsyn_defs, native_params,
                                     init_vals, creator, prefix)

    def build_genn_neuron(self, native_params, init_vals):
        # Build callable to create a custom neuron model from defs
        creator = partial(create_neuron_model, self.genn_neuron_name)

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

        wum_params = {}
        heterogeneous_on_device = []
        # Loop through connection parameters
        for n, p in conn_params.items():
            # if parameter is of type LazyArray it means that we prevented
            # expansion via PyNN and will be generated on device
            if isinstance(p, LazyArray):
                # if mutable or will be randomly generated on device
                # we add it to host_heterogeneous_var_names so that later code
                # properly initializes them
                if n in mutable_vars or isinstance(p.base_value, RandomDistribution):
                    heterogeneous_on_device.append(n)
                    continue

                # if the parameter is in the variable dictionary but it is homogeneous
                if (n in vars and n in conn.on_device_init_params and
                        conn.on_device_init_params[n].is_homogeneous):
                    # remove it from vars
                    del vars[n]
                    # add to param_names
                    param_names.append(n)
                    # and add to homogeneous wum_params
                    wum_params[n] = conn.on_device_init_params[n].base_value

            # if the parameter was already expanded because the user chose to or
            # it is not convenient to expand on device (e.g. a FromListConnector)
            else:
                # If this parameter is in the variable dictionary,
                # but it is homogeneous
                if (n in vars and np.allclose(p, p[0]) and n not in mutable_vars):
                    # remove it from vars
                    del vars[n]
                    # add to param_names
                    param_names.append(n)
                    # and add to homogeneous wum_params
                    wum_params[n] = p[0]


        # Copy updated vars and parameters back into defs
        genn_defs["var_name_types"] = vars.items()
        genn_defs["param_names"] = param_names

        # Create custom model
        genn_model = create_weight_update_model(self.__class__.__name__,
                                                **genn_defs)

        # Loop through GeNN variables
        wum_init = {}
        for n, t in vars.items():
            # Get type to use for variable
            var_type = (genn_to_numpy_types[t] if t in genn_to_numpy_types
                        else np.float32)

            # If this variable is set by connection parameters,
            # use these as initial values
            if n in conn_params and n not in heterogeneous_on_device:
                wum_init[n] = conn_params[n].astype(var_type, copy=False)
            # If there is a default in the model, use that
            elif n in self.default_initial_values:
                wum_init[n] = self.default_initial_values[n]
            # Otherwise, if the parameter is to be initialized on device
            elif n in conn.on_device_init_params:
                wum_init[n] = self._init_variable(True,
                                                  conn.on_device_init_params[n])
            else:
                raise Exception("Variable '{}' not "
                                "correctly initialised".format(n))
        # Zero all pre-synaptic variables
        # **TODO** other means of initialisation
        wum_pre_init = ({}
                        if "pre_var_name_types" not in genn_defs
                        else {n[0]: 0.0
                              for n in genn_defs["pre_var_name_types"]})

        # Zero all post-synaptic variables
        # **TODO** other means of initialisation
        wum_post_init = ({}
                         if "post_var_name_types" not in genn_defs
                         else {n[0]: 0.0
                               for n in genn_defs["post_var_name_types"]})

        return genn_model, wum_params, wum_init, wum_pre_init, wum_post_init


class GeNNStandardCurrentSource(GeNNStandardModelType, StandardCurrentSource):
    def __init__(self, **parameters):
        super().__init__(**parameters)
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

