import sys
from itertools import groupby
from copy import deepcopy
from lazyarray import larray
import numpy as np
from pygenn.genn_model import (create_custom_neuron_class, create_custom_postsynaptic_class,
                               create_custom_current_source_class, create_custom_weight_update_class)

from pygenn import genn_wrapper
from pyNN.standardmodels import (StandardModelType,
                                 StandardCellType,
                                 StandardCurrentSource,
                                 build_translations)
from six import iteritems, iterkeys

# Mapping from GeNN to numpy types
genn_to_numpy_types = {
    "scalar": np.float32,
    "float": np.float32,
    "double": np.float64,
    "unsigned char": np.uint8,
    "int": np.int32,
    "unsigned int": np.uint32}

def sanitize_label(label):
    # Strip out any non-alphanumerical characters
    clean_label = "".join(c for c in label if c.isalnum())

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
            if not param.is_homogeneous and field_name.startswith(prefix) and field_name_no_prefix in param_name_types:
                # Add it to variables
                var_name_types.append((field_name, param_name_types[field_name_no_prefix]))

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
                    var_name_types.append((param_name, param_name_types[param_name_no_prefix]))

                    # Remove from parameters
                    del param_name_types[field_name_no_prefix]

        # Set parameter names in defs
        genn_defs["param_names"] = list(param_name_types.keys())

        # Create custom model
        genn_model = create_custom_model(**genn_defs)()

        # Get set of native parameter names
        native_param_keys = set(native_params.keys())

        # Build dictionary mapping from the GeNN name of state variables to their PyNN name
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
                neuron_params[n] = native_params[pref_n].evaluate(simplify=True)
            elif pref_n in extra_param_values:
                neuron_params[n] = extra_param_values[pref_n].evaluate(simplify=True)
            else:
                raise Exception('Property "{}" not correctly initialised'.format(n))

        # Loop through GeNN variables
        neuron_ini = {}
        for n, t in var_name_types:
            pref_n = prefix + n
            # If this variable is a native parameter,
            # evaluate it into neuron_ini
            if pref_n in native_param_keys:
                neuron_ini[n] = native_params[pref_n].evaluate(simplify=False)
            # Otherwise if there is an initial value associated with it
            elif pref_n in init_val_lookup:
                # Get its PyNN name from the lookup
                pynn_n = init_val_lookup[pref_n]
                neuron_ini[n] = init_vals[pynn_n].evaluate(simplify=False)
            # Otherwise if this variable is manually initialised
            elif pref_n in extra_param_values:
                # Set data type
                extra_param_values[pref_n].dtype = (genn_to_numpy_types[t] 
                                                    if t in genn_to_numpy_types 
                                                    else np.float32)

                # Evaluate values into neuron initialiser
                neuron_ini[n] = extra_param_values[pref_n].evaluate(simplify=False)
            else:
                raise Exception('Variable "{}" not correctly initialised'.format(n))

        return genn_model, neuron_params, neuron_ini


class GeNNStandardCellType(GeNNStandardModelType, StandardCellType):

    def __init__(self, **parameters):
        super(GeNNStandardCellType, self).__init__(**parameters);

        # If cell has a postsynaptic model i.e. it's not a spike source
        if hasattr(self, "postsyn_defs"):
            # Build translations from both translation tuples
            self.translations = build_translations(
                *(self.neuron_defs.translations + self.postsyn_defs.translations))
        # Otherwise, build translations from neuron translations
        else:
            self.translations = build_translations(*self.neuron_defs.translations)

    def get_extra_global_neuron_params(self, native_params, init_vals):
        return {}

    def build_genn_psm(self, native_params, init_vals, prefix):
        # Build lambda function to create a custom PSM from defs
        creator = lambda **defs : create_custom_postsynaptic_class(
            self.genn_postsyn_name, **defs)

        # Build model
        return self.build_genn_model(self.postsyn_defs, native_params,
                                     init_vals, creator, prefix)

    def build_genn_neuron(self, native_params, init_vals):
        # Build lambda function to create a custom neuron model from defs
        creator = lambda **defs : create_custom_neuron_class(
            self.genn_neuron_name, **defs)

        # Build model
        return self.build_genn_model(self.neuron_defs, native_params,
                                     init_vals, creator)


class GeNNStandardSynapseType(GeNNStandardModelType):
    def build_genn_wum(self, conn_params, init_vals):
        # Take a deep copy of the definitions
        genn_defs = deepcopy(self.wum_defs)

        # Check that it doesn't already have its
        # variables and parameters seperated out
        assert "param_names" not in genn_defs
        assert "var_name_types" not in genn_defs

        # Extract variables from copy of defs and remove
        # **NOTE** all vars are by default GeNN variables
        vars = genn_defs["vars"]
        del genn_defs["vars"]

        # Start with an empty list of parameterss
        param_names = []

        # Get set of forcibly mutable vars if synapse type has one
        mutable_vars = (self.mutable_vars
                        if hasattr(self, "mutable_vars")
                        else set())

        # Loop through connection parameters
        for n, p in iteritems(conn_params):
            # If this parameter is in the variable dictionary,
            # but it is homogenous
            if n in vars and np.allclose(p, p[0]) and n not in mutable_vars:
                # Remove from vars
                del vars[n]

                # Add it to params
                param_names.append(n)

        # Copy updated vars and parameters back into defs
        genn_defs["var_name_types"] = vars.items()
        genn_defs["param_names"] = param_names

        # Create custom model
        genn_model = create_custom_weight_update_class(self.__class__.__name__, **genn_defs)()

        # Use first entry in conn param for parameters
        wum_params = {n: conn_params[n][0]
                      for n in genn_defs["param_names"]}

        # Convert variables to arrays with correct data type
        wum_init = {n: conn_params[n].astype(genn_to_numpy_types[t]
                                             if t in genn_to_numpy_types
                                             else np.float32,
                                             copy=False)
                    for n, t in iteritems(vars)}

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
        super(GeNNStandardCurrentSource, self).__init__(**parameters);
        self.translations = build_translations(
            *self.currentsource_defs.translations)

    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        for pop, cs in groupby(cells, key=lambda c: c.parent):
            pop._injected_currents.append(
                    ("%s_%s_%u" % (pop._genn_label, self.__class__.__name__, pop._simulator.state.num_current_sources),
                     self,
                     list(cs)) )
            pop._simulator.state.num_current_sources += 1

    def get_extra_global_params(self, native_params):
        return {}

    def build_genn_current_source(self, native_params):
        # Build lambda function to create a custom current source from defs
        creator = lambda **defs : create_custom_current_source_class(
            self.genn_currentsource_name, **defs)

        # Build model
        return self.build_genn_model(self.currentsource_defs, native_params,
                                     {}, creator)

class GeNNDefinitions(object):

    def __init__(self, definitions, translations, extra_param_values={}):
        self.definitions = definitions
        self.translations = translations
        self.extra_param_values = extra_param_values
