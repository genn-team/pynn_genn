from itertools import groupby
from copy import deepcopy
import numpy as np
from pygenn.genn_model import (create_custom_neuron_class, create_custom_postsynaptic_class,
                               create_custom_current_source_class)

from pygenn import genn_wrapper
from pyNN.standardmodels import (StandardModelType,
                                 StandardCellType,
                                 StandardCurrentSource,
                                 build_translations)
from conversions import convert_to_single, convert_to_array, convert_init_values
from six import iteritems, iterkeys

class GeNNStandardModelType(StandardModelType):

    genn_extra_parameters = {}

    def translate_dict(self, val_dict):
        return {self.translations[n]['translated_name'] : v.evaluate()
                for n, v in val_dict.items() if n in self.translations.keys()}

    def get_native_params(self, native_parameters, init_values, param_names, prefix=''):
        native_init_values = self.translate_dict(init_values)
        native_params = {}
        for pn in param_names:
            if prefix + pn in native_parameters.keys():
                native_params[pn] = native_parameters[prefix + pn]
            elif prefix + pn in native_init_values.keys():
                native_params[pn] = native_init_values[prefix + pn]
            elif pn in native_parameters.keys():
                native_params[pn] = native_parameters[pn]
            elif pn in native_init_values.keys():
                native_params[pn] = native_init_values[pn]
            elif pn in self.genn_extra_parameters:
                native_params[pn] = self.genn_extra_parameters[pn]
            else:
                raise Exception('Variable "{}" not found'.format(pn))

        return native_params

    def build_genn_model(self, defs, native_params, init_vals,
                         create_custom_model, prefix=""):
        assert not defs.native

        # Take a deep copy of the definitions
        genn_defs = deepcopy(defs.definitions)

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

        # Loop through native parameters
        prefix_len = len(prefix)
        for field_name, param in native_params.items():
            # If parameter is is_homogeneous, has correct prefix
            # and, without the prefix, it is in vars
            if param.is_homogeneous and field_name.startswith(prefix) and field_name[prefix_len:] in vars:
                # Remove from vars
                del vars[field_name[prefix_len:]]

                # Add it to params
                param_names.append(field_name[prefix_len:])

        # Copy updated vars and parameters back into defs
        genn_defs["var_name_types"] = vars.items()
        genn_defs["param_names"] = param_names

        # Create custom model
        genn_model = create_custom_model(**genn_defs)()

        # Simplify each of the native parameters
        # which have been implemented as GeNN parameters
        neuron_params = {n: native_params[prefix + n].evaluate(simplify=True)
                         for n in genn_defs["param_names"]}


        # Get set of native parameter names
        native_param_keys = set(native_params.keys())

        # Build dictionary mapping from the GeNN name of state variables to their PyNN name
        init_val_lookup = {self.translations[n]["translated_name"]: n
                           for n, v in init_vals.items()
                           if n in self.translations}

        # Loop through native parameters which have b
        neuron_ini = {}
        for n in iterkeys(vars):
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
            elif n in self.genn_extra_parameters:
                neuron_ini[n] = np.ones(shape=native_params.shape, dtype=np.float32) * self.genn_extra_parameters[n]
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

    def get_extra_global_params(self, native_parameters, init_values):
        var_names = [vnt[0] for vnt in self.genn_neuron.get_extra_global_params()]

        egps = self.get_native_params(native_parameters, init_values, var_names)
        # in GeNN, extra global parameters are defined for the whole population
        # and not for individual neurons.
        # The standard model which use extra global params are supposed to override
        # this function in order to convert values properly.
        return egps

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

    genn_weight_update = None

    def translate_dict(self, val_dict):
        return {self.translations[n]['translated_name'] : convert_to_array(v)
                for n, v in val_dict.items() if n in self.translations.keys()}

    def get_native_params(self, conn_params, init_values, param_names):
        default_init_values = self.default_initial_values.copy()
        default_init_values.update(init_values)
        native_init_values = self.translate_dict(default_init_values)
        native_params = {}
        
        # Loop through parameters
        for pn in param_names:
            # If they are already specified, 
            # copy parameter directly from connections
            if pn in conn_params:
                native_params[pn] = conn_params[pn]
            # Otherwise, use default
            # **NOTE** at this point it is fine for these to be scalar
            elif pn in native_init_values:
                native_params[pn] = native_init_values[pn]
            else:
                raise Exception('Variable "{}" not found'.format(pn))

        return native_params

    def get_params(self, conn_params, init_values):
        param_names = list(self.genn_weight_update.get_param_names())

        # parameters are single-valued in GeNN
        return convert_to_array(
                 self.get_native_params(conn_params,
                                        init_values,
                                        param_names))

    def get_vars(self, conn_params, init_values):
        var_names = [vnt[0] for vnt in self.genn_weight_update.get_vars()]

        return convert_init_values(
                    self.get_native_params(conn_params,
                                           init_values,
                                           var_names))
    def get_pre_vars(self):
        # Zero all presynaptic variables
        # **TODO** other means of initialisation
        return {vnt[0]: 0.0
                for vnt in self.genn_weight_update.get_pre_vars()}

    def get_post_vars(self):
        # Zero all postsynaptic variables
        # **TODO** other means of initialisation
        return {vnt[0]: 0.0
                for vnt in self.genn_weight_update.get_post_vars()}

class GeNNStandardCurrentSource(GeNNStandardModelType, StandardCurrentSource):

    def __init__(self, **parameters):
        super(GeNNStandardCurrentSource, self).__init__(**parameters);
        self.translations = build_translations(
            *self.currentsource_defs.translations)
        self.parameter_space._set_shape((1,))
        self.parameter_space.evaluate()
        self._genn_currentsource = None

    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        for pop, cs in groupby(cells, key=lambda c: c.parent):
            pop._injected_currents.append(
                    (pop.label + '_' + self.__class__.__name__ + '_' + str(pop._simulator.state.num_current_sources),
                     self,
                     list(cs)) )
            pop._simulator.state.num_current_sources += 1

    def get_currentsource_params(self):#, native_parameters, init_values):
        param_names = list(self.genn_currentsource.get_param_names())

        # parameters are single-valued in GeNN
        native_params = self.native_parameters
        native_params.evaluate()
        return convert_to_array(
                 self.get_native_params(native_params.as_dict(),
                                        {},
                                        param_names))

    def get_currentsource_vars(self):
        var_names = [vnt[0] for vnt in self.genn_currentsource.get_vars()]

        native_params = self.native_parameters
        native_params.evaluate()
        return convert_init_values(
                    self.get_native_params(native_params.as_dict(),
                                           {},
                                           var_names))

    def get_extra_global_params(self):
        var_names = [vnt[0] for vnt in self.genn_currentsource.get_extra_global_params()]

        native_params = self.native_parameters
        native_params.evaluate()

        egps = self.get_native_params(native_params, {}, var_names)
        # in GeNN, extra global parameters are defined for the whole population
        # and not for individual neurons.
        # The standard model which use extra global params are supposed to override
        # this function in order to convert values properly.
        return egps

    @property
    def genn_currentsource(self):
        if self.currentsource_defs.native:
            return getattr(genn_wrapper.CurrentSourceModels, self.genn_currentsource_name)()
        genn_defs = self.currentsource_defs.get_definitions()
        return create_custom_current_source_class(self.genn_currentsource_name,
                                              **genn_defs)()


class GeNNDefinitions(object):

    def __init__(self, definitions, translations, native=False):
        self.native = native
        self.definitions = definitions
        self.translations = translations

    def get_definitions(self, params_to_vars=None):
        """get definition for GeNN model
        Args:
        params_to_vars -- list with parameters which should be treated as variables in GeNN
        """
        defs = deepcopy(self.definitions)
        if params_to_vars is not None:
            par_names = defs["param_names"]
            var_names = defs["var_name_types"]
            for par in params_to_vars:
                par_names.remove(par)
                var_names.append((par, 'scalar'))
            defs.update({'param_names' : par_names, 'var_name_types' : var_names})
        return defs
