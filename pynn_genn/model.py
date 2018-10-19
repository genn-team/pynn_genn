from itertools import groupby
from copy import deepcopy
from pygenn.genn_model import (create_custom_neuron_class, create_custom_postsynaptic_class,
                               create_custom_current_source_class)

from pygenn import genn_wrapper
from pyNN.standardmodels import (StandardModelType,
                                 StandardCellType,
                                 StandardCurrentSource,
                                 build_translations)
from conversions import convert_to_single, convert_to_array, convert_init_values

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

class GeNNStandardCellType(GeNNStandardModelType, StandardCellType):

    def __init__(self, **parameters):
        super(GeNNStandardCellType, self).__init__(**parameters);
        self.translations = build_translations(
            *(self.neuron_defs.translations + self.postsyn_defs.translations))
        self._genn_neuron = None
        self._genn_postsyn = None
        self._params_to_vars = set([])

        for p in self.native_parameters.keys():
            if not self.native_parameters[p].is_homogeneous:
                self.params_to_vars = [p]

    def get_neuron_params(self, native_parameters, init_values):
        param_names = list(self.genn_neuron.get_param_names())

        # parameters are single-valued in GeNN
        return convert_to_array(
                 self.get_native_params(native_parameters,
                                        init_values,
                                        param_names))

    def get_neuron_vars(self, native_parameters, init_values):
        var_names = [vnt[0] for vnt in self.genn_neuron.get_vars()]

        return convert_init_values(
                    self.get_native_params(native_parameters,
                                           init_values,
                                           var_names))

    def get_postsynaptic_params(self, native_parameters, init_values, prefix):
        param_names = list(self.genn_postsyn.get_param_names())

        # parameters are single-valued in GeNN
        return convert_to_array(
                 self.get_native_params(native_parameters,
                                        init_values,
                                        param_names,
                                        prefix))

    def get_postsynaptic_vars(self, native_parameters, init_values, prefix):
        var_names = [vnt[0] for vnt in self.genn_postsyn.get_vars()]

        return convert_init_values(
                    self.get_native_params(native_parameters,
                                           init_values,
                                           var_names,
                                           prefix))

    def get_extra_global_params(self, native_parameters, init_values):
        var_names = [vnt[0] for vnt in self.genn_neuron.get_extra_global_params()]

        egps = self.get_native_params(native_parameters, init_values, var_names)
        # in GeNN, extra global parameters are defined for the whole population
        # and not for individual neurons.
        # The standard model which use extra global params are supposed to override
        # this function in order to convert values properly.
        return egps

    @property
    def genn_neuron(self):
        if self.neuron_defs.native:
            return getattr(genn_wrapper.NeuronModels, self.genn_neuron_name)()
        genn_defs = self.neuron_defs.get_definitions(self._params_to_vars)
        return create_custom_neuron_class(self.genn_neuron_name, **genn_defs)()

    @property
    def genn_postsyn(self):
        if self.postsyn_defs.native:
            return getattr(genn_wrapper.PostsynapticModels, self.genn_postsyn_name)()
        genn_defs = self.postsyn_defs.get_definitions()
        return create_custom_postsynaptic_class(self.genn_postsyn_name, **genn_defs)()

    @property
    def params_to_vars(self):
        return self._params_to_vars

    @params_to_vars.setter
    def params_to_vars(self, params_to_vars):
        self._params_to_vars = self._params_to_vars.union(set(params_to_vars))


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
        self._definitions = definitions.keys()
        self.translations = translations
        for key, value in definitions.items():
            setattr(self, key, value)

    @property
    def translations(self):
        return self._translations

    @translations.setter
    def translations(self, translations):
        self._translations = translations

    @property
    def definitions(self):
        return {defname : getattr(self, defname) for defname in self._definitions}

    def get_definitions(self, params_to_vars=None):
        """get definition for GeNN model
        Args:
        params_to_vars -- list with parameters which should be treated as variables in GeNN
        """
        defs = self.definitions
        if params_to_vars is not None:
            par_names = list(self.param_names)
            var_names = list(self.var_name_types)
            for par in params_to_vars:
                par_names.remove(par)
                var_names.append((par, 'scalar'))
            defs.update({'param_names' : par_names, 'var_name_types' : var_names})
        return defs
