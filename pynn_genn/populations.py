from copy import deepcopy
from six import iteritems
from collections import Sized
from lazyarray import larray
import numpy as np

from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import Sequence
from pyNN.parameters import simplify as simplify_params

from . import simulator
from .recording import Recorder
from .model import sanitize_label


class Assembly(common.Assembly):
    _simulator = simulator

    @property
    def local_size(self):
        """for reasons unknown, assemblies don't implement this
        which breaks connection building if there is a callback"""
        return sum(p.local_size for p in self.populations)

class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def set(self, **parameters):
        # Loop through all parameters
        parent_params = self.parent._parameters
        for n, v in iteritems(parameters):
            # Expand parent parameters
            param_vals = parent_params[n].evaluate(simplify=False)

            # If parameter is a sequence and value has a length
            # **NOTE** following logic comes from pyNN.parameters.ParameterSpace
            if parent_params[n].dtype is Sequence and isinstance(v, Sized):
                # If it's empty, replace v with empty sequence
                if len(v) == 0:
                    v = Sequence([])
                # Otherwise, if v isn't a sequence of sequences
                elif not isinstance(v[0], Sequence):
                    # If v is a sequence of some other things with length,
                    if isinstance(v[0], Sized):
                        v = type(v)([Sequence(x) for x in v])
                    # Otherwise, convert v into a Sequence
                    else:
                        v = Sequence(v)

            # Replace masked section of values
            param_vals[self.mask] = v

            # Convert result back into lazy array
            parent_params[n] = larray(param_vals,
                                      dtype=parent_params[n].dtype,
                                      shape=parent_params[n].shape)

    def get(self, parameter_names, gather=False, simplify=True):
        # if all the cells have the same value for a parameter, should
        # we return just the number, rather than an array?
        parent_params = self.parent._parameters
        if isinstance(parameter_names, basestring):
            val = parent_params[parameter_names][self.mask]
            return (simplify_params(val) if simplify else val)
        # Otherwise, if we should simplify
        elif simplify:
            return [simplify_params(parent_params[n][self.mask])
                    for n in parameter_names]
        # Otherwise
        else:
            return [parent_params[n][self.mask] for n in parameter_names]

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        """
        Create a population of neurons all of the same type.
        """
        super(Population, self).__init__(size, cellclass, cellparams, structure,
                initial_values, label)

        # Create empty list to hold injected currents
        self._injected_currents = []

        # Give population a unique GeNN label
        # If a label is passed we include a sanitized version of this in it
        # **NOTE** while superclass will always populate label PROPERTY the result isn't useful
        if label is None:
            self._genn_label = "population_%u" % Population._nPop
        else:
            self._genn_label = "population_%u_%s" % (Population._nPop, sanitize_label(label))

    def _create_cells(self):
        id_range = np.arange(simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)

        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank
        self._mask_local = is_local(self.all_cells)

        # Take a deep copy of cell type parameters
        # **NOTE** we always use PyNN parameters here
        self._parameters = deepcopy(self.celltype.parameter_space)

        # Set shape
        self._parameters.shape = (self.size,)

        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size
        self._simulator.state.populations.append(self)

    def _create_native_population(self):
        """Create a GeNN population
            This function is supposed to be called by the simulator
        """
        # Build GeNN neuon model
        native_parameters = self._native_parameters
        self._genn_nmodel, neuron_params, neuron_ini =\
            self.celltype.build_genn_neuron(native_parameters, self.initial_values)

        self._pop = simulator.state.model.add_neuron_population(
            self._genn_label, self.size, self._genn_nmodel,
            neuron_params, neuron_ini)

        # Get any extra global parameters defined by the model
        extra_global = self.celltype.get_extra_global_neuron_params(
            native_parameters, self.initial_values)

        # Add to underlying neuron group
        for n, v in iteritems(extra_global):
            self._pop.add_extra_global_param(n, v)

        for label, inj_curr, inj_cells in self._injected_currents:
            # Take a copy of current source parameters and
            # set its shape to match population
            inj_params = deepcopy(inj_curr.native_parameters)
            inj_params.shape = (self.size,)

            # Build current source model
            genn_cs, cs_params, cs_ini =\
                inj_curr.build_genn_current_source(inj_params)

            # Extract the applyInj variable
            apply_inj = cs_ini["applyIinj"]

            # Convert indices to a numpy array and make relative to 1st ID
            inj_indices = np.asarray(inj_cells, dtype=int)
            inj_indices -= self.first_id

            # Set all indices to one
            apply_inj[inj_indices] = 1

            cs = simulator.state.model.add_current_source(
                label, genn_cs, self._genn_label, cs_params, cs_ini)

            extra_global = inj_curr.get_extra_global_params(inj_params)

            for n, v in iteritems(extra_global):
                cs.add_extra_global_param(n, v)

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def set(self, **parameters):
        self._parameters.update(**parameters)

    def get(self, parameter_names, gather=False, simplify=True):
        # if all the cells have the same value for a parameter, should
        # we return just the number, rather than an array?
        if isinstance(parameter_names, basestring):
            param = self._parameters[parameter_names]
            return param.evaluate(simplify=simplify)
        # Otherwise
        else:
            return [self._parameters[n].evaluate(simplify=simplify)
                    for n in parameter_names]

    @property
    def _native_parameters(self):
        return self.celltype.translate(self._parameters)