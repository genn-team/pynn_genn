from copy import deepcopy
from six import iteritems
import numpy as np
from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, Sequence, simplify
from . import simulator
from .recording import Recorder
from .standardmodels.cells import SpikeSourceArray


class Assembly(common.Assembly):
    _simulator = simulator

    def single_population(self):
        ref_label = self.populations[0].label
        if hasattr(self.populations[0], 'parent'):
            ref_label = self.populations[0].grandparent.label
        for pop in self.populations:
            label = pop.label
            if hasattr(pop, 'parent'):
                label = pop.grandparent.label
            if label != ref_label:
                return False
        return True

    @property
    def base_populations(self):
        return [pop.grandparent if hasattr(pop, 'parent') else pop
                 for pop in self.populations]


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        """return a ParameterSpace containing native parameters"""
        parameter_dict = {}
        for name in names:
            value = self.parent._get_parameters(name)
            if isinstance(value, np.ndarray):
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, shape=(self.size,))  # or local size?

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        for name, value in parameter_space.items():
            self.parent._parameters[name][self.mask] = value.evaluate(simplify=True)

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly
    _injected_currents = []

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        """
        Create a population of neurons all of the same type.
        """
        # make sure that the label is alphanumeric
        if label is not None:
            label = ''.join( c for c in label if c.isalnum())
        super(Population, self).__init__(size, cellclass, cellparams, structure,
                initial_values, label)

    def _create_cells(self):
        id_range = np.arange(simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)

        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank
        self._mask_local = is_local(self.all_cells)

        # Take a deep copy of cell type parameters
        if isinstance(self.celltype, StandardCellType):
            self._parameters = deepcopy(self.celltype.native_parameters)
        else:
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
        self._genn_nmodel, neuron_params, neuron_ini =\
            self.celltype.build_genn_neuron(self._parameters, self.initial_values)

        self._pop = simulator.state.model.add_neuron_population(
            self.label, self.size, self._genn_nmodel,
            neuron_params, neuron_ini)

        # Get any extra global parameters defined by the model
        extra_global = self.celltype.get_extra_global_neuron_params(
            self._parameters, self.initial_values)

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
                label, genn_cs, self.label, cs_params, cs_ini)

            extra_global = inj_curr.get_extra_global_params(inj_params)

            for n, v in iteritems(extra_global):
                cs.add_extra_global_param(n, v)

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            parameter_dict[name] = simplify(self._parameters[name])
        return ParameterSpace(parameter_dict, shape=(self.local_size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        for name, value in parameter_space.items():
            self._parameters[name] = deepcopy(value)
