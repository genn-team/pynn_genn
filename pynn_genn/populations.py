import numpy
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
            if isinstance(value, numpy.ndarray):
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
        id_range = numpy.arange(simulator.state.id_counter,
                                simulator.state.id_counter + self.size)
        self.all_cells = numpy.array([simulator.ID(id) for id in id_range],
                                     dtype=simulator.ID)

        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank
        self._mask_local = is_local(self.all_cells)

        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        parameter_space.evaluate(mask=self._mask_local, simplify=False)
        self._parameters = parameter_space.as_dict()

        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size
        self._simulator.state.populations.append(self)

    def _create_native_population(self):
        """Create a GeNN population
            This function is supposed to be called by the simulator
        """
        neuron_params = self.celltype.get_neuron_params(
                self._parameters,
                self.initial_values
        )
        neuron_ini = self.celltype.get_neuron_vars(
                self._parameters,
                self.initial_values
        )

        self._pop = simulator.state.model.add_neuron_population(
            self.label, self.size, self.celltype.genn_neuron,
            neuron_params, neuron_ini)

        extra_global = self.celltype.get_extra_global_params(
                self._parameters,
                self.initial_values
        )

        for n, v in extra_global.items():
            self._pop.add_extra_global_param(n, v)

        for label, inj_curr, inj_cells in self._injected_currents:
            inj_ini = inj_curr.get_currentsource_vars()
            applyIinj = numpy.array([0.0 for i in range( self.size )])
            applyIinj[[ic - self.first_id for ic in inj_cells]] = 1.0
            inj_ini.update({'applyIinj' : list(applyIinj)})
            cs = simulator.state.model.add_current_source(
                label, inj_curr.genn_currentsource, self.label,
                inj_curr.get_currentsource_params(), inj_ini)

            extra_global = inj_curr.get_extra_global_params()

            for n, v in extra_global.items():
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
        parameter_space.evaluate(simplify=False, mask=self._mask_local)
        for name, value in parameter_space.items():
            self._parameters[name] = value
