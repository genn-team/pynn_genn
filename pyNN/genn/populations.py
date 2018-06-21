import numpy
from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, Sequence, simplify
from . import simulator
from .recording import Recorder


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        """return a ParameterSpace containing native parameters"""
        parameter_dict = {}
        for name in names:
            value = self.parent._parameters[name]
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
        self._neuron_parameters = {}
        self._postsyn_parameters = {}
        for n, v in self._parameters.items():
            if n.startswith( 'neuron_' ):
                self._neuron_parameters[n[len('neuron_'):]] = v
            elif n.startswith( 'postsyn_' ):
                self._postsyn_parameters[n[len('postsyn_'):]] = v
        
        self._neuron_ini = {}
        self._postsyn_ini = {}
        for n, v in self.celltype.default_initial_values.items():
            if n in self.celltype.translations:
                n = self.celltype.translations[n]['translated_name']
                if n.startswith( 'neuron_' ):
                    self._neuron_ini[n[len('neuron_'):]] = v
                elif n.startswith( 'postsyn_' ):
                    self._postsyn_ini[n[len('postsyn_'):]] = v
        
        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size
        self._simulator.state.populations.append( self )

    def _create_native_population(self):

        n_param = {}
        n_ini = {}

        for n, v in self._neuron_parameters.items():
            n_param[n] = self._get_first_value( v )
            #  if isinstance( v, float ):
            #      n_param[n] = v
            #  else:
            #
            #      n_param[n] = v[0]
        
        for n, v in self._neuron_ini.items():
            n_ini[n] = self._get_first_value( v )
            #  if isinstance( v, numpy.ndarray ):
            #      n_ini[n] = v[0]
            #  else:
            #      n_ini[n] = v

        simulator.state.model.addNeuronPopulation(
                self.label,
                self.size,
                self.celltype.genn_neuron,
                n_param,
                n_ini )

    def _initialize_native_population( self ):
        # TODO set values of the native population to correct ones
        for k, v in self.initial_values.items():
            if k in self.celltype.translations:
                k = self.celltype.translations[k]['translated_name']
                if k.startswith( 'neuron_' ):
                    k = k[len('neuron_'):]
                    self._simulator.state.model.initializeVarOnDevice(
                        self.label,
                        k,
                        list( range( self.size ) ),
                        list( v )
                    )

    def _get_first_value( self, vals ):
        v = vals
        while not isinstance( v, float ):
            if isinstance( v, Sequence ):
                v = v.value
            else:
                v = v[0]
        return v

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
            
            if name.startswith( 'neuron_' ):
                self._neuron_parameters[name[len('neuron_'):]] = value
            elif name.startswith( 'postsyn_inh_' ) and self.receptor_types == 'inhibitory':
                self._postsyn_parameters[name[len('postsyn_inh_'):]] = value
            elif name.startswith( 'postsyn_exc_' ) and self.receptor_types == 'excitatory':
                self._postsyn_parameters[name[len('postsyn_exc_'):]] = value
