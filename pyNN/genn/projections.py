from itertools import repeat
from copy import deepcopy
try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
import numpy as np
from pyNN import common
from pyNN.core import ezip
from pyNN.space import Space
from . import simulator


class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, **attributes):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        for name, value in attributes.items():
            setattr(self, name, value)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        return tuple([getattr(self, name) for name in attribute_names])


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        if label is None:
            if presynaptic_population.label and postsynaptic_population.label:
                label = '{}_{}'.format( presynaptic_population.label,
                                        postsynaptic_population.label )
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)

        #  Create connections
        self.connections = []
        connector.connect(self)


        prefixes = ('inh_', 'exc_')
        if self.receptor_type == 'inhibitory':
            prefix_idx = 0
        else:
            prefix_idx = 1
        
        self._postsyn_parameters = { ( k[len(prefixes[prefix_idx]):] 
                        if k.startswith(prefixes[prefix_idx]) else k ) : 
                    ( v[0] if isinstance( v, np.ndarray ) else v )
                    for k, v in self.pre._postsyn_parameters.items()
                    if not k.startswith( prefixes[1 - prefix_idx] ) }

        self._wupdate_parameters = { 
                k : ( v[0] if isinstance( v, np.ndarray ) else v )
                for k, v in self.synapse_type.native_parameters.items()
                if k != 'g' }
        self._wupdate_ini = {
                k : ( v[0] if isinstance( v, np.ndarray ) else v )
                for k, v in self.synapse_type.default_initial_values.items() }

        self._wupdate_ini['g'] = 0.0

        self._simulator.state.projections.append( self )

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        raise NotImplementedError

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
        for pre_idx, other in ezip(presynaptic_indices, *connection_parameters.values()):
            other_attributes = dict(zip(connection_parameters.keys(), other))
            self.connections.append(
                Connection(pre_idx, postsynaptic_index, **other_attributes)
            )
    def _set_initial_value_array( self, variable, initial_value ):
        pass

    def _create_native_projection(self):

        matrixType = 'DENSE_INDIVIDUALG'

        simulator.state.model.addSynapsePopulation(
                self.label,
                matrixType,
                int( self.synapse_type.native_parameters['delaySteps'].base_value ),
                self.pre.label,
                self.post.label,
                self.synapse_type.genn_weightUpdate,
                self._wupdate_parameters,
                self._wupdate_ini,
                self.pre.celltype.genn_postsyn,
                self._postsyn_parameters,
                self.pre._postsyn_ini
        )

    def _initialize_native_projection( self ):
        # TODO set values of the native projection to correct ones
        #  for k, v in self.initial_values.items():
        #      print( k, v )
        #  for k, v in self.synapse_type.default_initial_values.items():
        #      print( k, v )
        #  for k, v in self.pre.celltype.default_initial_values.items():
        #      print( k, v )
        #  for k, v in self.pre.initial_values.items():
        #      print( k, v )
        connection_mask = []
        connection_g = []
        pre_size = self.pre.size
        post_size = self.post.size
        for conn in self.connections:
            idx = conn.presynaptic_index * post_size + conn.postsynaptic_index
            connection_mask.append( idx )
            connection_g.append( conn.g )
        self._simulator.state.model.initializeVarOnDevice( self.label, 'g',
                connection_mask, connection_g )
