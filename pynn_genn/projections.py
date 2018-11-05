from collections import defaultdict, Iterable
from itertools import repeat
import logging
from copy import deepcopy
from six import iteritems, itervalues
try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
import numpy as np
from pyNN import common
from pyNN.connectors import AllToAllConnector
from pyNN.core import ezip
from pyNN.space import Space
from . import simulator
from model import sanitize_label
from contexts import ContextMixin
'''
class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, **attributes):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.attribute_names = attributes.keys()
        for name, value in attributes.items():
            setattr(self, name, value)

    def __eq__(self, other):
        if isinstance(other, Connection):
            return (self.presynaptic_index == other.presynaptic_index and
                self.postsynaptic_index == other.postsynaptic_index)
        if isinstance(other, tuple) and len(other) == 2:
            return (self.presynaptic_index == other[0] and
                self.postsynaptic_index == other[1])
        return False

    def __lt__(self, other):
        if self == other:
            return False
        if isinstance(other, Connection):
            opre = other.presynaptic_index
            opost = other.postsynaptic_index
        elif isinstance(other, tuple) and len(other) == 2:
            opre = other[0]
            opost = other[1]
        else:
            return False
        if self.presynaptic_index < opre:
            return True
        if self.presynaptic_index == opre and self.postsynaptic_index < opost:
            return True
        return False

    def __gt__(self, other):
        return not (self < other or self == other)

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other

    def __ne__(self, other):
        return not self == other

    def __isub__(self, val):
        if isinstance(val, Connection):
            self.presynaptic_index -= val.presynaptic_index
            self.postsynaptic_index -= val.postsynaptic_index
        if isinstance(val, tuple) and len(val) == 2:
            self.presynaptic_index -= val[0]
            self.postsynaptic_index -= val[1]

    def __sub__(self, val):
        if isinstance(val, Connection):
            preIdx = self.presynaptic_index - val.presynaptic_index
            postIdx = self.postsynaptic_index - val.postsynaptic_index
        if isinstance(val, tuple) and len(val) == 2:
            preIdx = self.presynaptic_index - val[0]
            postIdx = self.postsynaptic_index - val[1]
        attrs = self.as_dict(*(self.attribute_names))
        return Connection(preIdx, postIdx, **attrs)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        return tuple([getattr(self, name) for name in attribute_names])

    def as_dict(self, *attribute_names):
        # should return indices, not IDs for source and target
        return dict([(name, getattr(self, name)) for name in attribute_names])
'''

class Projection(common.Projection, ContextMixin):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)

        # Initialise the context stack
        ContextMixin.__init__(self, {})

        # **TODO** leave type up to Connector types
        self.use_sparse = (False if isinstance(connector, AllToAllConnector)
                           else True)

        # Give projection a unique GeNN label
        # **NOTE** superclass will always populate label PROPERTY with something moderately useful
        self._genn_label = "projection_%u_%s" % (Projection._nProj, sanitize_label(self.label))

        # Add projection to the simulator
        self._simulator.state.projections.append(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        raise NotImplementedError

    @ContextMixin.use_contextual_arguments()
    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            conn_pre_indices, conn_post_indices, conn_params,
                            **connection_parameters):
        num_synapses = len(presynaptic_indices)
        conn_pre_indices.extend(presynaptic_indices)
        conn_post_indices.extend(repeat(postsynaptic_index, times=num_synapses))
        
        # Loop through connection _parameters
        for p_name, p_val in iteritems(connection_parameters):
            if isinstance(p_val, Iterable):
                conn_params[p_name].extend(p_val)
            else:
                conn_params[p_name].extend(repeat(p_val, times=num_synapses))

    def _set_initial_value_array(self, variable, initial_value):
        pass

    def _get_attributes_as_arrays(self, names, multiple_synapses="sum"):
         # Dig out reference to GeNN model
        genn_model = self._simulator.state.model

        # Pull projection state from device
        genn_model.pull_state_from_device(self._genn_label)

        # If projection is sparse
        variables = []
        if self.use_sparse:
            raise Exception("Reading attributes of sparse projections "
                            "into arrays is currently not supported")
        # Otherwise 
        else:
            # Loop through variables
            for n in names[0]:
                variables.append(np.reshape(self._pop.get_var_values(n), 
                                            (self.pre.size, self.post.size)))

        # Return variables as tuple
        return tuple(variables)
    
    def _get_attributes_as_list(self, names):
        # Dig out reference to GeNN model
        genn_model = self._simulator.state.model

        # Pull projection state from device
        genn_model.pull_state_from_device(self._genn_label)

        # Loop through names of variables that are required
        variables = []
        for n in names:
            if n == "presynaptic_index":
                # If projection is sparse
                if self.use_sparse:
                    raise Exception("Reading presynaptic indices of sparse "
                                    "projections is currently not supported")
                # Otherwise generate presynaptic indices for dense structure
                else:
                    variables.append(np.repeat(np.arange(self.pre.size),
                                               self.post.size))
            elif n == "postsynaptic_index":
                # If projection is sparse
                if self.use_sparse:
                    raise Exception("Reading postsynaptic indices of sparse "
                                    "projections is currently not supported")
                # Otherwise generate postsynaptic indices for dense structure
                else:
                    variables.append(np.tile(np.arange(self.post.size),
                                             self.pre.size))

            # Otherwise add view of GeNN variable to list
            else:
                variables.append(self._pop.get_var_values(n))

        # Unzip into list of tuples and return
        return zip(*variables)

    def _create_native_projection(self):
        """Create a GeNN projection (aka synaptic population)
            This function is supposed to be called by the simulator
        """
        if self.use_sparse:
            matrixType = 'RAGGED_INDIVIDUALG'
        else:
            matrixType = 'DENSE_INDIVIDUALG'

        #  Create connections rows to hold synapses
        pre_indices = []
        post_indices = []
        conn_params = defaultdict(list)

        # Build connectivity
        # **NOTE** this build connector matching shape of PROJECTION
        # this means that it will match pre and post view or assembly
        with self.get_new_context(conn_pre_indices=pre_indices, 
                                  conn_post_indices=post_indices,
                                  conn_params=conn_params):
            self._connector.connect(self)

        # Convert pre and postsynaptic indices to numpy arrays
        pre_indices = np.asarray(pre_indices, dtype=np.uint32)
        post_indices = np.asarray(post_indices, dtype=np.uint32)

        # Convert connection parameters to numpy arrays
        # **THINK** should we do something with types here/use numpy record array?
        for c in conn_params:
            conn_params[c] = np.asarray(conn_params[c])

        num_synapses = len(pre_indices)
        if num_synapses == 0:
            logging.warning("Projection {} has no connections. "
                            "Ignoring.".format(self.label))
            return
        
        # Extract delays
        delay_steps = conn_params["delaySteps"]
        
        # If delays aren't all the same
        # **TODO** add support for heterogeneous dendritic delay
        if not np.allclose(delay_steps, delay_steps[0]):
            # Get average delay
            delay_steps = int(round(np.mean(delay_steps)))
            logging.warning('Projection {}: GeNN does not support variable delays for a single projection. '
                            'Using mean value {} ms for all connections.'.format(
                                self.label,
                                average_delay * simulator.state.dt))
        else:
            delay_steps = int(delay_steps[0])
        
        # If presynaptic population is assembly
        prePop = self.pre
        #pre_populations = []
        if isinstance(self.pre, common.Assembly):
            # Loop through populations that make up presynaptic assembly
            for p in self.pre.populations:
                ss
            assert False
            prePop = self.pre.populations[0]
            if hasattr(prePop, 'parent'):
                prePop = prePop.grandparent;
            popsIdc = []
            cumSize = 0
            for pop in self.pre.populations:
                mask = np.logical_and(cumSize <=  pre_indices, 
                                      pre_indices < (cumSize + pop.size))
                popsIdc.append(pre_indices[mask] - cumSize)
                cumSize += pop.size
                if hasattr(pop, 'parent'):
                    popsIdc[-1] = pop.index_in_grandparent(popsIdc[-1])
            pre_indices = [idx for idx in idc for idc in popsIdc]
        # Otherwise, if presynaptic population is a view
        if isinstance(self.pre, common.PopulationView):
            # Convert indices in presynaptic view into indices
            # in grandparent presynaptic population
            pre_indices = prePop.index_in_grandparent(pre_indices)

            # Use grandparent presynaptic population as prepop
            prePop = prePop.grandparent;

        # If postsynaptic population is assembly
        postPop = self.post
        if isinstance(self.post, common.Assembly):
            assert False
            for pop in self.post.populations:
                mask = np.logical_and(cumSize <=  post_indices, 
                                      post_indices < (cumSize + pop.size))
                popsIdc.append(post_indices[mask] - cumSize)
                cumSize += pop.size
                if hasattr(pop, 'parent'):
                    popsIdc[-1] = pop.index_in_grandparent(popsIdc[-1])
            pre_indices = [idx for idc in popsIdc for idx in idc]
        # Otherwise, if the postsynaptic population is a view
        if isinstance(self.post, common.PopulationView):
            # Convert indices in postsynaptic view into indices
            # in grandparent postsynaptic population
            post_indices = postPop.index_in_grandparent(post_indices)

            # Use grandparent presynaptic population as post pop
            postPop = postPop.grandparent;

        # Set prefix based on receptor type
        # **NOTE** this is used to translate the right set of
        # neuron parameters into postsynaptic model parameters
        prefix = 'inh_' if self.receptor_type == 'inhibitory' else 'exc_'

        # Build GeNN postsynaptic model
        self._psm_model, psm_params, psm_ini =\
            postPop.celltype.build_genn_psm(postPop._parameters,
                                            postPop.initial_values, prefix)

        # Build GeNN weight update model
        self._wum_model, wum_params, wum_init, wum_pre_init, wum_post_init =\
            self.synapse_type.build_genn_wum(conn_params, self.initial_values)

        self._pop = simulator.state.model.add_synapse_population(
            self._genn_label, matrixType, delay_steps,
            prePop._pop, postPop._pop,
            self._wum_model, wum_params, wum_init, wum_pre_init, wum_post_init,
            self._psm_model, psm_params, psm_ini)

        # If connectivity is sparse, configure sparse connectivity
        if self.use_sparse:
            self._pop.set_sparse_connections(pre_indices, post_indices)
