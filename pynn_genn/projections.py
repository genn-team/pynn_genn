from itertools import repeat
import logging
from copy import deepcopy
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


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        if label is None:
            # override label autogeneration because GeNN does not support unicode
            pre_label = presynaptic_population.label
            post_label = postsynaptic_population.label
            if pre_label and post_label:
                # adding a number to each projection to avoid collisions
                # there can be several projections between two populations
                label = '{}_{}_{}'.format(pre_label, post_label,
                        len(self._simulator.state.projections))
        # make sure that the label is alphanumeric
        else:
            label = ''.join(c for c in label if c.isalnum())
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)

        #  Create connections
        self.connections = []
        connector.connect(self)
        self.connections = np.array(self.connections)
        self.use_sparse = True
        if isinstance(connector, AllToAllConnector):
            self.use_sparse = False
        # process assemblies
        if hasattr(self.pre, 'populations') and not self.pre.single_population():
            setattr(self, 'subprojections', [])
            self.subprojections = []
            pre_pops = self.pre.populations
            if hasattr(self.post, 'populations') and not self.post.single_population():
                post_pops = self.post.populations
                pre_cum_size = 0
                for pre_pop in pre_pops:
                    post_cum_size = 0
                    for post_pop in post_pops:
                        subconns = self.connections[
                                np.logical_and(
                                    self.connections >= Connection(pre_cum_size, post_cum_size),
                                    self.connections < Connection(pre_cum_size + pre_pop.size - 1,
                                                                  post_cum_size + post_pop.size))]
                        if len(subconns) == 0:
                            continue
                        self.subprojections.append(
                                Projection(pre_pop, post_pop, connector,
                                            synapse_type, source, receptor_type,
                                            space, None))
                        subconns -= Connection(pre_cum_size, post_cum_size)
                        post_cum_size += post_pop.size
                    pre_cum_size += pre_pop.size
            else:
                pre_cum_size = 0
                for pre_pop in pre_pops:
                    subconns = self.connections[
                            np.logical_and(
                                self.connections >= Connection(pre_cum_size, 0),
                                self.connections < Connection(pre_cum_size + pre_pop.size, 0))]
                    if len(subconns) == 0:
                        continue
                    self.subprojections.append(Projection(pre_pop, self.post,
                        connector, synapse_type, source, receptor_type, space, None))
                    subconns -= Connection(pre_cum_size, 0)
                    self.subprojections[-1].connections = subconns
                    pre_cum_size += pre_pop.size

        elif hasattr(self.post, 'populations') and not self.post.single_population():
            setattr(self, 'subprojections', [])
            self.subprojections = []
            post_pops = self.post.populations
            post_cum_size = 0
            for post_pop in post_pops:
                self.subprojections.append(
                    Projection(self.pre, post_pop, connector, synapse_type,
                                source, receptor_type, space, None))
                self.subprojections[-1].connections = self.connections[
                        np.logical_and(
                            self.connections >= Connection(0, post_cum_size),
                            self.connections < Connection(self.pre.size,
                                                          post_cum_size + post_pop.size))]
                self.subprojections[-1].connections -= (0, post_cum_size)
                post_cum_size += post_pop.size
        else:
            # add projection to the simulator only if it does not have complex assemblies
            # otherwise subprojections are added above
            self._simulator.state.projections.append(self)

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
    def _set_initial_value_array(self, variable, initial_value):
        pass

    def _get_attributes_as_arrays(self, *names):
        assert False

    def _get_attributes_as_list(self, *names):
        # Dig out reference to GeNN model
        genn_model = self._simulator.state.model

        # Pull projection state from device
        genn_model.pull_state_from_device(self.label)

        # Loop through names of variables that are required
        # **NOTE** no idea why they come as a list inside a tuple
        variables = []
        for n in names[0]:
            if n == "presynaptic_index":
                assert False
            elif n == "postsynaptic_index":
                assert False
            # Otherwise add view of GeNN variable to list
            else:
                variables.append(self._pop.vars[n].view)

        # Unzip into list of tuples and return
        return zip(*variables)

    def _create_native_projection(self):
        """Create a GeNN projection (aka synaptic population)
            This function is supposed to be called by the simulator
        """
        if self.use_sparse:
            matrixType = 'SPARSE_INDIVIDUALG'
        else:
            matrixType = 'DENSE_INDIVIDUALG'

        if len(self.connections) == 0:
            logging.warning('Projection {} has no connections. Ignoring.'.format(self.label))
            return
        else:
            prIdc, poIdc, gs = zip(*((c.presynaptic_index, c.postsynaptic_index, c.g)
                                     for c in self.connections))
            delaySteps = getattr(self.connections[0], 'delaySteps')
            if not np.allclose([getattr(conn, 'delaySteps') for conn in self.connections], delaySteps):
                delaySteps = int(np.mean([getattr(conn, 'delaySteps') for conn in self.connections]))
                logging.warning('Projection {}: GeNN does not support variable delays for a single projection. '
                                'Using mean value {} ms for all connections.'.format(
                                    self.label,
                                    delaySteps * simulator.state.dt))
        prIdc = np.array(prIdc)
        poIdc = np.array(poIdc)

        prePop = self.pre
        # if assembly with one base population
        if hasattr(self.pre, 'populations'):
            prePop = self.pre.populations[0]
            if hasattr(prePop, 'parent'):
                prePop = prePop.grandparent;
            popsIdc = []
            cumSize = 0
            for pop in self.pre.populations:
                mask = np.logical_and(cumSize <=  prIdc, prIdc < (cumSize + pop.size))
                popsIdc.append(prIdc[mask] - cumSize)
                cumSize += pop.size
                if hasattr(pop, 'parent'):
                    popsIdc[-1] = pop.index_in_grandparent(popsIdc[-1])
            prIdc = [idx for idx in idc for idc in popsIdc]

        # if population view
        elif hasattr(prePop, 'parent'):
            prIdc = prePop.index_in_grandparent(list(prIdc))
            prePop = prePop.grandparent;

        postPop = self.post
        # if assembly with one base population
        if hasattr(self.post, 'populations'):
            postPop = self.post.populations[0]
            if hasattr(postPop, 'parent'):
                postPop = postPop.grandparent;
            popsIdc = []
            cumSize = 0
            for pop in self.post.populations:
                mask = np.logical_and(cumSize <=  poIdc, poIdc < (cumSize + pop.size))
                popsIdc.append(poIdc[mask] - cumSize)
                cumSize += pop.size
                if hasattr(pop, 'parent'):
                    popsIdc[-1] = pop.index_in_grandparent(popsIdc[-1])
            prIdc = [idx for idc in popsIdc for idx in idc]
        # if population view
        elif hasattr(postPop, 'parent'):
            poIdc = postPop.index_in_grandparent(list(poIdc))
            postPop = postPop.grandparent;

        conns = list(zip(prIdc, poIdc))

        wupdate_parameters = self.synapse_type.get_params(self.connections, self.initial_values)
        wupdate_ini = self.synapse_type.get_vars(self.connections, self.initial_values)
        wupdate_ini['g'] = None

        if self.receptor_type == 'inhibitory':
            prefix = 'inh_'
        else:
            prefix = 'exc_'

        postsyn_parameters = postPop.celltype.get_postsynaptic_params(
                postPop._parameters, postPop.initial_values, prefix)
        postsyn_ini = postPop.celltype.get_postsynaptic_vars(
                postPop._parameters, postPop.initial_values, prefix)

        self._pop = simulator.state.model.add_synapse_population(
            self.label, matrixType, int(delaySteps),
            prePop._pop, postPop._pop,
            self.synapse_type.genn_weight_update, wupdate_parameters, wupdate_ini, {}, {},
            self.post.celltype.genn_postsyn, postsyn_parameters, postsyn_ini)

        self._pop.set_connections(conns, gs)
