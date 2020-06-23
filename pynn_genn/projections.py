from collections import defaultdict, namedtuple, Iterable
from itertools import product, repeat
import logging
from copy import deepcopy
from six import iteritems, itervalues
try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
import numpy as np

from pyNN import common
from pyNN.connectors import AllToAllConnector, FromListConnector, \
                            FromFileConnector
from pyNN.core import ezip
from pyNN.space import Space
from pyNN.parameters import LazyArray

from . import simulator
from .standardmodels.synapses import StaticSynapse
from .model import sanitize_label
from .contexts import ContextMixin
from .random import NativeRNG

# Tuple type used to store details of GeNN sub-projections
SubProjection = namedtuple("SubProjection",
                           ["genn_label", "pre_pop", "post_pop",
                            "pre_slice", "post_slice", "syn_pop", "wum_params"])


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
    _static_synapse_class = StaticSynapse

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type=None, source=None,
                 receptor_type=None, space=Space(), label=None):
        # Make a deep copy of synapse type
        # so projection can independently change parameters
        synapse_type = deepcopy(synapse_type)
        common.Projection.__init__(self, presynaptic_population,
                                   postsynaptic_population, connector,
                                   synapse_type, source, receptor_type,
                                   space, label)

        # Initialise the context stack
        ContextMixin.__init__(self, {})

        # Native projections list. Remains empty if no projections are generated.
        self._sub_projections = []

        self.use_sparse = connector.use_sparse
        # Generate name stem for sub-projections created from this projection
        # **NOTE** superclass will always populate label PROPERTY
        # with something moderately useful i.e. at least unique
        self._genn_label_stem =\
            "projection_%u_%s" % (Projection._nProj,
                                  sanitize_label(self.label))

        # Add projection to the simulator
        self._simulator.state.projections.append(self)

    def set(self, **attributes):
        self.synapse_type.parameter_space.update(**attributes)

    @ContextMixin.use_contextual_arguments()
    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            conn_pre_indices, conn_post_indices, conn_params,
                            **connection_parameters):
        num_synapses = len(presynaptic_indices)
        conn_pre_indices.extend(presynaptic_indices)
        conn_post_indices.extend(repeat(postsynaptic_index,
                                        times=num_synapses))

        # Loop through connection _parameters
        for p_name, p_val in iteritems(connection_parameters):
            if isinstance(p_val, Iterable):
                conn_params[p_name].extend(p_val)
            else:
                conn_params[p_name].extend(repeat(p_val, times=num_synapses))

    @ContextMixin.use_contextual_arguments()
    def _on_device_connect(self, pre_size, post_size,
                           conn_pre_indices, conn_post_indices, conn_params,
                           **connection_parameters):
        for k, m in iteritems(connection_parameters):
            conn_params[k] = m

    def _set_initial_value_array(self, variable, initial_value):
        pass

    def _get_attributes_as_arrays(self, names, multiple_synapses="sum"):
        # Dig out reference to GeNN model
        genn_model = self._simulator.state.model

        # Loop through sub-populations
        for sub_pop in self._sub_projections:
            # Loop through names and pull variables
            for n in names:
                if n != "presynaptic_index" and n != "postsynaptic_index" and n in sub_pop.syn_pop.vars:
                    genn_model.pull_var_from_device(sub_pop.genn_label, n)

        # If projection is sparse
        variables = []
        if self.use_sparse:
            # Loop through variables
            for n in names[0]:
                # Create empty array to hold variable and initialise to NaN
                var = np.empty((self.pre.size, self.post.size))
                var[:] = np.nan

                # Loop through sub-populations
                for sub in self._sub_projections:
                    # Get connection indices in
                    sub_pre_inds = sub.syn_pop.get_sparse_pre_inds()
                    sub_post_inds = sub.syn_pop.get_sparse_post_inds()

                    # Get slice of variable matrix
                    sub_var = var[sub.pre_slice, sub.post_slice]

                    if n in sub.wum_params:
                        sub_var[sub_pre_inds,sub_post_inds] = sub.wum_params[n]
                    else:
                        sub_var[sub_pre_inds,sub_post_inds] = sub.syn_pop.get_var_values(n)

                # Add variable to list
                variables.append(var)

        # Otherwise
        else:
            # Loop through variables
            for n in names[0]:
                # Create empty array to hold variable
                var = np.empty((self.pre.size, self.post.size))

                # Loop through sub-populations
                for sub in self._sub_projections:
                    # Get shape of sliced region
                    sub_shape = (sub.pre_slice.stop - sub.pre_slice.start,
                                 sub.post_slice.stop - sub.post_slice.start)

                    # Reshape variable values from sub-population
                    # and copy into slice of var
                    if n in sub.wum_params:
                        var[sub.pre_slice, sub.post_slice] = sub.wum_params[n]
                    else:
                        var[sub.pre_slice, sub.post_slice] =\
                            np.reshape(sub.syn_pop.get_var_values(n),sub_shape)

                # Add variable to list
                variables.append(var)

        # Return variables as tuple
        return tuple(variables)

    def _get_attributes_as_list(self, names):
        # Dig out reference to GeNN model
        genn_model = self._simulator.state.model

        # Loop through sub-populations
        for sub_pop in self._sub_projections:
            # Loop through names and pull variables
            for n in names:
                if n != "presynaptic_index" and n != "postsynaptic_index":
                    genn_model.pull_var_from_device(sub_pop.genn_label, n)

        # Loop through names of variables that are required
        variables = []
        for n in names:
            if n == "presynaptic_index":
                # If projection is sparse
                if self.use_sparse:
                    # Stack together presynaptic indices from each
                    # sub-projection after adding slice offsets
                    pre_inds = np.hstack(
                        p.pre_slice.start + p.syn_pop.get_sparse_pre_inds()
                        for p in self._sub_projections)

                    # Add indices to list
                    variables.append(pre_inds)
                else:
                    variables.append(np.repeat(np.arange(self.pre.size),
                                               self.post.size))
            elif n == "postsynaptic_index":
                # If projection is sparse
                if self.use_sparse:
                    # Stack together postsynaptic indices from each
                    # sub-projection after adding slice offsets
                    post_inds = np.hstack(
                        p.post_slice.start + p.syn_pop.get_sparse_post_inds()
                        for p in self._sub_projections)

                    # Add indices to list
                    variables.append(post_inds)

                # Otherwise generate postsynaptic indices for dense structure
                else:
                    variables.append(np.tile(np.arange(self.post.size),
                                             self.pre.size))

            # Otherwise, stack together the variables
            # from each sub-projection and add to list
            else:
                variables.append(np.hstack(p.syn_pop.get_var_values(n)
                                           for p in self._sub_projections))

        # Unzip into list of tuples and return
        return list(zip(*variables))

    def _get_sub_pops(self, pop, neuron_slice, conn_inds, conn_mask):
        """ recursive helper function to split up connection indices generated
        by standard connect function into list containing sub-projections
        to actual projections i.e. not Assemblies or PopulationViews"""
        # If population is an assembly
        if isinstance(pop, common.Assembly):
            sub_pops = []
            n_slice_start = neuron_slice.start
            for child_pop in pop.populations:
                # Build slice
                n_slice_stop = min(n_slice_start + child_pop.size,
                                   neuron_slice.stop)
                child_slice = slice(n_slice_start, n_slice_stop)

                # Build mask to select connections in this slice
                child_mask = (conn_mask &
                              (conn_inds >= n_slice_start) &
                              (conn_inds < n_slice_stop))

                # Transform our block of indices so they're
                # in terms of child population's neurons
                conn_inds[child_mask] -= n_slice_start

                # Extend list of sub-populations with the
                # result of recursing through child
                sub_pops.extend(self._get_sub_pops(child_pop, child_slice,
                                                   conn_inds, child_mask))

                # Advance to next slice
                n_slice_start += child_pop.size
            return sub_pops
        # Otherwise, if population is a population view
        elif isinstance(pop, common.PopulationView):
            # Transform our block of indices so they are
            # in terms of the grandfather's neurons
            conn_inds[conn_mask] =\
                pop.index_in_grandparent(conn_inds[conn_mask])

            # Recurse to get to grandfather population
            return self._get_sub_pops(pop.grandparent, neuron_slice,
                                      conn_inds, conn_mask)
        # Otherwise, we've reached an actual population
        else:
            return [(pop, neuron_slice, conn_mask)]



    def _create_native_projection(self):
        """Create GeNN projections (aka synaptic populatiosn)
            This function is supposed to be called by the simulator
        """
        if self.use_sparse:
            matrix_type = "SPARSE_INDIVIDUALG"
        else:
            matrix_type = "DENSE_INDIVIDUALG"

        # Set prefix based on receptor type
        # **NOTE** this is used to translate the right set of
        # neuron parameters into postsynaptic model parameters
        prefix = "inh_" if self.receptor_type == "inhibitory" else "exc_"

        #  Create connections rows to hold synapses
        pre_indices = []
        post_indices = []
        params = defaultdict(list)

        # Build connectivity
        # **NOTE** this build connector matching shape of PROJECTION
        # this means that it will match pre and post view or assembly
        with self.get_new_context(conn_pre_indices=pre_indices,
                                  conn_post_indices=post_indices,
                                  conn_params=params):
            self._connector.connect(self)


        # Convert pre and postsynaptic indices to numpy arrays
        pre_indices = np.asarray(pre_indices, dtype=np.uint32)
        post_indices = np.asarray(post_indices, dtype=np.uint32)

        # Convert connection parameters to numpy arrays
        # **THINK** should we do something with use numpy record array?
        for c in params:
            params[c] = np.asarray(params[c])

        # put back the params which were not expanded by PyNN
        for c in self._connector.on_device_init_params:
            params[c] = self._connector.on_device_init_params[c]

        # Extract delays
        # If the delays were not expanded on host, check if homogeneous and
        # evaluate through the LazyArray method
        if "delaySteps" in self._connector.on_device_init_params:
            delay_steps = self._connector.on_device_init_params["delaySteps"]
            simplify = delay_steps.is_homogeneous
            delay_steps = delay_steps.evaluate(simplify=simplify)
            delay_steps = [delay_steps] if simplify else delay_steps
        # If delays were expanded, just pass them
        else:
            delay_steps = params["delaySteps"]


        # **TODO** add support for heterogeneous dendritic delay
        if not np.allclose(delay_steps, delay_steps[0]):
            # If delays aren't all the same
            # Get average delay
            delay_steps = int(round(np.mean(delay_steps)))
            logging.warning("Projection {}: GeNN does not support variable "
                            "delays for a single projection. Using mean "
                            "value {} ms for all connections.".format(
                                self.label,
                                delay_steps * self._simulator.state.dt))
        else:
            delay_steps = int(delay_steps[0])

        # As delay is homogeneous, use to obtain minimum delay
        # **TODO** this is a pretty hacky solution
        self.min_delay = (delay_steps + 1) * self._simulator.state.dt

        # If both pre_indices and post_indices are empty, it means that we
        # prevented PyNN from expanding indices
        if len(pre_indices) == 0 and len(post_indices) == 0:
            self._on_device_init_native_projection(
                matrix_type, prefix, params, delay_steps)
        else:
            self._on_host_init_native_projection(
                pre_indices, post_indices, matrix_type, prefix, params, delay_steps)


    def _on_device_init_native_projection(self, matrix_type, prefix, params, delay_steps):
        print('_on_device_native_projection')
        # If connectivity is sparse, configure sparse connectivity
        genn_label = "%s_%u" % (self._genn_label_stem,
                                len(self._sub_projections))
        conn_init = self._connector._init_connectivity(self)
        psm_model, psm_params, psm_ini = \
            self.post.celltype.build_genn_psm(self.post._native_parameters,
                                             self.post.initial_values,
                                             prefix)
        params['connector'] = self._connector
        wum_model, wum_params, wum_init, wum_pre_init, wum_post_init = \
                self.synapse_type.build_genn_wum(params,
                                             self.initial_values)
        syn_pop = simulator.state.model.add_synapse_population(
            genn_label, matrix_type, delay_steps,
            self.pre._pop, self.post._pop,
            wum_model, wum_params, wum_init, wum_pre_init, wum_post_init,
            psm_model, psm_params, psm_ini, conn_init)

        # if self.use_sparse:
        #     syn_pop.set_sparse_connections(conn_pre_inds, conn_post_inds)

        self._sub_projections.append(
            SubProjection(genn_label, self.pre, self.post,
                slice(0, self.pre.size), slice(0, self.post.size), syn_pop, wum_params))
        # # Build GeNN postsynaptic model

    def _on_host_init_native_projection(self, pre_indices, post_indices,
                                        matrix_type, prefix, params, delay_steps):
        num_synapses = len(pre_indices)
        if num_synapses == 0:
            logging.warning("Projection {} has no connections. "
                            "Ignoring.".format(self.label))
            return

        # Build lists of actual pre and postsynaptic populations
        # we are connecting (i.e. rather than views, assemblies etc)
        pre_pops = self._get_sub_pops(
            self.pre, slice(0, self.pre.size), pre_indices,
            np.ones(pre_indices.shape, dtype=bool))
        post_pops = self._get_sub_pops(
            self.post, slice(0, self.post.size), post_indices,
            np.ones(post_indices.shape, dtype=bool))

        # Loop through presynaptic populations and their corresponding
        # slices of presynaptic neurons
        for pre, post in product(pre_pops, post_pops):
            pre_pop, pre_slice, pre_mask = pre
            post_pop, post_slice, post_mask = post

            # Combine mask to select connections
            conn_mask = pre_mask & post_mask

            # Use mask to extract pre and post indices;
            # and connection parameters for this sub-projection
            conn_pre_inds = pre_indices[conn_mask]
            conn_post_inds = post_indices[conn_mask]

            # NOTE: had to add a check for non-expanded params
            if self.use_sparse:
                conn_params = {n: p[conn_mask] if not isinstance(p, LazyArray) else p
                               for n, p in iteritems(params)}
            else:
                ## GeNN stores synapses in this row-major order for dense matrices
                ## PyNN in some cases (FromListConnector) uses column-major
                ## thus we need to re-sort to row-major order
                to_row_major = np.lexsort((conn_post_inds, conn_pre_inds))
                conn_mask[:] = conn_mask[to_row_major]

                conn_params = {n: (p[to_row_major])[conn_mask]
                                if not isinstance(p, LazyArray) else p
                                    for n, p in iteritems(params)}

            conn_params['connector']= self._connector

            # Build GeNN postsynaptic model
            psm_model, psm_params, psm_ini =\
                post_pop.celltype.build_genn_psm(post_pop._native_parameters,
                                                 post_pop.initial_values,
                                                 prefix)

            # Build GeNN weight update model
            wum_model, wum_params, wum_init, wum_pre_init, wum_post_init =\
                self.synapse_type.build_genn_wum(conn_params,
                                                 self.initial_values)

            # Build a unique label for sub-projection
            genn_label = "%s_%u" % (self._genn_label_stem,
                                    len(self._sub_projections))

            # Create GeNN synapse population
            syn_pop = simulator.state.model.add_synapse_population(
                genn_label, matrix_type, delay_steps,
                pre_pop._pop, post_pop._pop,
                wum_model, wum_params, wum_init, wum_pre_init, wum_post_init,
                psm_model, psm_params, psm_ini)

            # If connectivity is sparse, configure sparse connectivity
            if self.use_sparse:
                syn_pop.set_sparse_connections(conn_pre_inds, conn_post_inds)

            self._sub_projections.append(
                SubProjection(genn_label, pre_pop, post_pop,
                              pre_slice, post_slice, syn_pop, wum_params))
