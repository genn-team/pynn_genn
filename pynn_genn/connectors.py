from copy import copy, deepcopy
from pyNN.core import IndexBasedExpression
from pyNN.parameters import LazyArray
from pygenn.genn_model import init_connectivity,\
    create_custom_sparse_connect_init_snippet_class
# NOTE: Had to change the import names due to naming
# resolution not working properly and losing the original
# class when initializing. This is mainly due to using multiple
# inheritance and trying to override an ancestor method with
# a mixin class.
from pyNN.connectors import (
    AllToAllConnector as AllToAllPyNN,
    OneToOneConnector as OneToOnePyNN,
    FixedProbabilityConnector as FixProbPyNN,
    FixedTotalNumberConnector as FixTotalPyNN,
    FixedNumberPreConnector as FixNumPrePyNN,
    FixedNumberPostConnector as FixNumPostPyNN,
    DistanceDependentProbabilityConnector as DistProbPyNN,
    DisplacementDependentProbabilityConnector as DisplaceProbPyNN,
    IndexBasedProbabilityConnector as IndexProbPyNN,
    SmallWorldConnector as SmallWorldPyNN,
    FromListConnector as FromListPyNN,
    FromFileConnector as FromFilePyNN,
    CloneConnector as ClonePyNN,
    ArrayConnector as ArrayPyNN,
    Connector
)

from pynn_genn.random import RandomDistribution, NativeRNG

# expose only the Connectors defined in PyNN
__all__ = [
    "AllToAllConnector", "OneToOneConnector",
    "FixedProbabilityConnector", "FixedTotalNumberConnector",
    "FixedNumberPreConnector", "FixedNumberPostConnector",
    "DistanceDependentProbabilityConnector",
    "DisplacementDependentProbabilityConnector",
    "IndexBasedProbabilityConnector", "SmallWorldConnector",
    "FromListConnector", "FromFileConnector",
    "CloneConnector", "ArrayConnector"
]


class GeNNConnectorMixin(object):
    _builtin_inits = {
        'OneToOneConnector': 'OneToOne',
        'FixedProbabilityConnector': 'FixedProbability',
        'FixedNumberPostConnector': 'FixedNumberPostWithReplacement',
        'FixedTotalNumberConnector': 'FixedNumberTotalWithReplacement'
    }
    def __init__(self, on_device_init=False, use_sparse=True):
        self.on_device_init = on_device_init
        self.use_sparse = use_sparse
        self.on_device_init_params = {}

    def connect(self, projection):
        # If we are going to expand on device, we don't need to generate masks,
        # pre- or post indices
        if self.on_device_init:
            param_space = self._parameters_from_synapse_type(projection)
            # Evaluate the lazy arrays containing the synaptic parameters
            connection_parameters = {}
            for name, map in param_space.items():
                if map.is_homogeneous:
                    connection_parameters[name] = map.evaluate(simplify=True)
                else:
                    connection_parameters[name] = map

            projection._on_device_connect(projection.pre.size,
                                          projection.post.size,
                                          **connection_parameters)
        # Otherwise, just go for the standard connect method and optimize
        # whenever possible
        else:
            # get parents, we know that the first one is this Mixin and the
            # second is the PyNN connector specific class
            mixin, pynn_conn = self.__class__.__bases__
            pynn_conn.connect(self, projection)

    def _parameters_from_synapse_type(self, projection, distance_map=None):
        """
        Obtain the parameters to be used for the connections from the projection's `synapse_type`
        attribute. Each parameter value is a `LazyArray`.
        """
        if distance_map is None:
            distance_map = Connector._generate_distance_map(self, projection)

        parameter_space = projection.synapse_type.native_parameters
        # TODO: in the documentation, we claim that a parameter value can be
        #       a list or 1D array of the same length as the number of connections.
        #       We do not currently handle this scenario, although it is only
        #       really useful for fixed-number connectors anyway.
        #       Probably the best solution is to remove the parameter at this stage,
        #       then set it after the connections have already been created.
        parameter_space.shape = (projection.pre.size, projection.post.size)

        # Remove randomly generated variables from the (host) parameter_space
        # if the user so chooses. We keep a copy of the removed (not expanded)
        # parameters in the Connector object
        if self.on_device_init:
            pops = []
            for name, map in parameter_space.items():
                # if len(map.operations):
                #     continue
                if ((isinstance(map.base_value, RandomDistribution) and
                        isinstance(map.base_value.rng, NativeRNG)) or
                        map.is_homogeneous):

                    self.on_device_init_params[name] = map
                    pops.append(name)

            for name in pops:
                parameter_space.pop(name)

        for name, map in parameter_space.items():
            if callable(map.base_value):
                if isinstance(map.base_value, IndexBasedExpression):
                    # Assumes map is a function of index and hence requires the projection to
                    # determine its value. It and its index function are copied so as to be able
                    # to set the projection without altering the connector, which would perhaps
                    # not be expected from the 'connect' call.
                    new_map = copy(map)
                    new_map.base_value = copy(map.base_value)
                    new_map.base_value.projection = projection
                    parameter_space[name] = new_map
                else:
                    # Assumes map is a function of distance
                    parameter_space[name] = map(distance_map)
        return parameter_space

    def _init_connectivity(self, projection):
        class_name = self.__class__.__name__
        params = self._get_conn_init_params(projection)
        if class_name in self._builtin_inits:
            builtin_name = self._builtin_inits[class_name]
            return init_connectivity(builtin_name, params)

    def _get_conn_init_params(self, projection):
        return {}

class OneToOneConnector(GeNNConnectorMixin, OneToOnePyNN):
    __doc__ = OneToOnePyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(
            self, on_device_init=on_device_init)
        OneToOnePyNN.__init__(
            self, safe=safe, callback=callback)


class AllToAllConnector(GeNNConnectorMixin, AllToAllPyNN):
    __doc__ = AllToAllPyNN.__doc__

    def __init__(self, allow_self_connections=True, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init, use_sparse=False)
        AllToAllPyNN.__init__(
                            self, allow_self_connections=allow_self_connections,
                            safe=safe, callback=callback)


class FixedProbabilityConnector(GeNNConnectorMixin, FixProbPyNN):
    __doc__ = FixProbPyNN.__doc__

    def __init__(self, p_connect, allow_self_connections=True,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        FixProbPyNN.__init__(self, p_connect, allow_self_connections,
                 rng, safe=safe, callback=callback)

    def _get_conn_init_params(self, projection):
        return {'prob': self.p_connect}

class FixedTotalNumberConnector(GeNNConnectorMixin, FixTotalPyNN):
    __doc__ = FixTotalPyNN.__doc__

    def __init__(self, n, allow_self_connections=True, with_replacement=False,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        FixTotalPyNN.__init__(self, n, allow_self_connections, with_replacement,
                              rng, safe=safe, callback=callback)

    def _get_conn_init_params(self, projection):
        return {'total': self.n}


class FixedNumberPreConnector(GeNNConnectorMixin, FixNumPrePyNN):
    __doc__ = FixNumPrePyNN.__doc__

    def __init__(self, n, allow_self_connections=True, with_replacement=False,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        FixNumPrePyNN.__init__(self, n, allow_self_connections, with_replacement,
                               rng, safe=safe, callback=callback)


class FixedNumberPostConnector(GeNNConnectorMixin, FixNumPostPyNN):
    __doc__ = FixNumPostPyNN.__doc__

    def __init__(self, n, allow_self_connections=True, with_replacement=False,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        FixNumPostPyNN.__init__(self, n, allow_self_connections,
                            with_replacement, rng, safe=safe, callback=callback)

    def _get_conn_init_params(self, projection):
        return {'rowLength': self.n}

class DistanceDependentProbabilityConnector(GeNNConnectorMixin, DistProbPyNN):
    __doc__ = DistProbPyNN.__doc__

    def __init__(self, d_expression, allow_self_connections=True,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        DistProbPyNN.__init__(self, d_expression, allow_self_connections,
                 rng, safe=safe, callback=callback)


class DisplacementDependentProbabilityConnector(
                                        GeNNConnectorMixin, DisplaceProbPyNN):
    __doc__ = DisplaceProbPyNN.__doc__

    def __init__(self, disp_function, allow_self_connections=True,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        DisplaceProbPyNN.__init__(self, disp_function, allow_self_connections,
                 rng, safe=safe, callback=callback)


class IndexBasedProbabilityConnector(GeNNConnectorMixin, IndexProbPyNN):
    __doc__ = IndexProbPyNN.__doc__

    def __init__(self, index_expression, allow_self_connections=True,
                 rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        IndexProbPyNN.__init__(self, index_expression, allow_self_connections,
                 rng, safe=safe, callback=callback)


class SmallWorldConnector(GeNNConnectorMixin, SmallWorldPyNN):
    __doc__ = SmallWorldPyNN.__doc__

    def __init__(self, degree, rewiring, allow_self_connections=True,
                 n_connections=None, rng=None, safe=True, callback=None,
                 on_device_init=False):
        GeNNConnectorMixin.__init__(self, on_device_init)
        SmallWorldPyNN.__init__(self, degree, rewiring, allow_self_connections,
                 n_connections, rng,  safe=safe, callback=callback)


class FromListConnector(GeNNConnectorMixin, FromListPyNN):
    __doc__ = FromListPyNN.__doc__

    def __init__(self, conn_list, column_names=None, safe=True, callback=None):
        GeNNConnectorMixin.__init__(self, on_device_init=False)
        FromListPyNN.__init__(self, conn_list, column_names,
                              safe=safe, callback=callback)


class FromFileConnector(GeNNConnectorMixin, FromFilePyNN):
    __doc__ = FromFilePyNN.__doc__

    def __init__(self,  file, distributed=False, safe=True, callback=None):
        GeNNConnectorMixin.__init__(self, on_device_init=False)
        FromFilePyNN.__init__(self, file, distributed,
                              safe=safe, callback=callback)


class CloneConnector(GeNNConnectorMixin, ClonePyNN):
    __doc__ = ClonePyNN.__doc__

    def __init__(self, reference_projection, safe=True, callback=None):
        conn = reference_projection._connector
        on_device_init = conn.on_device_init
        use_sparse = conn.use_sparse
        GeNNConnectorMixin.__init__(self, on_device_init,
                                    use_sparse=use_sparse)
        ClonePyNN.__init__(self, reference_projection, safe=safe,
                           callback=callback)
        self.on_device_init_params = deepcopy(conn.on_device_init_params)

class ArrayConnector(GeNNConnectorMixin, ArrayPyNN):
    __doc__ = ArrayPyNN.__doc__

    def __init__(self, array, safe=True, callback=None):
        GeNNConnectorMixin.__init__(self, on_device_init=False)
        ArrayPyNN.__init__(self, array, safe=safe, callback=callback)

