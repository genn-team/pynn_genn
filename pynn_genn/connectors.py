from copy import copy
from pyNN.core import IndexBasedExpression
from pyNN.parameters import LazyArray
from pygenn.genn_model import create_custom_sparse_connect_init_snippet_class
from pyNN.connectors import (
    AllToAllConnector as AllToAllPyNN,
    OneToOneConnector as OneToOnePyNN,
    # FixedProbabilityConnector as FixProbPyNN,
    # FixedTotalNumberConnector as FixTotalPyNN,
    # FixedNumberPreConnector as FixNumPrePyNN,
    # FixedNumberPostConnector as FixNumPostPyNN,
    # DistanceDependentProbabilityConnector as DistProbPyNN,
    # DisplacementDependentProbabilityConnector as DisplaceProbPyNN,
    # IndexBasedProbabilityConnector as IndexProbPyNN,
    # SmallWorldConnector as SmallWorldPyNN,
    # FromListConnector as FromListPyNN,
    # FromFileConnector as FromFilePyNN,
    # CloneConnector as ClonePyNN,
    # ArrayConnector as ArrayPyNN
)

from pynn_genn.random import RandomDistribution, NativeRNG

__all__ = [
    "AllToAllConnector", "OneToOneConnector",
    # "FixedProbabilityConnector", "FixedTotalNumberConnector",
    # "FixedNumberPreConnector", "FixedNumberPostConnector",
    # "DistanceDependentProbabilityConnector",
    # "DisplacementDependentProbabilityConnector",
    # "IndexBasedProbabilityConnector", "SmallWorldConnector",
    # "FromListConnector", "FromFileConnector",
    # "CloneConnector", "ArrayConnector"
]

class AbstractGeNNConnector(object):

    # __slots__ = ['_on_device', '_procedural', '_row_length', '_col_length',
    #              '_sparse']
    _row_code = None
    _row_state_vars = None

    def __init__(self, on_device_init=False, procedural=False):
        self._on_device_init = on_device_init
        self._procedural = procedural
        self._row_length = 0
        self._col_length = 0
        self._sparse = True
        self._on_device_params = {}
        self._conn_params = {}

    def _parameters_from_synapse_type(self, projection, distance_map=None):
        # print("in genn _parameters_from_synapse_type")
        """
        Obtain the parameters to be used for the connections from the projection's `synapse_type`
        attribute. Each parameter value is a `LazyArray`.
        """
        if distance_map is None:
            distance_map = self._generate_distance_map(projection)

        parameter_space = projection.synapse_type.native_parameters

        # Remove randomly generated variables from the (host) parameter_space
        # if the user so chooses. We keep a copy of the removed (not expanded)
        # parameters in the Connector object
        if self.on_device_init or self.procedural:
            pops = []
            for name, map in parameter_space.items():
                if isinstance(map.base_value, RandomDistribution) and \
                    isinstance(map.base_value.rng, NativeRNG):
                        self._on_device_params[name] = map
                        pops.append(name)

            for name in pops:
                parameter_space.pop(name)

        # TODO: in the documentation, we claim that a parameter value can be
        #       a list or 1D array of the same length as the number of connections.
        #       We do not currently handle this scenario, although it is only
        #       really useful for fixed-number connectors anyway.
        #       Probably the best solution is to remove the parameter at this stage,
        #       then set it after the connections have already been created.
        parameter_space.shape = (projection.pre.size, projection.post.size)
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

    @property
    def use_sparse(self):
        return self._sparse

    @property
    def row_length(self):
        return self._row_length

    @property
    def col_length(self):
        return self._col_length

    @property
    def on_device_init(self):
        return self._on_device_init
    @property
    def on_device_init_params(self):
        return self._on_device_params

    @property
    def procedural(self):
        return self._procedural

    def compute_use_sparse(self, projection):
        """
        compute if the projection requires a sparse or full matrix
        :param projection:
        :return:
        """
        pass

    def compute_row_length(self, projection):
        raise NotImplementedError()

    def compute_col_length(self, projection):
        raise NotImplementedError()

    def get_params(self):
        return self._conn_params

    def init_sparse_conn_snippet(self, **kwargs):
        kwargs['row_build_code'] = self._row_code
        kwargs['row_build_state_vars'] = self._row_state_vars
        return create_custom_sparse_connect_init_snippet_class(
                                            self.__class__.__name__, **kwargs)

class OneToOneConnector(AbstractGeNNConnector, OneToOnePyNN):
    _row_code = """
        $(addSynapse, $(id_pre));
        $(endRow);
    """

    __doc__ = OneToOnePyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        OneToOnePyNN.__init__(self, safe=safe, callback=callback)
        self._row_length = 1
        self._col_length = 1
        self._sparse = True

    # def connect(self, projection):
    #     # if not (self.on_device_init or self.procedural):
    #     OneToOnePyNN.connect(self, projection)

    def init_sparse_conn_snippet(self):
        args = {}
        return AbstractGeNNConnector.init_sparse_conn_snippet(self, **args)

class AllToAllConnector(AbstractGeNNConnector, AllToAllPyNN):
    _row_code = """
        if($(id_pre) < $(num_pre)){
            $(addSynapse, $(id_pre));
        } else {
            $(endRow);
        }
    """

    __doc__ = AllToAllPyNN.__doc__

    def __init__(self, allow_self_connections=True, safe=True, callback=None,
                 on_device_init=True, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        AllToAllPyNN.__init__(
                            self, allow_self_connections=allow_self_connections,
                            safe=safe, callback=callback)
        self._sparse = False
