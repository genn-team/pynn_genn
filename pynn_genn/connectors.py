from pyNN.connectors import (AllToAllConnector,
                             FixedProbabilityConnector,
                             FixedTotalNumberConnector,
                             OneToOneConnector,
                             FixedNumberPreConnector,
                             FixedNumberPostConnector,
                             DistanceDependentProbabilityConnector,
                             DisplacementDependentProbabilityConnector,
                             IndexBasedProbabilityConnector,
                             SmallWorldConnector,
                             FromListConnector,
                             FromFileConnector,
                             CloneConnector,
                             ArrayConnector)

class OneToOneConnector(OneToOneConnector):
    def __init__(self, on_device=True, safe=True, callback=None):
        super(OneToOneConnector, self).__init(safe, callback)
        self._on_device = on_device

    def _code(self):
        code = """
            $(addSynapse, $(id_pre));
            $(endRow);
        """
        return code

    def row_length(self):
        return 1

    def col_length(self):
        return 1

    def connect(self, projection):
        pass