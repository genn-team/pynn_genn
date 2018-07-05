from pyNN import common
import GeNNModel

name = "genn"


class ID(int, common.IDMixin):

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


class State(common.control.BaseState):

    def __init__(self):
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self.model = GeNNModel.GeNNModel('float', 'GeNNModel')
        self.min_delay = 0
        self.clear()
        self.use_sparse = False
        self.dt = 0.1

    @property
    def tt( self ):
        return self.model.T

    @property
    def dt( self ):
        return self.model.dT

    @dt.setter
    def dt( self, _dt ):
        self.model.dT = _dt

    def finalize(self):
        for pop in self.populations:
            pop._create_native_population()
        for proj in self.projections:
            proj._create_native_projection()
        self.model.build()
        self.model.load()

        self._built = True
        
    def run(self, simtime):
        self.run_until(self.t + simtime)
            
    def run_until(self, tstop):
        if not self._built:
            self.finalize()
        if not self.running:
            for rec in self.recorders:
                rec.init_data_views()
        self.running = True
        while self.t < tstop:
            self.model.stepTimeGPU()
            self.t += self.dt
            for rec in self.recorders:
                rec._record_vars(self.t)

    def clear(self):
        self.populations = []
        self.projections = []
        self.recorders = set([])
        self.id_counter = 0
        self.segment_counter = -1
        self._built = False
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        if self._built:
            self.model.load()
        self.segment_counter += 1

state = State()
