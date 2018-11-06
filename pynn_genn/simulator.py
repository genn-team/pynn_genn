from pyNN import common
from pygenn import GeNNModel

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
        self.min_delay = 0
        self.clear()
        self.dt = 0.1
        self.num_current_sources = 0

    @property
    def t( self ):
        if self._built:
            return self.model.t
        else:
            return 0.0

    @t.setter
    def t(self, t):
        if self._built:
            self.model.t = t
            self.model.timestep = int(round(t / self.dt))
        elif t != 0.0:
            assert False

    @t.setter
    def dt( self, _dt ):
        self.model.dT = _dt

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
        self.model.build(self.model_path)
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

        # Simulate
        self.running = True
        while True:
            # Get time at start of step (this is correct timestamp for recorded data)
            timestep = self.model.timestep

            # Advance model time
            self.model.step_time()

            # Record any variables being recorded
            for rec in self.recorders:
                rec._record_vars(timestep)

            # If we've passed stop time, stop
            if self.t > tstop:
                break

    def clear(self):
        self.model = GeNNModel('float', 'GeNNModel')
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
