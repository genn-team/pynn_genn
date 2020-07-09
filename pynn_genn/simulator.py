from pyNN import common
from pygenn import GeNNModel
from six import iteritems, itervalues

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
        self._min_delay = "auto"
        self.clear()
        self.dt = 0.1
        self.t = 0.0
        self.num_threads_per_spike = 8
        self.num_current_sources = 0
        self.native_rng = None

    @property
    def dt( self ):
        return self.model.dT

    @dt.setter
    def dt( self, _dt ):
        self.model.dT = _dt

    @property
    def min_delay(self):
        if self._min_delay == "auto":
            if len(self.projections) == 0:
                return self.dt
            else:
                return min(p.min_delay for p in self.projections)
        else:
            return self._min_delay

    @min_delay.setter
    def min_delay(self, d):
        self._min_delay = d

    def finalize(self):
        if not (self.native_rng is None):
            self.model._model.set_seed(0 if self.native_rng.seed is None 
                                       else self.native_rng.seed)

        for pop in self.populations:
            pop._create_native_population()
        for proj in self.projections:
            proj._create_native_projection()

        self.model.build(self.model_path)
        self.model.load(self.model_path)

        self._built = True

    def run(self, simtime):
        self.run_until(self.t + simtime)

    def run_until(self, tstop):
        if not self._built:
            self.finalize()
        if not self.running:
            for rec in self.recorders:
                rec.init_data_views()

        # Synchronise model with our timestep
        self.model.t = self.t
        self.model.timestep = int(round(self.t / self.dt))

        # Simulate
        self.running = True
        while self.t < tstop:
            # Get time at start of step (this is correct timestamp for recorded data)
            timestep = self.model.timestep

            # Record any variables being recorded
            for rec in self.recorders:
                rec._record_vars(timestep)

            # Also update our t from the GeNN timestep
            # **NOTE** GeNN steps time at end of timestep and PyNN expects opposite
            # **NOTE** GeNN often using 32-bit float timesteps which fail a lot of tests
            self.t = timestep * float(self.dt)

            # Advance model time
            self.model.step_time()

    def clear(self):
        self.model = GeNNModel("float", "GeNNModel")
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

        # reinitialise model if it has already been built
        if self._built:
            self.model.reinitialise()

        self.segment_counter += 1

state = State()
