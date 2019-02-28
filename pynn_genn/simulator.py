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
        self.num_current_sources = 0

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
        for pop in self.populations:
            pop._create_native_population()
        for proj in self.projections:
            proj._create_native_projection()
        self.model.build(self.model_path)
        self.model.load()

        self._built = True

    def run_until(self, tstop):
        if not self._built:
            self.finalize()
        if not self.running:
            for rec in self.recorders:
                rec.init_data_views()

        # Synchronise model with our timestep
        self.model.t = self.t
        self.model.timestep = int(round(self.t / self.dt))

        # Calculate corresponding timestep to tstop
        timestep_stop = int(round(tstop / self.dt))

        # Simulate
        self.running = True
        while self.model.timestep < timestep_stop:
            # Record any variables being recorded
            # **NOTE** this is essentially recording the state at the end of the LAST timestep
            for rec in self.recorders:
                rec._record_vars(self.model.timestep)

            # Advance model time
            self.model.step_time()

            # Update PyNN's t from the GeNN timestep (which will have been updated during the call to step_time)
            # **NOTE** Internally, GeNN often uses 32-bit float timesteps which fail a lot of tests so we totally disregard the,
            self.t = self.model.timestep * float(self.dt)

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
