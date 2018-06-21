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
        self.dt = 0.1
        self._built = False
        self.populations = []
        self.projections = []

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


        
        #  self.model.initializeVarOnDevice( 'Stim_population1', 'g', [0], [-0.2] )
        #
        #  self.model.initializeSpikesOnDevice( 'Stim', [0], [0], [1] )


        self._built = True

    def test_spikes(self):
       self.model.addNeuronPopulation( 'Stim', 1, 'SpikeSource', [], [] )
       self.model.addSynapsePopulation( 'Stim_population1', 'DENSE_INDIVIDUALG',
                0, 'Stim', 'population1',
                'StaticPulse', {}, {'g' : 0.0},
                'ExpCond', {'tau' : 1.0, 'E' : -80.0}, {},
                customWeightUpdate=True, customPostsynaptic=True)

        
    def run(self, simtime):
        if not self._built:
            self.finalize()
        if not self.running:
            for pop in self.populations:
                pop._initialize_native_population()
            for proj in self.projections:
                proj._initialize_native_projection()
        self.running = True
        for i in range( int( simtime / self.dt ) ):
            self.model.stepTimeGPU()
            self.t += self.dt
            for rec in self.recorders:
                rec._read_vars()
            
    def run_until(self, tstop):
        if not self._built:
            self.first_run()
        if not self.running:
            for pop in self.populations:
                pop._initialize_native_population()
            for proj in self.projections:
                proj._initialize_native_projection()
        self.running = True
        while self.t < tstop:
            self.model.stepTimeGPU()
            self.t += self.dt
            for rec in self.recorders:
                rec._read_vars()

    def clear(self):
        self.recorders = set([])
        self.id_counter = 42
        self.segment_counter = -1
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

state = State()
