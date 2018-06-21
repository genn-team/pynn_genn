import numpy
from pyNN import recording
from . import simulator


class Recorder(recording.Recorder):
    _simulator = simulator

    def __init__( self, population, file=None ):
        super(Recorder, self).__init__(population, file)
        self.vars = {}

    def _record(self, variable, new_ids, sampling_interval=None):
        start_id = self.population.all_cells[0]
        self.vars[variable] = { 
                'data' : numpy.empty( (1000, len(new_ids) ), dtype=numpy.float32 ),
                'cur' : 0,
                'ids' : [idd - start_id for idd in new_ids],
                'si' : sampling_interval or 1,
                'translated_name' : (self.population.celltype.translations
                    [variable]['translated_name'][len('neuron_'):])
        }

    def _read_vars(self):
        
        self._simulator.state.model.pullPopulationSpikesFromDevice( self.population.label )
        if len( self.recorded ) > 0:
            if 'spikes' in self.recorded:
                self._simulator.state.model.pullPopulationSpikesFromDevice( self.population.label )
            if len( self.recorded ) - ('spikes' in self.recorded) > 0:
                self._simulator.state.model.pullPopulationStateFromDevice( self.population.label )

        for var_name, var_data in self.vars.items():
            if var_data['cur'] >= var_data['data'].shape[0]:
                # double the array size if data limit is reached
                data_shape = list(var_data['data'].shape)
                data_shape[0] *= 2
                var_data['data'].resize( data_shape )
            
            
            var_data['data'][var_data['cur'], :] = (self._simulator.state.model
                    .neuronPopulations[self.population.label]['vars']
                    [var_data['translated_name']][var_data['ids']])

            var_data['cur'] += 1



    def _get_spiketimes(self, id):
        return numpy.array([id, id + 5], dtype=float) % self._simulator.state.t

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        n_samples = int(round(self._simulator.state.t / self._simulator.state.dt)) + 1
        return numpy.vstack((numpy.random.uniform(size=n_samples) for id in ids)).T

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = 2
        else:
            raise Exception("Only implemented for spikes")
        return N

    def _clear_simulator(self):
        pass

    def _reset(self):
        pass
