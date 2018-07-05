import numpy
from pyNN import recording
from . import simulator

class Monitor(object):

    def __init__(self, parent):
        self.recorder = parent
        self.start_id = parent.population.first_id
        self.data = None
        self.time = None

    @property
    def id_set(self):
        return self._id_set

    @id_set.setter
    def id_set(self, _id_set):
        self._id_set = _id_set
        self.ids = [idd - self.start_id for idd in self._id_set]

    def record(self, new_ids, sampling_interval):

        if self.data is None:
            self.id_set = new_ids
            self.data = numpy.empty((1000, len(new_ids)), dtype=float)
            self.time = numpy.empty((1000, 1), dtype=float )
            self.data_size = len(self.data)
            self.cur = 0
        else:
            self.id_set = self.id_set.union(new_ids)
            self.data.resize((self.data.shape[0], len(self.id_set)))

        self.sampling_interval = sampling_interval or 1

    def get_data(self, ids):
        ids = [idd - self.start_id for idd in self._id_set]
        return self.data[:self.cur, ids]

    def get_time(self):
        return self.time[:self.cur]

    def enlarge_storage(self):
        data_shape = list(self.data.shape)
        data_shape[0] *= 2
        self.data.resize(data_shape)
        self.time.resize((data_shape[0], 1))
        self.data_size = data_shape[0]

    def __call__(self, t):
        """Fetch new data"""

        if (t % self.sampling_interval < 1):
            self.time[self.cur] = t
            self.data[self.cur, :] = self.data_view[self.ids]

            self.cur += 1
            if self.cur == self.data_size:
                self.enlarge_storage()

class StateMonitor(Monitor):

    def __init__(self, parent, variable):
        super(StateMonitor, self).__init__(parent)
        self.var = variable
        self.translated = (parent.population.celltype.translations
                [variable]['translated_name'])

    def init_data_view(self):
        self.data_view = (self.recorder._simulator.state.model.neuronPopulations
                [self.recorder.population.label].vars[self.translated].view)


class SpikeMonitor(Monitor):

    def __init__(self, parent):
        super(SpikeMonitor, self).__init__(parent)

    def init_data_view(self):
        self.data_view = (self.recorder._simulator.state.model.neuronPopulations
                [self.recorder.population.label].spikeCount)


class Recorder(recording.Recorder):
    _simulator = simulator

    def __init__(self, population, file=None):
        super(Recorder, self).__init__(population, file)
        self.monitors = {}

    def _record(self, variable, new_ids, sampling_interval=None):
        if variable not in self.monitors:
            if variable == 'spikes':
                self.monitors[variable] = SpikeMonitor(self)
            else:
                self.monitors[variable] = StateMonitor(self, variable)
            self.monitors[variable].record(new_ids, sampling_interval)

    def init_data_views(self):
        for monitor in self.monitors.values():
            monitor.init_data_view()

    def _record_vars(self, t):

        self._simulator.state.model.pullPopulationSpikesFromDevice(self.population.label)
        if len(self.recorded) > 0:
            if 'spikes' in self.recorded:
                self._simulator.state.model.pullPopulationSpikesFromDevice(self.population.label)
            if len(self.recorded) - ('spikes' in self.recorded) > 0:
                self._simulator.state.model.pullPopulationStateFromDevice(self.population.label)

        for monitor in self.monitors.values():
            monitor(t)

    def _get_spiketimes(self, id):
        if 'spikes' not in self.monitors:
            spikes = numpy.array([])
        else:
            spikes = self.monitors['spikes'].get_data(id)
        return spikes

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        return self.monitors[variable].get_data(ids)

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
