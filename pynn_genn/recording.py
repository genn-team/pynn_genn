import numpy as np
from six import iteritems
from pyNN import recording
from . import simulator

class Monitor(object):

    def __init__(self, parent):
        self.recorder = parent
        self.start_id = parent.population.first_id
        self.data = None
        self.time = None
        self.id_data_idx_map = {}

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
            self.data = [[] for _ in new_ids]
            self.time = []
        else:
            old_id_len = len(self.id_set)
            self.id_set = self.id_set.union(new_ids)
            [self.data.append([]) for i in range(len(self.id_set) - old_id_len)]

        iimap_len = len(self.id_data_idx_map)
        self.id_data_idx_map.update({idd - self.start_id : i + iimap_len 
                                     for i, idd in enumerate(new_ids)})

        self.sampling_interval = sampling_interval or 1

    def get_data(self, ids):
        if isinstance(ids, list):
            ids = [idd - self.start_id for idd in ids]
        else:
            ids = ids - self.start_id

        data_ids = [self.id_data_idx_map[idd] for idd in ids]

        return (np.array(self.data)[data_ids,:].T 
                if len(self.data) > 1 
                else np.array(self.data).T)

    def get_time(self):
        return self.time

    def enlarge_storage(self, idx):
        self.data_size[idx] *= 2
        self.data[idx].resize((self.data_size[idx],))
        if self.data_size[idx]/2 <= len(self.time):
            self.time.resize((self.data_size[idx],))

    def __call__(self, t):
        """Fetch new data"""

        if (t % self.sampling_interval < 1):
            self.time.append(t)
            for idd, i in iteritems(self.id_data_idx_map):
                self.data[i].append(np.copy(self.data_view[idd]))


class StateMonitor(Monitor):

    def __init__(self, parent, variable):
        super(StateMonitor, self).__init__(parent)
        self.var = variable
        self.translated = (parent.population.celltype.translations
                [variable]['translated_name'])

    def init_data_view(self):
        self.data_view = (self.recorder.population._pop.vars[self.translated].view)


class SpikeMonitor(Monitor):

    def __init__(self, parent):
        super(SpikeMonitor, self).__init__(parent)

    def init_data_view(self):
        pass

    def get_data(self, ids):
        return self.data[ids-self.start_id]

    def __call__(self, t):
        """Fetch new data"""

        if (t % self.sampling_interval < 1):
            for i in self.recorder.population._pop.current_spikes:
                if i in self.id_data_idx_map:
                    self.data[self.id_data_idx_map[i]].append(t)


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
        if len(self.recorded) > 0:
            if 'spikes' in self.recorded:
                self._simulator.state.model.pull_current_spikes_from_device(self.population._genn_label)
            if len(self.recorded) - ('spikes' in self.recorded) > 0:
                self._simulator.state.model.pull_state_from_device(self.population._genn_label)

        for monitor in self.monitors.values():
            monitor(t)

    def _get_spiketimes(self, id):
        if 'spikes' not in self.monitors:
            spikes = np.array([])
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
