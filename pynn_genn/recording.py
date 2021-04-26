import numpy as np
from math import fmod
from six import iteritems, itervalues
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

    def record(self, new_ids, sampling_timesteps):
        self.sampling_timesteps = sampling_timesteps

        # If no data has yet been allocated
        if self.data is None:
            self.id_set = new_ids
            self.data = [[] for _ in new_ids]
            self.time = []
        else:
            old_id_len = len(self.id_set)
            self.id_set = self.id_set.union(new_ids)

            self.data.extend([] for _ in range(len(self.id_set) - old_id_len))

        iimap_len = len(self.id_data_idx_map)
        self.id_data_idx_map.update({idd - self.start_id: i + iimap_len
                                     for i, idd in enumerate(new_ids)})

    def get_data(self, ids):
        if isinstance(ids, list):
            ids = [idd - self.start_id for idd in ids]
        else:
            ids = ids - self.start_id

        data_ids = [self.id_data_idx_map[idd] for idd in ids]

        return (np.array(self.data)[data_ids, :].T
                if len(self.data) > 1
                else np.array(self.data).T)

    def get_time(self):
        return self.time

    def __call__(self, timestep):
        """Fetch new data"""
        if timestep % self.sampling_timesteps == 0:
            self.time.append(timestep * self.recorder._simulator.state.dt)
            for idd, i in iteritems(self.id_data_idx_map):
                # **TODO** we could just stack numpy arrays
                self.data[i].append(np.copy(self.data_view[idd]))

    def store_to_cache(self):
        # If anything is being recorded
        if self.data is not None:
            # Empty list of times
            self.time = []

            # Create an empty list to hold recorded data for each ID
            self.data = [[] for _ in range(len(self.id_set))]


class StateMonitor(Monitor):
    def __init__(self, parent, variable):
        super(StateMonitor, self).__init__(parent)
        parent_translations = parent.population.celltype.translations
        self.translated = (parent_translations[variable]["translated_name"])

    def init_data_view(self):
        vars = self.recorder.population._pop.vars
        self.data_view = vars[self.translated].view


class SpikeMonitor(Monitor):
    def __init__(self, parent):
        super(SpikeMonitor, self).__init__(parent)

    def init_data_view(self):
        pass

    def get_data(self, ids):
        return self.data[ids-self.start_id]

    def __call__(self, timestep):
        """Fetch new data"""
        if timestep % self.sampling_timesteps == 0:
            t = timestep * self.recorder._simulator.state.dt
            for i in self.recorder.population._pop.current_spikes:
                if i in self.id_data_idx_map:
                    self.data[self.id_data_idx_map[i]].append(t)


class Recorder(recording.Recorder):
    _simulator = simulator

    def __init__(self, population, file=None):
        super(Recorder, self).__init__(population, file)
        self.monitors = {}

    def _record(self, variable, new_ids, sampling_interval=None):
        # Cache sampling interval
        # **NOTE** base class sets default
        if sampling_interval is not None:
            self.sampling_interval = sampling_interval

        # Convert to timesteps
        sampling_timesteps =\
            int(round(self.sampling_interval / self._simulator.state.dt))

        # If there isn't already a monitor for this variable
        if variable not in self.monitors:
            if variable == "spikes":
                self.monitors[variable] = SpikeMonitor(self)
            else:
                self.monitors[variable] = StateMonitor(self, variable)

        # Tell monitor to record these ids
        self.monitors[variable].record(new_ids, sampling_timesteps)

    def init_data_views(self):
        for monitor in itervalues(self.monitors):
            monitor.init_data_view()

    def _record_vars(self, t):
        model = self._simulator.state.model
        label = self.population._genn_label

        # Loop through monitors
        for var, monitor in iteritems(self.monitors):
            # If this variable is spikes, pull current spikes
            if var == "spikes":
                model.pull_current_spikes_from_device(label)
            # Otherwise pull variable (using translated name cached in monitor)
            else:
                model.pull_var_from_device(label, monitor.translated)

            # Update monitor
            monitor(t)

    def _get_spiketimes(self, id):
        if "spikes" not in self.monitors:
            spikes = np.array([])
        else:
            spikes = self.monitors["spikes"].get_data(id)
        return spikes

    def _get_all_signals(self, variable, ids, clear=False):
        # assuming not using cvode, otherwise need to get times as
        # well and use IrregularlySampledAnalogSignal
        return self.monitors[variable].get_data(ids)

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == "spikes":
            for id in self.filter_recorded(variable, filter_ids):
                id = int(id)
                data_idx = mon.id_data_idx_map[id - mon.start_id]
                N[id] = len(mon.data[data_idx])
        else:
            raise Exception("Only implemented for spikes")
        return N

    def _clear_simulator(self):
        pass

    def _reset(self):
        # Reset what is recorded
        self.monitors = {}

    def store_to_cache(self, annotations=None):
        # Allow base recorder to do ITS reinitialisation
        super(Recorder, self).store_to_cache(annotations)

        # Clear out data
        for m in itervalues(self.monitors):
            m.store_to_cache()
