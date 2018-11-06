# encoding: utf-8
"""
Standard electrodes for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
import numpy as np
from functools import partial
from lazyarray import larray
from pyNN.standardmodels import electrodes, build_translations#, StandardCurrentSource
from ..simulator import state
import logging
from pygenn.genn_model import create_custom_current_source_class
from ..model import GeNNStandardCurrentSource, GeNNDefinitions

logger = logging.getLogger("PyNN")

# Function to convert mean 'rate' parameter 
# to mean interspike interval (in timesteps)
def mulDT(param_name, dt, **kwargs):
    return kwargs[param_name] * dt

def freqToOmega(frequency, **kwargs):
    return frequency * 2.0 * np.pi / 1000.0

def phaseToRad(phase, **kwargs):
    return phase / 180.0 * np.pi

def stepStart(times, **kwargs):
    return larray(times.base_value.value[0])

def stepStop(times, **kwargs):
    return larray(times.base_value.value[-1])

class DCSource(GeNNStandardCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__
    genn_currentsource_name = 'DCSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injection_code' : '''
            if ($(applyIinj) && $(t) >= $(tStart) && $(t) < $(tStop)) {
                $(injectCurrent, $(amplitude));
            }
        ''',
        'var_name_types': [
            ('applyIinj', "unsigned char")],
        'param_name_types' : {
            'tStart': 'scalar',
            'tStop': 'scalar',
            'amplitude': 'scalar'}
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('amplitude',  'amplitude')
    ),
    # extra param values
    {
        'applyIinj' : 0
    })

class StepCurrentSource(GeNNStandardCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__
    genn_currentsource_name = 'StepCurrentSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injection_code' : '''
            if ($(applyIinj)) {
                if ($(startStep) < ($(endStep) - 1) && $(t) >= $(stepTimes)[$(startStep) + 1]) {
                    $(startStep)++;
                }

                if($(t) >= $(stepTimes)[$(startStep)]) {
                    $(injectCurrent, $(stepAmpls)[$(startStep)]);
                }
            }
        ''',
        'var_name_types': [
            ('applyIinj', "unsigned char"),
            ("startStep", "unsigned int"),
            ("endStep", "unsigned int")],
        'extra_global_params' : [('stepAmpls', 'scalar*'), ('stepTimes', 'scalar*')]
    },
    # translations
    (
        ('amplitudes',  'stepAmpls'),
        ('times',       'stepTimes')
    ))

    def get_extra_global_params(self, native_params):
        # Concatenate together step amplitudes and times to form extra global parameter
        return {
            "stepAmpls" : np.concatenate([seq.value
                                           for seq in native_params["stepAmpls"]]),
            "stepTimes" : np.concatenate([seq.value
                                           for seq in native_params["stepTimes"]])}

    def build_genn_current_source(self, native_params):
        # Create model using unmodified defs
        genn_model = create_custom_current_source_class(
            self.genn_currentsource_name, **self.currentsource_defs.definitions)()

        # Get spike times
        step_times = native_params["stepTimes"]

        # Create empty numpy arrays to hold start and end spikes indices
        start_step = np.empty(shape=native_params.shape, dtype=np.float32)
        end_step = np.empty(shape=native_params.shape, dtype=np.float32)

        # Calculate indices for each sequence
        cum_size = 0
        for i, seq in enumerate(step_times):
            start_step[i] = cum_size
            cum_size += len(seq.value)
            end_step[i] = cum_size

        # Build initialisation dictionary
        cs_ini = {"startStep": start_step,
                  "endStep": end_step,
                  "applyIinj": np.zeros(shape=native_params.shape, dtype=np.uint8)}

        # Return with model
        return genn_model, [], cs_ini

    def get_extra_global_params(self, native_params):
        # Concatenate together step amplitudes and times to form extra global parameter
        return {
            "stepAmpls" : np.concatenate([seq.value
                                           for seq in native_params["stepAmpls"]]),
            "stepTimes" : np.concatenate([seq.value
                                           for seq in native_params["stepTimes"]])}


class ACSource(GeNNStandardCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__
    genn_currentsource_name = 'ACSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injection_code' : '''
            if ($(applyIinj) && $(t) >= $(tStart) && $(t) < $(tStop)) {
                $(injectCurrent, $(Ampl) * sin($(Omega) * $(t) + $(PhaseRad)) + $(Offset));
            }
        ''',
        'var_name_types': [
            ('applyIinj', "unsigned char")],
        'param_name_types' : {
            'tStart': 'scalar', 
            'tStop': 'scalar', 
            'Ampl': 'scalar', 
            'Omega': 'scalar', 
            'PhaseRad': 'scalar', 
            'Offset': 'scalar'},
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('amplitude',  'Ampl'),
        ('frequency',  'Omega',     freqToOmega,    None),
        ('phase',      'PhaseRad',  phaseToRad,     None),
        ('offset',     'Offset')
    ),
    # extra param values
    {
        'applyIinj' : 0
    })

class NoisyCurrentSource(GeNNStandardCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__
    genn_currentsource_name = 'NoisyCurrentSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injection_code' : '''
            if ($(applyIinj) && $(t) >= $(tStart) && $(t) < $(tStop)) {
                $(injectCurrent, $(meanDT) + $(gennrand_normal) * $(sdDT));
            }
        ''',
        
        'var_name_types': [
            ('applyIinj', "unsigned char")],
        'param_name_types' : {
            'tStart': 'scalar', 
            'tStop': 'scalar', 
            'meanDT': 'scalar', 
            'sdDT': 'scalar'}
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('mean',       'meanDT',    partial(mulDT, "mean"),     None),
        ('stdev',      'sdDT',      partial(mulDT, "stdev"),    None),
        ('dt',         '_DT'),
    ),
    # extra param values
    {
        'applyIinj' : 0
    })
