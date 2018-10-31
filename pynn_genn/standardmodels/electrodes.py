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
from pygenn.genn_model import create_dpf_class
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
            if ($(applyIinj) && $(t) > $(tStart) && $(t) < $(tStop)) {
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
            if ($(applyIinj) && $(t) >= $(tStart) && $(t) <= $(tStop)) {
                if ($(t) >= $(stepTimes)[$(currStep)])
                    ++$(currStep); // having step variable for each neuron is not optimal, but avoids concurrency issues
                $(injectCurrent, $(stepAmpls)[$(currStep)-1]);
            }
        ''',
        'var_name_types': [
            ('applyIinj', "unsigned char"),
            ('currStep', 'int')],
        'param_name_types' : {
            'tStart': 'scalar',
            'tStop': 'scalar'},

        'extra_global_params' : [('stepAmpls', 'scalar*'), ('stepTimes', 'scalar*')]
    },
    # translations
    (
        ('amplitudes',  'stepAmpls'),
        ('times',       'stepTimes')
    ),
    # extra param values
    {
        'tStart' : stepStart,
        'tStop'  : stepStop,
        'currStep' : 0,
        'applyIinj' : 0
    })

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
            if ($(applyIinj) && $(t) > $(tStart) && $(t) < $(tStop)) {
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
            if ($(applyIinj) && $(t) > $(tStart) && $(t) < $(tStop)) {
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
