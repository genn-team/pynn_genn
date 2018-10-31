# encoding: utf-8
"""
Standard electrodes for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
import numpy as np
from functools import partial
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
        'tStart' : None,
        'tStop'  : None,
        'currStep' : 0,
        'applyIinj' : 0
    })


    def get_currentsource_params(self):
        ret = {}
        ret['tStart'] = self.native_parameters['stepTimes'].base_value[0].value[0]
        ret['tStop'] = self.native_parameters['stepTimes'].base_value[0].value[-1]
        return ret

    def get_extra_global_params(self):
        egps = super(StepCurrentSource, self).get_extra_global_params()
        return {k : np.concatenate([convert_to_array(seq) for seq in v])
                for k, v in egps.items()}



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
