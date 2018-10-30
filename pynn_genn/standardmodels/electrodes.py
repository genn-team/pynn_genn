# encoding: utf-8
"""
Standard electrodes for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
import numpy
from pyNN.standardmodels import electrodes, build_translations#, StandardCurrentSource
from ..simulator import state
import logging
from pygenn.genn_model import create_dpf_class
from ..conversions import convert_to_single, convert_to_array, convert_init_values
from ..model import GeNNStandardCurrentSource, GeNNDefinitions

logger = logging.getLogger("PyNN")

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
        'vars' : {'tStart': 'scalar',
                  'tStop': 'scalar',
                  'amplitude': 'scalar',
                  'applyIinj': 'unsigned char'}
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('amplitude',  'amplitude')
    ),
    )

    genn_extra_parameters = {'applyIinj' : 0}

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
        'vars' : {'tStart': 'scalar',
                  'tStop': 'scalar',
                  'applyIinj': 'unsigned char',
                  'currStep': 'int'},

        'extra_global_params' : [('stepAmpls', 'scalar*'), ('stepTimes', 'scalar*')]
    },
    # translations
    (
        ('amplitudes',  'stepAmpls'),
        ('times',       'stepTimes')
    ),
    )

    genn_extra_parameters = {
        'tStart' : None,
        'tStop'  : None,
        'currStep' : 0,
        'applyIinj' : 0
    }

    def get_currentsource_params(self):
        ret = {}
        ret['tStart'] = self.native_parameters['stepTimes'].base_value[0].value[0]
        ret['tStop'] = self.native_parameters['stepTimes'].base_value[0].value[-1]
        return ret

    def get_extra_global_params(self):
        egps = super(StepCurrentSource, self).get_extra_global_params()
        return {k : numpy.concatenate([convert_to_array(seq) for seq in v])
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
        'param_names' : ['tStart', 'tStop', 'Ampl', 'Freq', 'Phase', 'Offset'],
        'var_name_types' : [('applyIinj', 'unsigned char')],
        'derived_params' : [
            ('Omega', create_dpf_class(lambda pars, dt: pars[3] * 2 * numpy.pi / 1000.0)()),
            ('PhaseRad', create_dpf_class(lambda pars, dt: pars[4] / 180 * numpy.pi)())]
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('amplitude',  'Ampl'),
        ('frequency',  'Freq'),
        ('phase',      'Phase'),
        ('offset',     'Offset')
    ),
    )

    genn_extra_parameters = {'applyIinj' : 0}

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
        'param_names' : ['tStart', 'tStop', 'mean', 'sd', '_DT'],
        'var_name_types' : [('applyIinj', 'unsigned char')],
        'derived_params' : [
            ('meanDT', create_dpf_class(lambda pars, dt: pars[2] * pars[4])()),
            ('sdDT', create_dpf_class(lambda pars, dt: pars[3] * pars[4])())]
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('mean',       'mean'),
        ('stdev',      'sd'),
        ('dt',         '_DT'),
    ),
    )

    genn_extra_parameters = {'applyIinj' : 0}
