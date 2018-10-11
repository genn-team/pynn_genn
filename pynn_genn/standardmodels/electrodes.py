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
from pygenn.genn_model import createDPFClass
from ..conversions import convert_to_single, convert_to_array, convert_init_values
from ..model import GeNNStandardCurrentSource, GeNNDefinitions

logger = logging.getLogger("PyNN")

class DCSource(GeNNStandardCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__
    genn_currentsource_name = 'DCSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injectionCode' : '''
            if ($(applyIinj) && $(t) > $(tStart) && $(t) < $(tStop)) {
                $(injectCurrent, $(amplitude));
            }
        ''',
        'paramNames' : ['tStart', 'tStop', 'amplitude'],
        'varNameTypes' : [('applyIinj', 'unsigned char')]
    },
    # translations
    (
        ('start',      'tStart'),
        ('stop',       'tStop'),
        ('amplitude',  'amplitude')
    ),
    )

    genn_extra_parameters = {'applyIinj' : False}

class StepCurrentSource(GeNNStandardCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__
    genn_currentsource_name = 'StepCurrentSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injectionCode' : '''
            if ($(applyIinj) && $(t) >= $(tStart) && $(t) <= $(tStop)) {
                if ($(t) >= $(stepTimes)[$(currStep)])
                    ++$(currStep); // having step variable for each neuron is not optimal, but avoids concurrency issues
                $(injectCurrent, $(stepAmpls)[$(currStep)-1]);
            }
        ''',
        'paramNames' : ['tStart', 'tStop'],
        'varNameTypes' : [('applyIinj', 'unsigned char'), ('currStep', 'int')],
        'extraGlobalParams' : [('stepAmpls', 'scalar*'), ('stepTimes', 'scalar*')]
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
        'applyIinj' : False
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
        'injectionCode' : '''
            if ($(applyIinj) && $(t) > $(tStart) && $(t) < $(tStop)) {
                $(injectCurrent, $(Ampl) * sin($(Omega) * $(t) + $(PhaseRad)) + $(Offset));
            }
        ''',
        'paramNames' : ['tStart', 'tStop', 'Ampl', 'Freq', 'Phase', 'Offset'],
        'varNameTypes' : [('applyIinj', 'unsigned char')],
        'derivedParams' : [
            ('Omega', createDPFClass(lambda pars, dt: pars[3] * 2 * numpy.pi / 1000.0)()),
            ('PhaseRad', createDPFClass(lambda pars, dt: pars[4] / 180 * numpy.pi)())]
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

    genn_extra_parameters = {'applyIinj' : False}

class NoisyCurrentSource(GeNNStandardCurrentSource, electrodes.NoisyCurrentSource):
    __doc__ = electrodes.NoisyCurrentSource.__doc__
    genn_currentsource_name = 'NoisyCurrentSource'
    currentsource_defs = GeNNDefinitions(
    # definitions
    {
        'injectionCode' : '''
            if ($(applyIinj) && $(t) > $(tStart) && $(t) < $(tStop)) {
                $(injectCurrent, $(meanDT) + $(gennrand_normal) * $(sdDT));
            }
        ''',
        'paramNames' : ['tStart', 'tStop', 'mean', 'sd', '_DT'],
        'varNameTypes' : [('applyIinj', 'unsigned char')],
        'derivedParams' : [
            ('meanDT', createDPFClass(lambda pars, dt: pars[2] * pars[4])()),
            ('sdDT', createDPFClass(lambda pars, dt: pars[3] * pars[4])())]
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

    genn_extra_parameters = {'applyIinj' : False}
