# encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
from copy import copy, deepcopy
from string import Template
from pyNN.standardmodels import synapses, StandardModelType, build_translations
from ..simulator import state
import logging
from pygenn.genn_model import createCustomWeightUpdateClass
from pygenn.genn_wrapper.WeightUpdateModels import StaticPulse
from ..conversions import convert_to_single, convert_to_array, convert_init_values
from ..model import GeNNStandardSynapseType, GeNNDefinitions

logger = logging.getLogger("PyNN")

# Convert delays from milliseconds into timesteps
# **NOTE** in GeNN delay 0 is one timestep
def delayMsToSteps(delay, **kwargs):
    return max(0, (delay / state.dt) - 1)

# Convert delay from timesteps back to milliseconds
# **NOTE** in GeNN delay 0 is one timestep
def delayStepsToMs(delaySteps, **kwargs):
    return (delaySteps + 1.0) * state.dt

class DDTemplate(Template):
    '''Template string class with the delimiter overridden with double $'''
    delimiter = '$$'

genn_tsodyksMakram = (
    # definitions
    {
        'simCode' : '''
            scalar deltaST = $(t) - $(sT_pre);
            $(z) *= exp( -deltaST / $(tauRec) );
            $(z) += $(y) * ( exp( -deltaST / $(tauPsc) ) - 
               exp( -deltaST / $(tauRec) ) ) / ( ( $(tauPsc) / $(tauRec) ) - 1 );
            $(y) *= exp( -deltaST / $(tauPsc) );
            $(x) = 1 - $(y) - $(z);
            $(u) *= exp( -deltaST / $(tauFacil) );
            $(u) += $(U) * ( 1 - $(u) );
            if ( $(u) > $(U) ) {
               $(u) = $(U);
            }
            $(y) += $(x) * $(u);
            $(addToInSyn, $(g) * $(x) * $(u));
            $(updatelinsyn);
        ''',
        'paramNames' : [
            'U',        # asymptotic value of probability of release
            'tauRec',   # recovery time from synaptic depression [ms]
            'tauFacil', # time constant for facilitation [ms]
            'tauPsc'    # decay time constant of postsynaptic current [ms]
        ],
        'varNameTypes' : [ ('g', 'scalar'), ('u', 'scalar'), ('x', 'scalar'),
            ('y', 'scalar'), ('z', 'scalar') ],
        'isPreSpikeTimeRequired' : True
    },
    # translations
    (
        ('weight',    'g'),
        ('delay',     'delaySteps', delayMsToSteps, delayStepsToMs),
        ('U',         'U'),
        ('tau_rec',   'tauRec'),
        ('tau_facil', 'tauFacil'),
        ('tau_psc',   'tauPsc'),
        ('u',         'u'),
        ('x',         'x'),
        ('y',         'y'),
        ('z',         'z'),
    )
)

class StaticSynapse(synapses.StaticSynapse, GeNNStandardSynapseType):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ('weight', 'g'),
        ('delay', 'delaySteps', delayMsToSteps, delayStepsToMs))

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    genn_weightUpdate = StaticPulse()


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse, GeNNStandardSynapseType):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    default_parameters = synapses.TsodyksMarkramSynapse.default_parameters
    default_parameters.update({
        'tau_psc' : 1.0,
    })
    default_initial_values = synapses.TsodyksMarkramSynapse.default_initial_values
    default_initial_values.update({
        'x' : 1.0,
        'y' : 0.0,
        'z' : 0.0
    })

    translations = build_translations(
        *(genn_tsodyksMakram[1])
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    genn_weightUpdate = createCustomWeightUpdateClass('TsodyksMarkramSynapse',
                                                      **(genn_tsodyksMakram[0]))()

genn_stdp = (
    {
        'paramNames' : [],
        'varNameTypes' : [('g', 'scalar')],

        'simCode' : DDTemplate('''
            $(addToInSyn, $(g));
            scalar dt = $(t) - $(sT_post);
            $${TD_CODE}
        '''),
        'learnPostCode' : DDTemplate('''
            scalar dt = $(t) - $(sT_pre);
            $${TD_CODE}
        '''),

        'isPreSpikeTimeRequired' : True,
        'isPostSpikeTimeRequired' : True,
    },
    (
        ('weight', 'g'),
        ('delay', 'delaySteps', delayMsToSteps, delayStepsToMs),
    )
)


class STDPMechanism(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ('dendritic_delay_fraction', '_dendritic_delay_fraction'),
        *genn_stdp[1])

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    def __init__(self, timing_dependence=None, weight_dependence=None,
            voltage_dependence=None, dendritic_delay_fraction=1.0,
            weight=0.0, delay=None):
        super(STDPMechanism, self).__init__(timing_dependence,
                weight_dependence, voltage_dependence, dendritic_delay_fraction,
                weight, delay)

        settings = deepcopy(genn_stdp[0])
        settings['paramNames'].extend(self.timing_dependence.paramNames)
        settings['paramNames'].extend(self.weight_dependence.paramNames)

        settings['varNameTypes'].extend(self.timing_dependence.varNameTypes)
        settings['varNameTypes'].extend(self.weight_dependence.varNameTypes)

        settings['simCode'] = settings['simCode'].substitute(
                TD_CODE=self.timing_dependence.simCode.substitute(
                    WD_CODE=self.weight_dependence.depressionUpdateCode)
        )
        settings['learnPostCode'] = settings['learnPostCode'].substitute(
                TD_CODE=self.timing_dependence.learnPostCode.substitute(
                    WD_CODE=self.weight_dependence.potentiationUpdateCode)
        )
        self.genn_weightUpdate = createCustomWeightUpdateClass('STDP',
                                                               **settings)()



class WeightDependence(object):

    paramNames = [
        'Wmin',     # td + 1 - Minimum weight
        'Wmax',     # td + 2 - Maximum weight
    ]

    varNameTypes = []

    depressionUpdateCode = None

    potentiationUpdateCode = None

    wd_translations = (
        ('w_max',     'Wmax'),
        ('w_min',     'Wmin'),
    )


class AdditiveWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    depressionUpdateCode = '$(g) = min($(Wmax), max($(Wmin), $(g) - update));\n'

    potentiationUpdateCode = '$(g) = min($(Wmax), max($(Wmin), $(g) + update));\n'

    translations = build_translations(*WeightDependence.wd_translations)


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence, WeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    depressionUpdateCode = '$(g) -= ($(g) - $(Wmin)) * update;\n'

    potentiationUpdateCode = '$(g) += ($(Wmax) - $(g)) * update;\n'

    translations = build_translations(*WeightDependence.wd_translations)


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression, WeightDependence):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    depressionUpdateCode = '$(g) -= ($(g) - $(Wmin)) * update;\n'

    potentiationUpdateCode = '$(g) = min($(Wmax), max($(Wmin), $(g) - update));\n'

    translations = build_translations(*WeightDependence.wd_translations)


class GutigWeightDependence(synapses.GutigWeightDependence, WeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    paramNames = copy(WeightDependence.paramNames)

    paramNames.append('muPlus')
    paramNames.append('muMinus')

    depressionUpdateCode = '$(g) -=  pow(($(g) - $(Wmin)), $(muMinus)) * update;\n'

    potentiationUpdateCode = '$(g) += pow(($(Wmax) - $(g)), $(muPlus)) * update;\n'

    translations = build_translations(
        ('mu_plus',  'muPlus'),
        ('mu_minus', 'muMinus'),
        *WeightDependence.wd_translations)

class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    paramNames = [
        'tauPlus',  # 0 - Potentiation time constant (ms)
        'tauMinus', # 1 - Depression time constant (ms)
        'Aplus',    # 2 - Rate of potentiation
        'Aminus',   # 3 - Rate of depression
    ]

    varNameTypes = []

    # using {.brc} for left{ or right} so that .format() does not freak out
    simCode = DDTemplate('''
        if (dt > 0)
        {
            const scalar scale = ($(Wmax) - $(Wmin)) * $(Aminus);
            const scalar update = scale * exp(-dt / $(tauMinus));
            $${WD_CODE}
        }
    ''')

    learnPostCode = DDTemplate('''
        if (dt > 0)
        {
            const scalar scale = ($(Wmax) - $(Wmin)) * $(Aplus);
            const scalar update = scale * exp(-dt / $(tauPlus));
            $${WD_CODE}
        }
    ''')

    translations = build_translations(
        ('tau_plus',   'tauPlus'),
        ('tau_minus',  'tauMinus'),
        ('A_plus',     'Aplus'),
        ('A_minus',    'Aminus'))


class Vogels2011Rule(synapses.Vogels2011Rule):
    __doc__ = synapses.Vogels2011Rule.__doc__

    paramNames = [
        'Tau',      # 0 - Plasticity time constant (ms)
        'Rho',      # 1 - Target rate
        'Eta',      # 2 - Learning rate
    ]

    varNameTypes = []

    simCode = DDTemplate('''
        const scalar scale = ($(Wmax) - $(Wmin)) * $(Eta);
        const scalar update = scale * exp(-dt / $(Tau)) - $(Rho);
        $${WD_CODE}
    ''')

    learnPostCode = DDTemplate('''
        const scalar scale = ($(Wmax) - $(Wmin)) * $(Eta);
        const scalar update = scale * exp(-dt / $(Tau));
        $${WD_CODE}
    ''')

    translations = build_translations(
        ('tau', 'Tau'),
        ('eta', 'Eta'),
        ('rho', 'Rho'))

