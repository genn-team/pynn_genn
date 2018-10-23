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
from pygenn.genn_model import create_custom_weight_update_class
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
        'sim_code' : '''
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
        'param_names' : [
            'U',        # asymptotic value of probability of release
            'tauRec',   # recovery time from synaptic depression [ms]
            'tauFacil', # time constant for facilitation [ms]
            'tauPsc'    # decay time constant of postsynaptic current [ms]
        ],
        'var_name_types' : [ ('g', 'scalar'), ('u', 'scalar'), ('x', 'scalar'),
            ('y', 'scalar'), ('z', 'scalar') ],
        'is_pre_spike_time_required' : True
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

    genn_weight_update = StaticPulse()


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

    genn_weight_update = create_custom_weight_update_class('TsodyksMarkramSynapse',
                                                      **(genn_tsodyksMakram[0]))()

genn_stdp = (
    {
        'param_names' : [],
        'var_name_types' : [('g', 'scalar')],

        'sim_code' : DDTemplate('''
            $(addToInSyn, $(g));
            scalar dt = $(t) - $(sT_post);
            $${TD_CODE}
        '''),
        'learn_post_code' : DDTemplate('''
            scalar dt = $(t) - $(sT_pre);
            $${TD_CODE}
        '''),

        'is_pre_spike_time_required' : True,
        'is_post_spike_time_required' : True,
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
        settings['param_names'].extend(self.timing_dependence.param_names)
        settings['param_names'].extend(self.weight_dependence.param_names)

        settings['var_name_types'].extend(self.timing_dependence.var_name_types)
        settings['var_name_types'].extend(self.weight_dependence.var_name_types)

        settings['pre_var_name_types'] = self.timing_dependence.pre_var_name_types
        settings['post_var_name_types'] = self.timing_dependence.post_var_name_types

        settings['sim_code'] = settings['sim_code'].substitute(
                TD_CODE=self.timing_dependence.sim_code.substitute(
                    WD_CODE=self.weight_dependence.depression_update_code))
        settings['learn_post_code'] = settings['learn_post_code'].substitute(
                TD_CODE=self.timing_dependence.learn_post_code.substitute(
                    WD_CODE=self.weight_dependence.potentiation_update_code))

        settings['post_spike_code'] = self.timing_dependence.post_spike_code
        settings['pre_spike_code'] = self.timing_dependence.pre_spike_code

        self.genn_weight_update = create_custom_weight_update_class('STDP',
                                                                    **settings)()



class WeightDependence(object):

    param_names = [
        'Wmin',     # td + 1 - Minimum weight
        'Wmax',     # td + 2 - Maximum weight
    ]

    var_name_types = []

    depression_update_code = None

    potentiation_update_code = None

    wd_translations = (
        ('w_max',     'Wmax'),
        ('w_min',     'Wmin'),
    )


class AdditiveWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    depression_update_code = '$(g) = min($(Wmax), max($(Wmin), $(g) - (($(Wmax) - $(Wmin)) * update)));\n'

    potentiation_update_code = '$(g) = min($(Wmax), max($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n'

    translations = build_translations(*WeightDependence.wd_translations)


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence, WeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    depression_update_code = '$(g) -= ($(g) - $(Wmin)) * update;\n'

    potentiation_update_code = '$(g) += ($(Wmax) - $(g)) * update;\n'

    translations = build_translations(*WeightDependence.wd_translations)


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression, WeightDependence):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    depression_update_code = '$(g) -= ($(g) - $(Wmin)) * update;\n'

    potentiation_update_code = '$(g) = min($(Wmax), max($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n'

    translations = build_translations(*WeightDependence.wd_translations)


class GutigWeightDependence(synapses.GutigWeightDependence, WeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    param_names = copy(WeightDependence.param_names)

    param_names.append('muPlus')
    param_names.append('muMinus')

    depression_update_code = '$(g) -=  pow(($(g) - $(Wmin)), $(muMinus)) * update;\n'

    potentiation_update_code = '$(g) += pow(($(Wmax) - $(g)), $(muPlus)) * update;\n'

    translations = build_translations(
        ('mu_plus',  'muPlus'),
        ('mu_minus', 'muMinus'),
        *WeightDependence.wd_translations)

class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    param_names = [
        'tauPlus',  # 0 - Potentiation time constant (ms)
        'tauMinus', # 1 - Depression time constant (ms)
        'Aplus',    # 2 - Rate of potentiation
        'Aminus',   # 3 - Rate of depression
    ]

    var_name_types = []
    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # using {.brc} for left{ or right} so that .format() does not freak out
    sim_code = DDTemplate('''
        if (dt > 0)
        {
            const scalar update = $(Aminus) * $(postTrace) * exp(-dt / $(tauMinus));
            $${WD_CODE}
        }
        ''')

    learn_post_code = DDTemplate('''
        if (dt > 0)
        {
            const scalar update = $(Aplus) * $(preTrace) * exp(-dt / $(tauPlus));
            $${WD_CODE}
        }
        ''')

    pre_spike_code = '''\
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
        '''

    post_spike_code = '''\
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
        '''

    translations = build_translations(
        ('tau_plus',   'tauPlus'),
        ('tau_minus',  'tauMinus'),
        ('A_plus',     'Aplus'),
        ('A_minus',    'Aminus'))


class Vogels2011Rule(synapses.Vogels2011Rule):
    __doc__ = synapses.Vogels2011Rule.__doc__

    param_names = [
        'Tau',      # 0 - Plasticity time constant (ms)
        'Rho',      # 1 - Target rate
        'Eta',      # 2 - Learning rate
    ]

    var_name_types = []

    sim_code = DDTemplate('''
        const scalar scale = ($(Wmax) - $(Wmin)) * $(Eta);
        const scalar update = scale * (exp(-dt / $(Tau)) - $(Rho));
        $${WD_CODE}
    ''')

    learn_post_code = DDTemplate('''
        const scalar scale = ($(Wmax) - $(Wmin)) * $(Eta);
        const scalar update = -scale * exp(-dt / $(Tau));
        $${WD_CODE}
    ''')

    translations = build_translations(
        ('tau', 'Tau'),
        ('eta', 'Eta'),
        ('rho', 'Rho'))

