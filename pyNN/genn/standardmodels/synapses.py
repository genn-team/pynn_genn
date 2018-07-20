# encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
from copy import copy, deepcopy
from pyNN.standardmodels import synapses, StandardModelType, build_translations
from ..simulator import state
import logging
import libgenn
import GeNNModel
from ..conversions import convert_to_single, convert_to_array, convert_init_values

logger = logging.getLogger("PyNN")

class GeNNStandardSynapseType(StandardModelType):

    genn_weightUpdate = None

    def translate_dict(self, val_dict):
        return {self.translations[n]['translated_name'] : convert_to_array(v)
                for n, v in val_dict.items() if n in self.translations.keys()}

    def get_native_params(self, connections, init_values, param_names):
        init_values_ = self.default_initial_values.copy()
        init_values_.update(init_values)
        native_init_values = self.translate_dict(init_values_)
        native_params = {}
        for pn in param_names:
            if any([hasattr(conn, pn) for conn in connections]):
                native_params[pn] = []
                for conn in connections:
                    if hasattr(conn, pn):
                        native_params[pn].append(getattr(conn, pn))
            elif pn in native_init_values.keys():
                native_params[pn] = native_init_values[pn]
            else:
                raise Exception('Variable "{}" not found'.format(pn))

        return native_params

    def get_params(self, connections, init_values):
        param_names = list(self.genn_weightUpdate.getParamNames())

        # parameters are single-valued in GeNN
        return convert_to_array(
                 self.get_native_params(connections,
                                        init_values,
                                        param_names))

    def get_vars(self, connections, init_values):
        var_names = [vnt[0] for vnt in self.genn_weightUpdate.getVars()]

        return convert_init_values(
                    self.get_native_params(connections,
                                           init_values,
                                           var_names))

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
            $(addtoinSyn) = $(g) * $(x) * $(u);
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
        ('delay',     'delaySteps', 1/state.dt),
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
        ('delay', 'delaySteps', 1/state.dt),
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    genn_weightUpdate = libgenn.WeightUpdateModels.StaticPulse()


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

    genn_weightUpdate = GeNNModel.createCustomWeightUpdateClass('TsodyksMarkramSynapse', **(genn_tsodyksMakram[0]))()

genn_stdp = [
    {
        'paramNames' : [],
        'varNameTypes' : [('g', 'scalar')],

        'simCode' : '''
            $(addtoinSyn) = $(g);
            $(updatelinsyn);
            scalar dt = $(t) - $(sT_post);
            {0}
        ''',
        'learnPostCode' : '''
            scalar dt = $(t) - $(sT_pre);
            {0}
        ''',

        'isPreSpikeTimeRequired' : True,
        'isPostSpikeTimeRequired' : True,
    },
    (
        ('weight', 'g'),
        ('delay', 'delaySteps', 1/state.dt),
    )
]

genn_add_wd = [
    # definitions
    {
        'paramNames' : [
            'Wmin',     # td + 1 - Minimum weight
            'Wmax',     # td + 2 - Maximum weight
        ],

        'varNameTypes' : [],

        'depressionUpdateCode' : '''
            scalar newWeight = $(g) - update;
        ''',

        'potentiationUpdateCode' : '''
            scalar newWeight = $(g) + update;
        ''',
        'boundaryCode' : '''
            if(newWeight < $(Wmin))
            {
                $(g) = $(Wmin);
            }
            else if(newWeight > $(Wmax))
            {
                $(g) = $(Wmax);
            }
            else
            {
                $(g) = newWeight;
            }
        '''
    },
    # translations
    (
        ('w_max',     'Wmax'),
        ('w_min',     'Wmin'),
    )
]

genn_mult_wd = deepcopy(genn_add_wd)
genn_mult_wd[0]['depressionCode'] = '''
    scalar newWeight = $(g) - $(g) * update;
'''
genn_mult_wd[0]['potentiationCode'] = '''
    scalar newWeight = $(g) + $(g) * update;
'''

genn_add_p_mult_d_wd = deepcopy(genn_add_wd)
genn_add_p_mult_d_wd[0]['potentiationCode'] = '''
    scalar newWeight = $(g) + $(g) * update;
'''

genn_spikePair_td = (
    # definitions
    {
        'paramNames' : [
            'tauPlus',  # 0 - Potentiation time constant (ms)
            'tauMinus', # 1 - Depression time constant (ms)
            'Aplus',    # 2 - Rate of potentiation
            'Aminus',   # 3 - Rate of depression
        ],

        'varNameTypes' : [],

        'simCode' : '''
            if (dt > 0)
            {
                scalar update = $(Aminus) * exp(-dt / $(tau_minus));
                {0}
            }
        ''',

        'learnPostCode' : '''
            if (dt > 0)
            {
                scalar update = $(Aplus) * exp(-dt / $(tauPlus));
                {0}
            }
        ''',
    },
    # translations
    (
        ('tau_plus',   'tauPlus'),
        ('tau_minus',  'tauMinus'),
        ('A_plus',     'Aplus'),
        ('A_minus',    'Aminus'),
    )
)

genn_vogels2011_td = (
    # definitions
    {
        'paramNames' : [
            'Tau',      # 0 - Plasticity time constant (ms)
            'Rho',      # 1 - Target rate
            'Eta',      # 2 - Learning rate
        ],

        'varNameTypes' : [],

        'simCode' : '''
             scalar update = $(Eta) * exp(-dt / $(Tau)) - $(Rho);
             {0}
        ''',

        'learnPostCode' : '''
            scalar update = $(Eta) * exp(-dt / $(Tau));
            {0}
        ''',

    },
    # translations
    (
        ('tau', 'Tau'),
        ('eta', 'Eta'),
        ('rho', 'Rho')
    )
)

class STDPMechanism(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        *(genn_stdp[1]),
        ('dendritic_delay_fraction', '_dendritic_delay_fraction')
    )

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

        settings['simCode'] = settings['simCode'].format(
                self.timing_dependence.simCode.format(
                    wd=(self.weight_dependence.depressionUpdateCode +
                    self.weight_dependence.boundaryCode),
                    lbrc='{', rbrc='}'
                ),
                lbrc='{', rbrc='}'
        )
        settings['learnPostCode'] = settings['learnPostCode'].format(
                self.timing_dependence.learnPostCode.format(
                    wd=self.weight_dependence.potentiationUpdateCode +
                    self.weight_dependence.boundaryCode,
                    lbrc='{', rbrc='}'
                ),
                lbrc='{', rbrc='}'
        )
        self.genn_weightUpdate = GeNNModel.createCustomWeightUpdateClass('STDP',
                **settings)()



class WeightDependence(object):

    paramNames = [
        'Wmin',     # td + 1 - Minimum weight
        'Wmax',     # td + 2 - Maximum weight
    ]

    varNameTypes = []

    depressionUpdateCode = None

    potentiationUpdateCode = None

    boundaryCode = '''
        if(newWeight < $(Wmin))
        {
            $(g) = $(Wmin);
        }
        else if(newWeight > $(Wmax))
        {
            $(g) = $(Wmax);
        }
        else
        {
            $(g) = newWeight;
        }
    '''
    wd_translations = (
        ('w_max',     'Wmax'),
        ('w_min',     'Wmin'),
    )


class AdditiveWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    depressionUpdateCode = 'scalar newWeight = $(g) - update;\n'

    potentiationUpdateCode = 'scalar newWeight = $(g) + update;\n'

    translations = build_translations(
        *(WeightDependence.wd_translations)
    )


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence, WeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    depressionUpdateCode = 'scalar newWeight = $(g) - $(g) * update;\n'

    potentiationUpdateCode = 'scalar newWeight = $(g) + $(g) * update;\n'

    translations = build_translations(
        *(WeightDependence.wd_translations)
    )


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression, WeightDependence):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    depressionUpdateCode = 'scalar newWeight = $(g) - $(g) * update;\n'

    potentiationUpdateCode = 'scalar newWeight = $(g) + update;\n'

    translations = build_translations(
        *(WeightDependence.wd_translations)
    )


class GutigWeightDependence(synapses.GutigWeightDependence, WeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    paramNames = copy(WeightDependence.paramNames)

    paramNames.append('muPlus')
    paramNames.append('muMinus')

    depressionUpdateCode = 'scalar newWeight = $(g) - pow(($(g) - $(Wmin)), $(muMinus)) * update;\n'

    potentiationUpdateCode = 'scalar newWeight = $(g) + pow(($(Wmax) - $(g)), $(muPlus)) * update;\n'

    translations = build_translations(
        *(WeightDependence.wd_translations),
        ('mu_plus',  'muPlus'),
        ('mu_minus', 'muMinus')
    )


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
    simCode = '''
        if (dt > 0)
        {lbrc}
            scalar update = $(Aminus) * exp(-dt / $(tauMinus));
            {wd}
        {rbrc}
    '''

    learnPostCode = '''
        if (dt > 0)
        {lbrc}
            scalar update = $(Aplus) * exp(-dt / $(tauPlus));
            {wd}
        {rbrc}
    '''

    translations = build_translations(
        ('tau_plus',   'tauPlus'),
        ('tau_minus',  'tauMinus'),
        ('A_plus',     'Aplus'),
        ('A_minus',    'Aminus'),
    )


class Vogels2011Rule(synapses.Vogels2011Rule):
    __doc__ = synapses.Vogels2011Rule.__doc__

    paramNames = [
        'Tau',      # 0 - Plasticity time constant (ms)
        'Rho',      # 1 - Target rate
        'Eta',      # 2 - Learning rate
    ]

    varNameTypes = []

    simCode = '''
         scalar update = $(Eta) * exp(-dt / $(Tau)) - $(Rho);
         {wd}
    '''

    learnPostCode = '''
        scalar update = $(Eta) * exp(-dt / $(Tau));
        {wd}
    '''

    translations = build_translations(
        ('tau', 'Tau'),
        ('eta', 'Eta'),
        ('rho', 'Rho')
    )
