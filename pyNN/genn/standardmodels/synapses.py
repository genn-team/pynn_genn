# encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

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


class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
        ('dendritic_delay_fraction', 'dendritic_delay_fraction')
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d


class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )


class GutigWeightDependence(synapses.GutigWeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    translations = build_translations(
        ('w_max',     'wmax'),
        ('w_min',     'wmin'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
        ('mu_plus',   'muLTP'),
        ('mu_minus',  'muLTD'),
    )


class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ('tau_plus',  'tauLTP'),
        ('tau_minus', 'tauLTD'),
        ('A_plus',    'aLTP'),
        ('A_minus',   'aLTD'),
    )
