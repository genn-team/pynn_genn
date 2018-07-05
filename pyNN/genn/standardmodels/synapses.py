# encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import synapses, build_translations
from ..simulator import state
import logging
import libgenn
import GeNNModel
from ..conversions import convert_to_single, convert_to_array, convert_init_values

logger = logging.getLogger("PyNN")

genn_tsodyksMakram = (
    # definitions
    {
        'simCode' : (
            'scalar deltaST = $(t) - $(sT_pre);\n'
            '$(z) *= std::exp( -deltaST / $(tauRec) );\n'
            '$(z) += $(y) * (std::exp( -deltaST / $(tauPsc) ) - '
            '   std::exp( -deltaST / $(tauRec) ) ) / ( ( $(tauPsc) / $(tauRec) ) - 1 );\n'
            '$(y) *= std::exp( -deltaST / $(tauPsc) );\n'
            '$(x) = 1 - y - z;\n'
            '$(u) *= std::exp( -deltaST ) / $(tauFacil);\n'
            '$(u) += $(U) * ( 1 - $(u) );\n'
            '$if ( $(u) > $(U) ) {\n'
            '   $(u) = $(U);\n'
            '}\n'
            'y += x * u;\n'
            #
            #  '$(X) += $(Z) / $(tauRec) - $(u)*$(X)*deltaT;\n'
            #  '$(Y) += $(Y) / $(tauI) - $(u)*$(X)*deltaT;\n'
            #  '$(Z) += $(Y) / $(tauI) - $(Z) / $(tauRec);\n'
            #  ''
        ),
        'paramNames' : [
            'U',        # asymptotic value of probability of release
            'tauRec',   # recovery time from synaptic depression [ms]
            'tauFacil', # time constant for facilitation [ms]
            'tauPsc'    # decay time constant of postsynaptic current [ms]
        ],
        'varNameTypes' : [ ('u', 'scalar'), ('x', 'scalar'),
            ('y', 'scalar'), ('z', 'scalar') ],
        'needsPreSpikeTime' : True
    },
    # translations
    (
        (),
    )
)

class StaticSynapse(synapses.StaticSynapse):
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


class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    translations = build_translations(
        ('weight', 'WEIGHT'),
        ('delay', 'DELAY'),
        ('U', 'UU'),
        ('tau_rec', 'tauRec'),
        ('tau_facil', 'tauFacil'),
        ('u0', 'U0'),
        ('x0', 'X'),
        ('y0', 'Y')
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d


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
