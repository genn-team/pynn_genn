# encoding: utf-8
"""
Standard cells for the mock module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import cells, synapses, electrodes, build_translations, StandardCurrentSource
from .simulator import state
import logging
import libgenn
import GeNNModel

logger = logging.getLogger("PyNN")


class IF_curr_alpha(cells.IF_curr_alpha):
    __doc__ = cells.IF_curr_alpha.__doc__

    translations = build_translations(  # should add some computed/scaled parameters
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )


class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__
    
    default_initial_values = cells.IF_curr_exp.default_initial_values
    default_initial_values['RefracTime'] = 0.0

    translations = build_translations(
        ('v_rest',     'neuron_Vrest'),
        ('v_reset',    'neuron_Vreset'),
        ('cm',         'neuron_C'),  # C_m is in pF, cm in nF
        ('tau_m',      'neuron_TauM'),
        ('tau_refrac', 'neuron_TauRefrac'),
        ('tau_syn_E',  'postsyn_exc_tau'),
        ('tau_syn_I',  'postsyn_inh_tau'),
        ('v_thresh',   'neuron_Vthresh'),
        ('i_offset',   'neuron_Ioffset'),  # I_e is in pA, i_offset in nA
        ('v',          'neuron_V'),
        ('RefracTime', 'neuron_RefracTime')
    )
    
    # neuron_vars: V, RefracTime
    # postsyn_vars: none


    genn_neuron = libgenn.NeuronModels.LIF()
    genn_postsyn = libgenn.PostsynapticModels.ExpCurr()


class IF_cond_alpha(cells.IF_cond_alpha):
    __doc__ = cells.IF_cond_alpha.__doc__

    translations = build_translations(
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I')
    )


class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__

    translations = build_translations(
        ('tau_m',      'TAU_M'),
        ('cm',         'CM'),
        ('v_rest',     'V_REST'),
        ('v_thresh',   'V_THRESH'),
        ('v_reset',    'V_RESET'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('i_offset',   'I_OFFSET'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('tau_syn_I',  'TAU_SYN_I'),
        ('e_rev_E',    'E_REV_E'),
        ('e_rev_I',    'E_REV_I')
    )


class IF_facets_hardware1(cells.IF_facets_hardware1):
    __doc__ = cells.IF_facets_hardware1.__doc__


class HH_cond_exp(cells.HH_cond_exp):
    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'neuron_gNa'),
        ('gbar_K',     'neuron_gK'),
        ('g_leak',     'neuron_gl'),
        ('cm',         'neuron_C'),
        ('v_offset',   'V_OFFSET'),
        ('e_rev_Na',   'neuron_ENa'),
        ('e_rev_K',    'neuron_EK'),
        ('e_rev_leak', 'neuron_El'),
        ('e_rev_E',    'postsyn_exc_E'),
        ('e_rev_I',    'postsyn_inh_E'),
        ('tau_syn_E',  'postsyn_exc_tau'),
        ('tau_syn_I',  'postsyn_ihh_tau'),
        ('i_offset',   'I_OFFSET'),
        ('v',          'neuron_V'),
        ('m',          'neuron_m'),
        ('h',          'neuron_h'),
        ('n',          'neuron_n'),
    )

    default_initial_values = cells.HH_cond_exp.default_initial_values
    default_initial_values.update({
        'm' : 0.5,
        'h' : 0.5,
        'n' : 0.5
    })


    genn_neuron = libgenn.NeuronModels.TraubMiles()
    genn_postsyn = libgenn.PostsynapticModels.ExpCond()


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr):
    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('start',    'START'),
        ('rate',     'INTERVAL',  "1000.0/rate",  "1000.0/INTERVAL"),
        ('duration', 'DURATION'),
    )

        
class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('spike_times', 'neuron_spkTms'),
        ('e_rev_E',    'postsyn_exc_E'),
        ('e_rev_I',    'postsyn_inh_E'),
        ('tau_syn_E',  'postsyn_exc_tau'),
        ('tau_syn_I',  'postsyn_ihh_tau'),
    )
    
    default_parameters = cells.SpikeSourceArray.default_parameters

    default_parameters.update({
        'e_rev_E'   : 0.0,
        'e_rev_I'   : 0.0,
        'tau_syn_E' : 1.0,
        'tau_syn_I' : 1.0
    })

    genn_neuron = GeNNModel.createCustomNeuronClass(
            'SpikeSourceArray',
            paramNames=['spkTms'],
            #  varNameTypes=[('curSpk', 'scalar'), ('spkTms', 'scalar*')],
            #  simCode=(
            #      'if ( $(t) - $(spk_tm)[(int) curSpk] < $(DT) ) {\n'
            #      '   $(glbSpkCnt) = 1;\n'
            #      '   $(glbSpk)[0] = 0;\n'
            #      '   ++curSpk;\n'
            #      '}\n' ),
            simCode='oldSpike = ( abs( $(t) - $(spkTms) - DT ) < DT );\n',
            thresholdConditionCode='abs( $(t) - $(spkTms) ) < DT' )()

    genn_postsyn = libgenn.PostsynapticModels.ExpCond()


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'CM'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('v_spike',    'V_SPIKE'),
        ('v_reset',    'V_RESET'),
        ('v_rest',     'V_REST'),
        ('tau_m',      'TAU_M'),
        ('i_offset',   'I_OFFSET'),
        ('a',          'A'),
        ('b',          'B'),
        ('delta_T',    'DELTA_T'),
        ('tau_w',      'TAU_W'),
        ('v_thresh',   'V_THRESH'),
        ('e_rev_E',    'E_REV_E'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

    translations = build_translations(
        ('cm',         'CM'),
        ('tau_refrac', 'TAU_REFRAC'),
        ('v_spike',    'V_SPIKE'),
        ('v_reset',    'V_RESET'),
        ('v_rest',     'V_REST'),
        ('tau_m',      'TAU_M'),
        ('i_offset',   'I_OFFSET'),
        ('a',          'A'),
        ('b',          'B'),
        ('delta_T',    'DELTA_T'),
        ('tau_w',      'TAU_W'),
        ('v_thresh',   'V_THRESH'),
        ('e_rev_E',    'E_REV_E'),
        ('tau_syn_E',  'TAU_SYN_E'),
        ('e_rev_I',    'E_REV_I'),
        ('tau_syn_I',  'TAU_SYN_I'),
    )


class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__
    
    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'I_e'),
    )
    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling
    

class MockCurrentSource(object):

    def inject_into(self, cells):
        __doc__ = StandardCurrentSource.inject_into.__doc__
        pass


class DCSource(MockCurrentSource, electrodes.DCSource):
    __doc__ = electrodes.DCSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop')
    )


class StepCurrentSource(MockCurrentSource, electrodes.StepCurrentSource):
    __doc__ = electrodes.StepCurrentSource.__doc__

    translations = build_translations(
        ('amplitudes',  'amplitudes'),
        ('times',       'times')
    )


class ACSource(MockCurrentSource, electrodes.ACSource):
    __doc__ = electrodes.ACSource.__doc__

    translations = build_translations(
        ('amplitude',  'amplitude'),
        ('start',      'start'),
        ('stop',       'stop'),
        ('frequency',  'frequency'),
        ('offset',     'offset'),
        ('phase',      'phase')
    )


class NoisyCurrentSource(MockCurrentSource, electrodes.NoisyCurrentSource):

    translations = build_translations(
        ('mean',  'mean'),
        ('start', 'start'),
        ('stop',  'stop'),
        ('stdev', 'stdev'),
        ('dt',    'dt')
    )


class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ('weight', 'g'),
        ('delay', 'delaySteps'),
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
        ('tau_rec', 'TAU_REC'),
        ('tau_facil', 'TAU_FACIL'),
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
