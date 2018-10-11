# encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from copy import deepcopy
import numpy as np
from pyNN.standardmodels import cells, build_translations
from ..simulator import state
import logging
from ..conversions import convert_to_single, convert_to_array, convert_init_values
from ..model import GeNNStandardCellType, GeNNDefinitions

logger = logging.getLogger("PyNN")

genn_neuron_defs = {}

genn_neuron_defs['IF'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            if ($(RefracTime) <= 0.0)
            {
              scalar alpha = (($(Isyn) + $(Ioffset)) * $(TauM) / $(C)) + $(Vrest);
              $(V) = alpha - (exp(-DT / $(TauM)) * (alpha - $(V)));
            }
            else
            {
                $(RefracTime) -= DT;
            }
        ''',

        'thresholdConditionCode' : '$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)',

        'resetCode' : '''
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
        ''',

        'paramNames' : [
            'C',          # Membrane capacitance [nF?]
            'TauM',       # Membrane time constant [ms]
            'Vrest',      # Resting membrane potential [mV]
            'Vreset',     # Reset voltage [mV]
            'Vthresh',    # Spiking threshold [mV]
            'Ioffset',    # Offset current
            'TauRefrac'
        ],

        'varNameTypes' : [('V', 'scalar'), ('RefracTime', 'scalar')]
    },
    # translations
    (
        ('v_rest',     'Vrest'),
        ('v_reset',    'Vreset'),
        ('cm',         'C'),
        ('tau_m',      'TauM'),
        ('tau_refrac', 'TauRefrac'),
        ('v_thresh',   'Vthresh'),
        ('i_offset',   'Ioffset'),
        ('v',          'V'),
    )
)

genn_neuron_defs['Adapt'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            if ($(RefracTime) <= 0.0)
            {
              scalar alpha = (($(Isyn) + $(Ioffset) - $(GRr) * ($(V) -
                $(ERr)) - $(GSfa) * ($(V) - $(ESfa))) * $(TauM) / $(C)) + $(Vrest);
              $(V) = alpha - (exp(-DT / $(TauM)) * (alpha - $(V)));
              $(GSfa) *= exp(-DT / $(TauSfa));
              $(GRr) *= exp(-DT / $(TauRr));
            }
            else
            {
              $(RefracTime) -= DT;
            }
        ''',

        'thresholdConditionCode' : '$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)',

        'resetCode' : '''
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
            $(GSfa) += $(QSfa);
            $(GRr) += $(QRr);
        ''',

        'paramNames' : [
            'C',          # Membrane capacitance [nF]
            'TauM',       # Membrane time constant [ms]
            'Vrest',      # Resting membrane potential [mV]
            'Vreset',     # Reset voltage [mV]
            'Vthresh',    # Spiking threshold [mV]
            'Ioffset',    # Offset current [nA]
            'TauRefrac',  # Refractoriness [ms]
            'TauSfa',     # Spike frequency adaptation time constant [ms]
            'TauRr',      # Relative refractoriness time constant [ms] 
            'ESfa',       # Spike frequency adaptation reversal potention [mV]
            'ERr',        # Relative refractoriness reversal potention [mV]
            'QSfa',       # Quantal spike frequency adaptation conductance increase [pS]
            'QRr'         # Quantal relative refractoriness conductance increase [pS]
 
        ],

        'varNameTypes' : [('V', 'scalar'), ('RefracTime', 'scalar'),
                          ('GSfa', 'scalar'), ('GRr', 'scalar')]
    },
    # translations
    (
        ('v_rest',     'Vrest'),
        ('v_reset',    'Vreset'),
        ('cm',         'C'),
        ('tau_m',      'TauM'),
        ('tau_refrac', 'TauRefrac'),
        ('v_thresh',   'Vthresh'),
        ('i_offset',   'Ioffset'),
        ('v',          'V'),
        ('tau_sfa',    'TauSfa'),
        ('e_rev_sfa',  'ESfa'),
        ('tau_rr',     'TauRr'),
        ('e_rev_rr',   'ERr'),
        ('g_s',        'GSfa', 0.001),
        ('g_r',        'GRr', 0.001),
        ('q_sfa',      'QSfa', 0.001),
        ('q_rr',       'QRr', 0.001)
    )
)

#  genn_neuron_defs['GIF'] = GeNNNeuronDefinitions(
#      # definitions
#      {
#          'simCode' : '''
#
#          ''',
#
#          'thresholdConditionCode' : '',
#
#          'resetCode' : '',
#
#          'paramNames'

genn_neuron_defs['AdExp'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            #define DV(V, W) (1.0 / $(TauM)) * (-((V) - $(Vrest)) + ($(deltaT) * exp(((V) - $(vThresh)) / $(deltaT)))) + (i - (W)) / $(C)
            #define DW(V, W) (1.0 / $(tauW)) * (($(a) * (V - $(Vrest))) - W)
            const scalar i = $(Isyn) + $(iOffset);
            // If voltage is above artificial spike height
            if($(V) >= $(vSpike)) {
               $(V) = $(vReset);
            }
            // Calculate RK4 terms
            const scalar v1 = DV($(V), $(W));
            const scalar w1 = DW($(V), $(W));
            const scalar v2 = DV($(V) + (DT * 0.5 * v1), $(W) + (DT * 0.5 * w1));
            const scalar w2 = DW($(V) + (DT * 0.5 * v1), $(W) + (DT * 0.5 * w1));
            const scalar v3 = DV($(V) + (DT * 0.5 * v2), $(W) + (DT * 0.5 * w2));
            const scalar w3 = DW($(V) + (DT * 0.5 * v2), $(W) + (DT * 0.5 * w2));
            const scalar v4 = DV($(V) + (DT * v3), $(W) + (DT * w3));
            const scalar w4 = DW($(V) + (DT * v3), $(W) + (DT * w3));
            // Update V
            $(V) += (DT / 6.0) * (v1 + (2.0f * (v2 + v3)) + v4);
            // If we're not above peak, update w
            // **NOTE** it's not safe to do this at peak as wn may well be huge
            if($(V) <= -40.0) {
               $(W) += (DT / 6.0) * (w1 + (2.0 * (w2 + w3)) + w4);
            }
        ''',

        'thresholdConditionCode' : '$(V) > -40',

        'resetCode' : '''
            // **NOTE** we reset v to arbitrary plotting peak rather than to actual reset voltage
            $(V) = $(vSpike);
            $(W) += ($(b));
        ''',

        'paramNames' : [
            'C',        # Membrane capacitance [nF]
            'TauM',     # Membrane time constant [ms]
            'Vrest',    # Resting membrane voltage (Leak reversal potential) [mV]
            'deltaT',   # Slope factor [mV]
            'vThresh',  # Threshold voltage [mV]
            'vSpike',   # Artificial spike height [mV]
            'vReset',   # Reset voltage [mV]
            'tauW',     # Adaption time constant [ms]
            'a',        # Subthreshold adaption [pS]
            'b',        # Spike-triggered adaptation [nA]
            'iOffset',  # Offset current [nA]
        ],

        'varNameTypes' : [('V', 'scalar'),
                ('W', 'scalar')] # adaptation current, [nA]
    },
    # translations
    (
        ('cm',         'C'),
        ('tau_refrac', '_TAU_REFRAC'),
        ('v_spike',    'vSpike'),
        ('v_reset',    'vReset'),
        ('v_rest',     'Vrest'),
        ('tau_m',      'TauM'),
        ('i_offset',   'iOffset'),
        ('a',          'a', 0.001),
        ('b',          'b'),
        ('delta_T',    'deltaT'),
        ('tau_w',      'tauW'),
        ('v_thresh',   'vThresh'),
        ('v',          'V'),
        ('w',          'W'),
    )
)


genn_neuron_defs['SpikeSourceArray'] = GeNNDefinitions({},
    # translations
    (
        ('spike_times', 'spikeTimes'),
    ),
    True # use native
)

genn_neuron_defs['Poisson'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            oldSpike = false;
            if($(timeStepToSpike) <= 0.0f) {
                $(timeStepToSpike) += 1000.0f / $(rate) * DT * $(gennrand_exponential);
            }
            $(timeStepToSpike) -= 1.0;
        ''',

        'thresholdConditionCode' : '$(t) > $(spikeStart) && $(t) < $(spikeStart) + $(duration) && $(timeStepToSpike) <= 0.0',

        'paramNames' : ['rate'],
        'varNameTypes' : [('timeStepToSpike', 'scalar'), ('spikeStart', 'scalar'),
                          ('duration', 'scalar')],
    },
    # translations
    (
        ('rate',     'rate'),
        ('start',    'spikeStart'),
        ('duration', 'duration')
    )
)

genn_neuron_defs['PoissonRef'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            oldSpike = false;
            if($(timeStepToSpike) <= 0.0f) {
                $(timeStepToSpike) += 1000.0f / $(rate) * DT * $(gennrand_exponential);
            }
            $(timeStepToSpike) -= 1.0;
            $(RefracTime) -= DT;
        ''',

        'thresholdConditionCode' : '$(t) > $(spikeStart) && $(t) < $(spikeStart) + $(duration) && $(RefracTime) <= 0.0f && $(timeStepToSpike) <= 0.0',

        'resetCode' : '$(RefracTime) = $(TauRefrac)',

        'paramNames' : ['rate', 'TauRefrac'],
        'varNameTypes' : [('timeStepToSpike', 'scalar'), ('spikeStart', 'scalar'),
                          ('duration', 'scalar'), ('RefracTime', 'scalar')],
    },
    # translations
    (
        ('rate',       'rate'),
        ('start',      'spikeStart'),
        ('duration',   'duration'),
        ('tau_refrac', 'TauRefrac')
    )
)

genn_neuron_defs['Izhikevich'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            if ($(V) >= 30.0){
               $(V)=$(c);
               $(U)+=$(d);
            }
            $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT; //at two times for numerical stability
            $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT;
            $(U)+=$(a)*($(b)*$(V)-$(U))*DT;
            //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise
            //  $(V)=30.0;
            //}
        ''',

        'thresholdConditionCode' : '$(V) >= 29.99',

        'paramNames' : ['a', 'b', 'c', 'd', 'Ioffset'],
        'varNameTypes' : [('V', 'scalar'), ('U', 'scalar')],
    },
    # translations
    (
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'Ioffset', 1000),
        ('v'         'V'),
        ('u'         'U')
    )
)

genn_neuron_defs['HH'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            scalar Imem;
            unsigned int mt;
            scalar mdt= DT/25.0;
            for (mt=0; mt < 25; mt++) {
               Imem= -($(m)*$(m)*$(m)*$(h)*$(gNa)*($(V)-($(ENa)))+
                   $(n)*$(n)*$(n)*$(n)*$(gK)*($(V)-($(EK)))+
                   $(gl)*($(V)-($(El)))-$(Isyn)-$(Ioffset));
               scalar _a;
               if (lV == -52.0) {
                   _a= 1.28;
               }
               else {
                   _a= 0.32*(-52.0-$(V))/(exp((-52.0-$(V))/4.0)-1.0);
               }
               scalar _b;
               if (lV == -25.0) {
                   _b= 1.4;
               }
               else {
                   _b= 0.28*($(V)+25.0)/(exp(($(V)+25.0)/5.0)-1.0);
               }
               $(m)+= (_a*(1.0-$(m))-_b*$(m))*mdt;
               _a= 0.128*exp((-48.0-$(V))/18.0);
               _b= 4.0 / (exp((-25.0-$(V))/5.0)+1.0);
               $(h)+= (_a*(1.0-$(h))-_b*$(h))*mdt;
               if (lV == -50.0) {
                   _a= 0.16;
               }
               else {
                   _a= 0.032*(-50.0-$(V))/(exp((-50.0-$(V))/5.0)-1.0);
               }
               _b= 0.5*exp((-55.0-$(V))/40.0);
               $(n)+= (_a*(1.0-$(n))-_b*$(n))*mdt;
               $(V)+= Imem/$(C)*mdt;
            }
        ''',

        'thresholdConditionCode' : '$(V) >= 0.0',

        'paramNames' : ['gNa', 'ENa', 'gK', 'EK', 'gl', 'El', 'C', 'Ioffset'],
        'varNameTypes' : [('V', 'scalar'), ('m', 'scalar'), ('h', 'scalar'), ('n', 'scalar')]
    },
    # translations
    (
        ('gbar_Na',    'gNa'),
        ('gbar_K',     'gK'),
        ('g_leak',     'gl'),
        ('cm',         'C'),
        ('e_rev_Na',   'ENa'),
        ('e_rev_K',    'EK'),
        ('e_rev_leak', 'El'),
        ('v',          'V'),
        ('v_offset',   'V_OFFSET'),
        ('i_offset',   'Ioffset'),
    )
)

genn_postsyn_defs = {}
genn_postsyn_defs['ExpCurr'] = GeNNDefinitions(
    # definitions
    {
        'decayCode' : '$(inSyn)*= exp(-DT / $(tau));',

        'applyInputCode' : '$(Isyn) += ($(tau) * (1.0 - exp(-DT / $(tau)))) * (1.0 / DT) * $(inSyn);',

        'paramNames' : ['tau'],
    },
    # translations
    (
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau'),
    )
)

genn_postsyn_defs['AlphaCurr'] = GeNNDefinitions(
    # definitions
    {
        'decayCode' : '''
            $(x) = exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)) + $(x));
            $(inSyn)*=exp(-DT/$(tau));
        ''',

        'applyInputCode' : '$(Isyn) += $(x);',

        'paramNames' : ['tau'],

        'varNameTypes' : [('x', 'scalar')],
    },
    # translations
    (
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau'),
    )
)
genn_postsyn_defs['AlphaCond'] = GeNNDefinitions(
    # definitions
    {
        'decayCode' : '''
            $(x) = exp(-DT/$(tau)) * ((DT * $(inSyn) * exp(1.0f) / $(tau)) + $(x));
            $(inSyn)*=exp(-DT/$(tau));
        ''',

        'applyInputCode' : '$(Isyn) += ($(E) - $(V)) * $(x);',

        'paramNames' : ['tau', 'E'],

        'varNameTypes' : [('x', 'scalar')],
    },
    # translations
    (
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau'),
    )
)

genn_postsyn_defs['ExpCond'] = GeNNDefinitions({},
    # translations
    (
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau')
    ),
    True # use native
)

genn_postsyn_defs['DeltaCurr'] = GeNNDefinitions({}, (), True)

class IF_curr_alpha(cells.IF_curr_alpha, GeNNStandardCellType):
    __doc__ = cells.IF_curr_alpha.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'AlphaCurr'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]
    
    genn_extra_parameters = {
        'RefracTime' : 0.0,
        'x'          : 0.0
    }

class IF_curr_exp(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'ExpCurr'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'RefracTime' : 0.0,
    }

class IF_cond_alpha(cells.IF_cond_alpha, GeNNStandardCellType):
    __doc__ = cells.IF_cond_alpha.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'AlphaCond'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]
    
    genn_extra_parameters = {
        'RefracTime' : 0.0,
        'x'          : 0.0
    }

class IF_cond_exp(cells.IF_cond_exp, GeNNStandardCellType):
    __doc__ = cells.IF_cond_exp.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'ExpCond'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]
    
    genn_extra_parameters = {
        'RefracTime' : 0.0,
    }

class IF_facets_hardware1(cells.IF_facets_hardware1):
    __doc__ = cells.IF_facets_hardware1.__doc__


class HH_cond_exp(cells.HH_cond_exp, GeNNStandardCellType):
    __doc__ = cells.HH_cond_exp.__doc__

    genn_neuron_name = 'HH'
    genn_postsyn_name = 'ExpCond'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'm' : 0.0529324,
        'h' : 0.3176767,
        'n' : 0.5961207
    }

class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr, GeNNStandardCellType):
    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__

    genn_neuron_name = 'Adapt'
    genn_postsyn_name = 'ExpCond'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'RefracTime' : 0.0,
    }

class SpikeSourcePoisson(cells.SpikeSourcePoisson, GeNNStandardCellType):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    genn_neuron_name = 'Poisson'
    genn_postsyn_name = 'DeltaCurr'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'timeStepToSpike' : 0.0,
    }

class SpikeSourcePoissonRefractory(cells.SpikeSourcePoissonRefractory, GeNNStandardCellType):
    __doc__ = cells.SpikeSourcePoissonRefractory.__doc__

    genn_neuron_name = 'PoissonRef'
    genn_postsyn_name = 'DeltaCurr'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'timeStepToSpike' : 0.0,
        'RefraTime' : 0.0
    }

class SpikeSourceArray(cells.SpikeSourceArray, GeNNStandardCellType):
    __doc__ = cells.SpikeSourceArray.__doc__
    genn_neuron_name = 'SpikeSourceArray'
    genn_postsyn_name = 'DeltaCurr'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'startSpike' : [],
        'endSpike'   : []
    }
    
    def get_extra_global_params(self, native_parameters, init_values):
        egps = super(SpikeSourceArray, self).get_extra_global_params(
                                                            native_parameters,
                                                            init_values)
        return {k : np.concatenate([convert_to_array(seq) for seq in v])
                for k, v in egps.items()}

    def get_neuron_vars(self, native_parameters, init_values):
        converted_vars = super(SpikeSourceArray, self).get_neuron_vars(
                                                            native_parameters,
                                                            init_values)
        spk_times = native_parameters['spikeTimes']

        cumSize = 0
        for i, seq in enumerate(spk_times):
            converted_vars['startSpike'].append(cumSize)
            cumSize += len(seq.value)
            converted_vars['endSpike'].append(cumSize)

        return converted_vars


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista, GeNNStandardCellType):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__
    genn_neuron_name = 'AdExp'
    genn_postsyn_name = 'AlphaCond'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'x'  : 0.0
    }

class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista, GeNNStandardCellType):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__
    genn_neuron_name = 'AdExp'
    genn_postsyn_name = 'ExpCond'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    genn_extra_parameters = {
        'x'  : 0.0
    }

class Izhikevich(cells.Izhikevich, GeNNStandardCellType):
    __doc__ = cells.Izhikevich.__doc__
    genn_neuron_name = 'Izhikevich'
    genn_postsyn_name = 'DeltaCurr'
    neuron_defs = genn_neuron_defs[genn_neuron_name]
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]

    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling

