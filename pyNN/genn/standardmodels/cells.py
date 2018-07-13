# encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from copy import deepcopy
import numpy as np
from pyNN.standardmodels import cells, build_translations, StandardModelType
from ..simulator import state
import logging
import libgenn
import GeNNModel
from ..conversions import convert_to_single, convert_to_array, convert_init_values

logger = logging.getLogger("PyNN")

class GeNNStandardCellType(StandardModelType):

    genn_neuron = None
    genn_postsyn = None

    def translate_dict(self, val_dict):
        return {self.translations[n]['translated_name'] : v.evaluate()
                for n, v in val_dict.items() if n in self.translations.keys()}

    def get_native_params(self, native_parameters, init_values, param_names, prefix=''):
        native_init_values = self.translate_dict(init_values)
        native_params = {}
        for pn in param_names:
            if prefix + pn in native_parameters.keys():
                native_params[pn] = native_parameters[prefix + pn]
            elif prefix + pn in native_init_values.keys():
                native_params[pn] = native_init_values[prefix + pn]
            else:
                raise Exception('Variable "{}" not found'.format(prefix + pn))

        return native_params

    def get_neuron_params(self, native_parameters, init_values):
        param_names = list(self.genn_neuron.getParamNames())

        # parameters are single-valued in GeNN
        return convert_to_array(
                 self.get_native_params(native_parameters,
                                        init_values,
                                        param_names))

    def get_neuron_vars(self, native_parameters, init_values):
        var_names = [vnt[0] for vnt in self.genn_neuron.getVars()]

        return convert_init_values(
                    self.get_native_params(native_parameters,
                                           init_values,
                                           var_names))

    def get_postsynaptic_params(self, native_parameters, init_values, prefix):
        param_names = list(self.genn_postsyn.getParamNames())

        # parameters are single-valued in GeNN
        return convert_to_array(
                 self.get_native_params(native_parameters,
                                        init_values,
                                        param_names,
                                        prefix))

    def get_postsynaptic_vars(self, native_parameters, init_values, prefix):
        var_names = [vnt[0] for vnt in self.genn_postsyn.getVars()]

        return convert_init_values(
                    self.get_native_params(native_parameters,
                                           init_values,
                                           var_names,
                                           prefix))

    def get_extra_global_params(self, native_parameters, init_values):
        var_names = [vnt[0] for vnt in self.genn_neuron.getExtraGlobalParams()]
        
        egps = self.get_native_params(native_parameters, init_values, var_names)
        # in GeNN world, extra global parameters are defined for the whole population
        # and not for individual neurons.
        # The standard model which use extra global params are supposed to override
        # this function in order to convert values properly.
        return egps

ExpTC = GeNNModel.createDPFClass(lambda pars, dt: np.exp(-dt / pars[1]))()
Rmembrane = GeNNModel.createDPFClass(lambda pars, dt: pars[1] / pars[0])()
genn_lif = (
    # definitions
    {
        'simCode' : '''
            if ($(RefracTime) <= 0.0)
            {
              scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
              $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
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
            'C',          # Membrane capacitance
            'TauM',       # Membrane time constant [ms]
            'Vrest',      # Resting membrane potential [mV]
            'Vreset',     # Reset voltage [mV]
            'Vthresh',    # Spiking threshold [mV]
            'Ioffset',    # Offset current
            'TauRefrac'
        ],

        'derivedParams' : [('ExpTC', ExpTC), ('Rmembrane', Rmembrane)],
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
        ('refrac_time','RefracTime')
    )
)

genn_lif_sfa_rr = (
    # definitions
    {
        'simCode' : '''
            if ($(RefracTime) <= 0.0)
            {
              scalar alpha = (($(Isyn) + $(Ioffset) - $(GR) * ($(V) - $(ERr)) - $(GS) * ($(V) - $(ESfa))) * $(Rmembrane)) + $(Vrest);
              $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
              $(GS) -= $(GS)/$(TauSfa);
              $(GR) -= $(GR)/$(TauRr);
            }
            else
            {
              $(RefracTime) -= DT;
            }
        ''',

                #  dv/dt = (v_rest-v)/tau_m + (>>-g_r*(v-E_r) - g_s*(v-E_s)<< + i_syn + i_offset + i_inj)/c_m  : mV
                #  dg_s/dt = -g_s/tau_s    : nS
                #  dg_r/dt = -g_r/tau_r    : nS
        'thresholdConditionCode' : '$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)',

        'resetCode' : '''
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
        ''',

        'paramNames' : [
            'C',          # Membrane capacitance
            'TauM',       # Membrane time constant [ms]
            'Vrest',      # Resting membrane potential [mV]
            'Vreset',     # Reset voltage [mV]
            'Vthresh',    # Spiking threshold [mV]
            'Ioffset',    # Offset current
            'TauRefrac',
            'TauSfa',
            'TauRr',
            'ESfa',
            'ERr'
        ],

        'derivedParams' : [('ExpTC', ExpTC), ('Rmembrane', Rmembrane)],
        'varNameTypes' : [('V', 'scalar'), ('RefracTime', 'scalar'), ('GS', 'scalar'), ('GR', 'scalar')]
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
        ('refrac_time','RefracTime'),
        ('tau_sfa',    'TauSfa'),
        ('e_rev_sfa',  'ESfa'),
        ('tau_rr',     'TauRr'),
        ('e_rev_rr',   'ERr'),
        ('g_s',        'GS'),
        ('g_r',        'GR'),
        ('q_sfa',      '_QSFA'),
        ('q_rr',       '_QRR')
    )
)

genn_adexp = (
    # definitions
    {
        'simCode' : '''
            #define DV(V, W) (1.0 / $(c)) * ((-$(gL) * ((V) - $(eL))) + ($(gL) * $(deltaT) * exp(((V) - $(vThresh)) / $(deltaT))) + i - (W))
            #define DW(V, W) (1.0 / $(tauW)) * (($(a) * (V - $(eL))) - W)
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
            $(W) += ($(b) * 1000.0);
        ''',

        'paramNames' : [
            'c',        # Membrane capacitance [pF]
            'gL',       # Leak conductance [nS]
            'eL',       # Leak reversal potential [mV]
            'deltaT',   # Slope factor [mV]
            'vThresh',  # Threshold voltage [mV]
            'vSpike',   # Artificial spike height [mV]
            'vReset',   # Reset voltage [mV]
            'tauW',     # Adaption time constant
            'a',        # Subthreshold adaption [nS]
            'b',        # Spike-triggered adaptation [nA]
            'iOffset',  # Offset current
        ],

        'varNameTypes' : [('V', 'scalar'), ('W', 'scalar')]
    },
    # translations
    (
        ('cm',         'c'),
        ('tau_refrac', '_TAU_REFRAC'),
        ('v_spike',    'vSpike'),
        ('v_reset',    'vReset'),
        ('v_rest',     'V_REST'),
        ('tau_m',      '_TAU_M'),
        ('i_offset',   'iOffset'),
        ('a',          'a'),
        ('b',          'b'),
        ('delta_T',    'deltaT'),
        ('tau_w',      'tauW'),
        ('v_thresh',   'vThresh'),
        ('v',          'V'),
        ('w',          'W')
    )
)

expDecay = GeNNModel.createDPFClass(lambda pars, dt: np.exp(-dt / pars[0]))
initExp = GeNNModel.createDPFClass(lambda pars, dt: (pars[0] * (1.0 - np.exp(-dt / pars[0]))) * (1.0 / dt))
genn_exp_curr = (
    # definitions
    {
        'decayCode' : '$(inSyn)*=$(expDecay);',

        'applyInputCode' : '$(Isyn) += $(init) * $(inSyn);',

        'paramNames' : ['tau'],

        'derivedParams' : [('expDecay', expDecay), ('init', initExp)]
    },
    # translations
    (
        #('RefracTime', 'RefracTime'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau')
    )
)

initAlpha = GeNNModel.createDPFClass(lambda pars, dt: np.exp(1) / pars[0])
genn_alpha_curr = (
    # definitions
    {
        'decayCode' : '''
            $(x) = (DT * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));
            $(inSyn)*=$(expDecay);
        ''',

        'applyInputCode' : '$(Isyn) += $(x);',

        'paramNames' : ['tau'],

        'varNameTypes' : [('x', 'scalar')],

        'derivedParams' : [('expDecay', expDecay), ('init', initAlpha)]
    },
    # translations
    (
        # TODO: x -- additional var?
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau'),
        ('x',          'x')
    )
)
genn_alpha_cond = (
    # definitions
    {
        'decayCode' : '''
            $(x) = (DT * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));
            $(inSyn)*=$(expDecay);
        ''',

        'applyInputCode' : '$(Isyn) += ($(E) - $(V)) * $(x);',

        'paramNames' : ['tau', 'E'],

        'varNameTypes' : [('x', 'scalar')],

        'derivedParams' : [('expDecay', expDecay), ('init', initAlpha)]
    },
    # translations
    (
        # TODO: x -- additional var?
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau'),
        ('x',          'x')
    )
)

class IF_curr_alpha(cells.IF_curr_alpha, GeNNStandardCellType):
    __doc__ = cells.IF_curr_alpha.__doc__
    
    default_initial_values = cells.IF_curr_alpha.default_initial_values
    default_initial_values['refrac_time'] = 0.0

    translations = build_translations(
        *(genn_lif[1]),
        *(genn_alpha_curr[1])
    )

    genn_neuron = GeNNModel.createCustomNeuronClass('LIF', **(genn_lif[0]))()
    genn_postsyn = GeNNModel.createCustomPostsynapticClass('AlphaCurr', **(genn_alpha_curr[0]))()


class IF_curr_exp(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    default_initial_values = cells.IF_curr_exp.default_initial_values
    default_initial_values['refrac_time'] = 0.0

    translations = build_translations(
        *(genn_lif[1]),
        *(genn_exp_curr[1])
    )

    genn_neuron = GeNNModel.createCustomNeuronClass('LIF', **(genn_lif[0]))()
    genn_postsyn = GeNNModel.createCustomPostsynapticClass('ExpCurr', **(genn_exp_curr[0]))()


class IF_cond_alpha(cells.IF_cond_alpha, GeNNStandardCellType):
    __doc__ = cells.IF_cond_alpha.__doc__
    
    default_initial_values = cells.IF_cond_alpha.default_initial_values
    default_initial_values['refrac_time'] = 0.0

    translations = build_translations(
        *(genn_lif[1]),
        *(genn_alpha_cond[1])
    )
    genn_neuron = GeNNModel.createCustomNeuronClass('LIF', **(genn_lif[0]))()
    genn_postsyn = GeNNModel.createCustomPostsynapticClass('AlphaCond', **(genn_alpha_cond[0]))()



class IF_cond_exp(cells.IF_cond_exp, GeNNStandardCellType):
    __doc__ = cells.IF_cond_exp.__doc__
    
    default_initial_values = cells.IF_cond_exp.default_initial_values
    default_initial_values['refrac_time'] = 0.0

    translations = build_translations(
        *(genn_lif[1]),
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau')
    )
    genn_neuron = GeNNModel.createCustomNeuronClass('LIF', **(genn_lif[0]))()
    genn_postsyn = libgenn.PostsynapticModels.ExpCond()

class IF_facets_hardware1(cells.IF_facets_hardware1):
    __doc__ = cells.IF_facets_hardware1.__doc__


class HH_cond_exp(cells.HH_cond_exp, GeNNStandardCellType):
    __doc__ = cells.HH_cond_exp.__doc__

    translations = build_translations(
        ('gbar_Na',    'gNa'),
        ('gbar_K',     'gK'),
        ('g_leak',     'gl'),
        ('cm',         'C'),
        ('e_rev_Na',   'ENa'),
        ('e_rev_K',    'EK'),
        ('e_rev_leak', 'El'),
        ('v',          'V'),
        ('m',          'm'),
        ('h',          'h'),
        ('n',          'n'),
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'ihh_tau'),
        ('v_offset',   'V_OFFSET'),
        ('i_offset',   'I_OFFSET'),
    )

    default_initial_values = cells.HH_cond_exp.default_initial_values
    default_initial_values.update({
        'm' : 0.5,
        'h' : 0.5,
        'n' : 0.5
    })

    genn_neuron = libgenn.NeuronModels.TraubMiles()
    genn_postsyn = libgenn.PostsynapticModels.ExpCond()


class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr, GeNNStandardCellType):
    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__

    default_initial_values = cells.IF_cond_exp_gsfa_grr.default_initial_values
    default_initial_values['refrac_time'] = 0.0

    translations = build_translations(
        *(genn_lif_sfa_rr[1]),
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau')
    )
    genn_neuron = GeNNModel.createCustomNeuronClass('LIF_sfa_rr', **(genn_lif_sfa_rr[0]))()
    genn_postsyn = libgenn.PostsynapticModels.ExpCond()


class SpikeSourcePoisson(cells.SpikeSourcePoisson, GeNNStandardCellType):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ('start',    'START'),
        ('rate',     'INTERVAL',  "1000.0/rate",  "1000.0/INTERVAL"),
        ('duration', 'DURATION'),
    )


class SpikeSourceArray(cells.SpikeSourceArray, GeNNStandardCellType):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ('e_rev_E',     'exc_E'),
        ('e_rev_I',     'inh_E'),
        ('tau_syn_E',   'exc_tau'),
        ('tau_syn_I',   'inh_tau'),
        ('spike_times', 'spikeTimes'),
        ('start_spike', 'startSpike'),
        ('end_spike',   'endSpike')
    )

    default_parameters = cells.SpikeSourceArray.default_parameters
    default_parameters.update({
        'e_rev_E'   : 0.0,
        'e_rev_I'   : 0.0,
        'tau_syn_E' : 1.0,
        'tau_syn_I' : 1.0
    })

    default_initial_values = cells.SpikeSourceArray.default_initial_values
    default_initial_values.update( {
        'start_spike' : 0,
        'end_spike'   : 0
    })

    genn_neuron = libgenn.NeuronModels.SpikeSourceArray()
    genn_postsyn = libgenn.PostsynapticModels.ExpCond()

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
            converted_vars['startSpike'][i] = cumSize
            cumSize += len(seq.value)
            converted_vars['endSpike'][i] = cumSize

        return converted_vars


class EIF_cond_alpha_isfa_ista(cells.EIF_cond_alpha_isfa_ista, GeNNStandardCellType):
    __doc__ = cells.EIF_cond_alpha_isfa_ista.__doc__

    translations = build_translations(
        *(genn_adexp[1]),
        *(genn_alpha_cond[1])
    )

    genn_neuron = GeNNModel.createCustomNeuronClass('AdExp', **(genn_adexp[0]))()
    genn_postsyn = GeNNModel.createCustomPostsynapticClass('AlphaCond', **(genn_alpha_cond[0]))()


class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista, GeNNStandardCellType):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__

    translations = build_translations(
        *(genn_adexp[1]),
        ('e_rev_E',    'exc_E'),
        ('e_rev_I',    'inh_E'),
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau')
    )

    genn_neuron = GeNNModel.createCustomNeuronClass('AdExp', **(genn_adexp[0]))()
    genn_postsyn = libgenn.PostsynapticModels.ExpCond()


class Izhikevich(cells.Izhikevich, GeNNStandardCellType):
    __doc__ = cells.Izhikevich.__doc__

    translations = build_translations(
        ('a',        'a'),
        ('b',        'b'),
        ('c',        'c'),
        ('d',        'd'),
        ('i_offset', 'I_OFFSET'),
        ('v'         'V'),
        ('u'         'U')
    )
    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling

    genn_neuron = libgenn.NeuronModels.Izhikevich()
    genn_postsyn = libgenn.PostsynapticModels.DeltaCurr()
