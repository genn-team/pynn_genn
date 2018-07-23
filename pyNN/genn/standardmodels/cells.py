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

    def __init__(self, **parameters):
        super(GeNNStandardCellType, self).__init__(**parameters);
        self.translations = build_translations(
            *(genn_neuron_defs[self.genn_neuron_name].translations),
            *(genn_postsyn_defs[self.genn_postsyn_name].translations)
        )
        self._genn_neuron = None
        self._genn_postsyn = None
        self._params_to_vars = set([])

        for param_name in self.native_parameters.keys():
            if not self.native_parameters[param_name].is_homogeneous:
                self.params_to_vars = [param_name]

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
            elif pn in native_parameters.keys():
                native_params[pn] = native_parameters[pn]
            elif pn in native_init_values.keys():
                native_params[pn] = native_init_values[pn]
            elif pn in self.extra_parameters:
                native_params[pn] = self.extra_parameters[pn]
            else:
                raise Exception('Variable "{}" or "{}" not found'.format(pn,prefix + pn))

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

    @property
    def genn_neuron(self):
        if genn_neuron_defs[self.genn_neuron_name].native:
            return getattr(libgenn.NeuronModels, self.genn_neuron_name)()
        genn_defs = genn_neuron_defs[self.genn_neuron_name].get_definitions(self._params_to_vars)
        genn_defs['simCode'] = 'scalar Iinj = 0.0;\n' + genn_defs['simCode']
        return GeNNModel.createCustomNeuronClass(self.genn_neuron_name, **genn_defs)()

    @property
    def genn_postsyn(self):
        if genn_postsyn_defs[self.genn_postsyn_name].native:
            return getattr(libgenn.PostsynapticModels, self.genn_postsyn_name)()
        genn_defs = genn_postsyn_defs[self.genn_postsyn_name].get_definitions()
        return GeNNModel.createCustomPostsynapticClass(self.genn_postsyn_name, **genn_defs)()

    @property
    def params_to_vars(self):
        return self._params_to_vars

    @params_to_vars.setter
    def params_to_vars(self, params_to_vars):
        self._params_to_vars = self._params_to_vars.union(set(params_to_vars))

class GeNNDefinitions(object):

    def __init__(self, definitions, translations, native=False):
        self.native = native
        self._definitions = definitions.keys()
        self.translations = translations
        for key, value in definitions.items():
            setattr(self, key, value)

    @property
    def translations(self):
        return self._translations

    @translations.setter
    def translations(self, translations):
        self._translations = translations

    @property
    def definitions(self):
        return {defname : getattr(self, defname) for defname in self._definitions}

    def get_definitions(self, params_to_vars=None):
        """get definition for GeNN model
        Args:
        params_to_vars -- list with parameters which should be treated as variables in GeNN
        """
        defs = self.definitions
        if params_to_vars is not None:
            par_names = self.paramNames.copy()
            var_names = self.varNameTypes.copy()
            for par in params_to_vars:
                par_names.remove(par)
                var_names.append((par, 'scalar'))
            defs.update({'paramNames' : par_names, 'varNameTypes' : var_names})
        return defs

ExpTC = GeNNModel.createDPFClass(lambda pars, dt: np.exp(-dt / pars[1]))()
Rmembrane = GeNNModel.createDPFClass(lambda pars, dt: pars[1] / pars[0])()
genn_neuron_defs = {}
genn_neuron_defs['IF'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            if ($(RefracTime) <= 0.0)
            {
              scalar alpha = (($(Isyn) + $(Ioffset) + Iinj) * $(Rmembrane)) + $(Vrest);
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
            'C',          # Membrane capacitance [nF?]
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
    )
)

ExpDecayGSfa = GeNNModel.createDPFClass(lambda pars, dt: np.exp(-dt / pars[7]))()
ExpDecayGRr = GeNNModel.createDPFClass(lambda pars, dt: np.exp(-dt / pars[8]))()
genn_neuron_defs['Adapt'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            if ($(RefracTime) <= 0.0)
            {
              scalar alpha = (($(Isyn) + $(Ioffset) + Iinj - $(GRr) * ($(V) -
                $(ERr)) - $(GSfa) * ($(V) - $(ESfa))) * $(Rmembrane)) + $(Vrest);
              $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
              $(GSfa) *= $(ExpDecayGSfa);
              $(GRr) *= $(ExpDecayGRr);
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

        'derivedParams' : [('ExpTC', ExpTC), ('Rmembrane', Rmembrane),
                           ('ExpDecayGRr', ExpDecayGRr), ('ExpDecayGSfa', ExpDecayGSfa)],
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
            const scalar i = $(Isyn) + $(iOffset) + Iinj;
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

isi = GeNNModel.createDPFClass(lambda pars, dt: 1000 / (pars[0] * dt))()
genn_neuron_defs['Poisson'] = GeNNDefinitions(
    # definitions
    {
        'simCode' : '''
            oldSpike = false;
            if($(timeStepToSpike) <= 0.0f) {
                $(timeStepToSpike) += $(isi) * $(gennrand_exponential);
            }
            $(timeStepToSpike) -= 1.0;
        ''',

        'thresholdConditionCode' : '$(t) > $(spikeStart) && $(t) < $(spikeStart) + $(duration) && $(timeStepToSpike) <= 0.0',

        'paramNames' : ['rate'],
        'varNameTypes' : [('timeStepToSpike', 'scalar'), ('spikeStart', 'scalar'),
                          ('duration', 'scalar')],
        'derivedParams' : [('isi', isi)]
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
                $(timeStepToSpike) += $(isi) * $(gennrand_exponential);
            }
            $(timeStepToSpike) -= 1.0;
            $(RefracTime) -= DT;
        ''',

        'thresholdConditionCode' : '$(t) > $(spikeStart) && $(t) < $(spikeStart) + $(duration) && $(RefracTime) <= 0.0f && $(timeStepToSpike) <= 0.0',

        'resetCode' : '$(RefracTime) = $(TauRefrac)',

        'paramNames' : ['rate', 'TauRefrac'],
        'varNameTypes' : [('timeStepToSpike', 'scalar'), ('spikeStart', 'scalar'),
                          ('duration', 'scalar'), ('RefracTime', 'scalar')],
        'derivedParams' : [('isi', isi)]
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
            $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset)+Iinj)*DT; //at two times for numerical stability
            $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset)+Iinj)*DT;
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
                   $(gl)*($(V)-($(El)))-$(Isyn)-$(Ioffset)-Iinj);
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

expDecay = GeNNModel.createDPFClass(lambda pars, dt: np.exp(-dt / pars[0]))()
initExp = GeNNModel.createDPFClass(lambda pars, dt: (pars[0] * (1.0 - np.exp(-dt / pars[0]))) * (1.0 / dt))()
genn_postsyn_defs = {}
genn_postsyn_defs['ExpCurr'] = GeNNDefinitions(
    # definitions
    {
        'decayCode' : '$(inSyn)*=$(expDecay);',

        'applyInputCode' : '$(Isyn) += $(init) * $(inSyn);',

        'paramNames' : ['tau'],

        'derivedParams' : [('expDecay', expDecay), ('init', initExp)]
    },
    # translations
    (
        ('tau_syn_E',  'exc_tau'),
        ('tau_syn_I',  'inh_tau')
    )
)

initAlpha = GeNNModel.createDPFClass(lambda pars, dt: np.exp(1) / pars[0])()
genn_postsyn_defs['AlphaCurr'] = GeNNDefinitions(
    # definitions
    {
        'decayCode' : '''
            $(x) = $(expDecay) * ((DT * $(inSyn) * $(init)) + $(x));
            $(inSyn)*=$(expDecay);
        ''',

        'applyInputCode' : '$(Isyn) += $(x);',

        'paramNames' : ['tau'],

        'varNameTypes' : [('x', 'scalar')],

        'derivedParams' : [('expDecay', expDecay), ('init', initAlpha)]
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
            $(x) = $(expDecay) * ((DT * $(inSyn) * $(init)) + $(x));
            $(inSyn)*=$(expDecay);
        ''',

        'applyInputCode' : '$(Isyn) += ($(E) - $(V)) * $(x);',

        'paramNames' : ['tau', 'E'],

        'varNameTypes' : [('x', 'scalar')],

        'derivedParams' : [('expDecay', expDecay), ('init', initAlpha)]
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
    
    extra_parameters = {
        'RefracTime' : 0.0,
        'x'          : 0.0
    }

class IF_curr_exp(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'ExpCurr'

    extra_parameters = {
        'RefracTime' : 0.0,
    }

class IF_cond_alpha(cells.IF_cond_alpha, GeNNStandardCellType):
    __doc__ = cells.IF_cond_alpha.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'AlphaCond'
    
    extra_parameters = {
        'RefracTime' : 0.0,
        'x'          : 0.0
    }

class IF_cond_exp(cells.IF_cond_exp, GeNNStandardCellType):
    __doc__ = cells.IF_cond_exp.__doc__

    genn_neuron_name = 'IF'
    genn_postsyn_name = 'ExpCond'
    
    extra_parameters = {
        'RefracTime' : 0.0,
    }

class IF_facets_hardware1(cells.IF_facets_hardware1):
    __doc__ = cells.IF_facets_hardware1.__doc__


class HH_cond_exp(cells.HH_cond_exp, GeNNStandardCellType):
    __doc__ = cells.HH_cond_exp.__doc__

    genn_neuron_name = 'HH'
    genn_postsyn_name = 'ExpCond'

    extra_parameters = {
        'm' : 0.0529324,
        'h' : 0.3176767,
        'n' : 0.5961207
    }

class IF_cond_exp_gsfa_grr(cells.IF_cond_exp_gsfa_grr, GeNNStandardCellType):
    __doc__ = cells.IF_cond_exp_gsfa_grr.__doc__

    genn_neuron_name = 'Adapt'
    genn_postsyn_name = 'ExpCond'

    extra_parameters = {
        'RefracTime' : 0.0,
    }

class SpikeSourcePoisson(cells.SpikeSourcePoisson, GeNNStandardCellType):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    genn_neuron_name = 'Poisson'
    genn_postsyn_name = 'DeltaCurr'

    extra_parameters = {
        'timeStepToSpike' : 0.0,
    }

class SpikeSourcePoissonRefractory(cells.SpikeSourcePoissonRefractory, GeNNStandardCellType):
    __doc__ = cells.SpikeSourcePoissonRefractory.__doc__

    genn_neuron_name = 'PoissonRef'
    genn_postsyn_name = 'DeltaCurr'

    extra_parameters = {
        'timeStepToSpike' : 0.0,
        'RefraTime' : 0.0
    }

class SpikeSourceArray(cells.SpikeSourceArray, GeNNStandardCellType):
    __doc__ = cells.SpikeSourceArray.__doc__
    genn_neuron_name = 'SpikeSourceArray'
    genn_postsyn_name = 'DeltaCurr'

    extra_parameters = {
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

    extra_parameters = {
        'x'  : 0.0
    }

class EIF_cond_exp_isfa_ista(cells.EIF_cond_exp_isfa_ista, GeNNStandardCellType):
    __doc__ = cells.EIF_cond_exp_isfa_ista.__doc__
    genn_neuron_name = 'AdExp'
    genn_postsyn_name = 'ExpCond'

    extra_parameters = {
        'x'  : 0.0
    }

class Izhikevich(cells.Izhikevich, GeNNStandardCellType):
    __doc__ = cells.Izhikevich.__doc__
    genn_neuron_name = 'Izhikevich'
    genn_postsyn_name = 'DeltaCurr'

    standard_receptor_type = True
    receptor_scale = 1e-3  # synaptic weight is in mV, so need to undo usual weight scaling

