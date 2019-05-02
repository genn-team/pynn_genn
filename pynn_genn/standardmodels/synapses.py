## encoding: utf-8
"""
Standard cells for the GeNN module.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""
from copy import deepcopy
from string import Template
from pyNN.standardmodels import synapses, StandardModelType, build_translations
from ..simulator import state
import logging
from pygenn.genn_wrapper.WeightUpdateModels import StaticPulse
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
    """Template string class with the delimiter overridden with double $"""
    delimiter = "$$"

class StaticSynapse(synapses.StaticSynapse, GeNNStandardSynapseType):
    __doc__ = synapses.StaticSynapse.__doc__

    wum_defs = {
        "sim_code" : "$(addToInSyn, $(g));\n",
        "vars" : {"g": "scalar"}}

    translations = build_translations(
        ("weight", "g"),
        ("delay", "delaySteps", delayMsToSteps, delayStepsToMs))

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

class TsodyksMarkramSynapse(synapses.TsodyksMarkramSynapse, GeNNStandardSynapseType):
    __doc__ = synapses.TsodyksMarkramSynapse.__doc__

    default_parameters = synapses.TsodyksMarkramSynapse.default_parameters
    default_parameters.update({
        "tau_psc" : 1.0,
    })
    default_initial_values = synapses.TsodyksMarkramSynapse.default_initial_values
    default_initial_values.update({
        "x" : 1.0,
        "y" : 0.0,
        "z" : 0.0
    })

    wum_defs = {
        "sim_code" : """
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
        """,
        "vars" : {
            "U": "scalar",        # asymptotic value of probability of release
            "tauRec": "scalar",   # recovery time from synaptic depression [ms]
            "tauFacil": "scalar", # time constant for facilitation [ms]
            "tauPsc": "scalar",    # decay time constant of postsynaptic current [ms]
            "g": "scalar",
            "u": "scalar",
            "x": "scalar",
            "y": "scalar",
            "z": "scalar"},

        "is_pre_spike_time_required" : True}

    translations = build_translations(
        ("weight",    "g"),
        ("delay",     "delaySteps", delayMsToSteps, delayStepsToMs),
        ("U",         "U"),
        ("tau_rec",   "tauRec"),
        ("tau_facil", "tauFacil"),
        ("tau_psc",   "tauPsc"),
        ("u",         "u"),
        ("x",         "x"),
        ("y",         "y"),
        ("z",         "z"))

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

class STDPMechanism(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    mutable_vars = set(["g"])

    base_translations = build_translations(
        ("weight", "g"),
        ("delay", "delaySteps", delayMsToSteps, delayStepsToMs),
        ("dendritic_delay_fraction", "_dendritic_delay_fraction"))

    base_defs = {
        "vars" : {"g": "scalar"},
        "pre_var_name_types": [],
        "post_var_name_types": [],

        "sim_code" : DDTemplate("""
            $(addToInSyn, $(g));
            scalar dt = $(t) - $(sT_post);
            $${TD_CODE}
        """),
        "learn_post_code" : DDTemplate("""
            scalar dt = $(t) - $(sT_pre);
            $${TD_CODE}
        """),

        "is_pre_spike_time_required" : True,
        "is_post_spike_time_required" : True,
    }

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

    def __init__(self, timing_dependence=None, weight_dependence=None,
            voltage_dependence=None, dendritic_delay_fraction=1.0,
            weight=0.0, delay=None):
        super(STDPMechanism, self).__init__(
            timing_dependence, weight_dependence, voltage_dependence,
            dendritic_delay_fraction, weight, delay)

        # Create a copy of the standard STDP defs
        self.wum_defs = deepcopy(self.base_defs)

        # Adds variables from timing and weight dependence to definitions
        self.wum_defs["vars"].update(self.timing_dependence.vars)
        self.wum_defs["vars"].update(self.weight_dependence.vars)

        # Add pre and postsynaptic variables from timing dependence to definition
        if hasattr(self.timing_dependence, "pre_var_name_types"):
            self.wum_defs["pre_var_name_types"].extend(
                self.timing_dependence.pre_var_name_types)

        if hasattr(self.timing_dependence, "post_var_name_types"):
            self.wum_defs["post_var_name_types"].extend(
                self.timing_dependence.post_var_name_types)

        # Apply substitutions to sim code
        td_sim_code = self.timing_dependence.sim_code.substitute(
            WD_CODE=self.weight_dependence.depression_update_code)
        self.wum_defs["sim_code"] =\
            self.wum_defs["sim_code"].substitute(TD_CODE=td_sim_code)

        # Apply substitutions to post learn code
        td_post_learn_code = self.timing_dependence.learn_post_code.substitute(
            WD_CODE=self.weight_dependence.potentiation_update_code)
        self.wum_defs["learn_post_code"] =\
            self.wum_defs["learn_post_code"].substitute(TD_CODE=td_post_learn_code)

        # Use pre and postsynaptic spike code from timing dependence
        if hasattr(self.timing_dependence, "pre_spike_code"):
            self.wum_defs["pre_spike_code"] = self.timing_dependence.pre_spike_code

        if hasattr(self.timing_dependence, "post_spike_code"):
            self.wum_defs["post_spike_code"] = self.timing_dependence.post_spike_code


class DVDTPlasticity(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    mutable_vars = set(["g"])

    base_translations = build_translations(
        ("weight", "g"),
        ("delay", "delaySteps", delayMsToSteps, delayStepsToMs),
        ("dendritic_delay_fraction", "_dendritic_delay_fraction"),
    )

    base_defs = {
        "vars" : {"g": "scalar",},
        "pre_var_name_types": [],
        "post_var_name_types": [],

        "sim_code" : DDTemplate("""
            $(addToInSyn, $(g));
            $${TD_CODE}
        """),
        "learn_post_code" : DDTemplate("""
            $${TD_CODE}
        """),

        "is_pre_spike_time_required" : True,
        "is_post_spike_time_required" : False,
    }

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

    def __init__(self, timing_dependence=None, weight_dependence=None,
            voltage_dependence=None, dendritic_delay_fraction=1.0,
            weight=0.0, delay=None):

        super(DVDTPlasticity, self).__init__(
            timing_dependence, weight_dependence, voltage_dependence,
            dendritic_delay_fraction, weight, delay)

        # Create a copy of the standard STDP defs
        self.wum_defs = deepcopy(self.base_defs)

        # Adds variables from timing and weight dependence to definitions
        self.wum_defs["vars"].update(self.timing_dependence.vars)
        self.wum_defs["vars"].update(self.weight_dependence.vars)

        # Add pre and postsynaptic variables from timing dependence to definition
        if hasattr(self.timing_dependence, "pre_var_name_types"):
            self.wum_defs["pre_var_name_types"].extend(
                self.timing_dependence.pre_var_name_types)

        if hasattr(self.timing_dependence, "post_var_name_types"):
            self.wum_defs["post_var_name_types"].extend(
                self.timing_dependence.post_var_name_types)

        # Apply substitutions to sim code
        td_sim_code = self.timing_dependence.sim_code.substitute(
            WD_CODE=self.weight_dependence.depression_update_code)
        self.wum_defs["sim_code"] =\
            self.wum_defs["sim_code"].substitute(TD_CODE=td_sim_code)

        # Apply substitutions to post learn code
        td_post_learn_code = self.timing_dependence.learn_post_code.substitute(
            WD_CODE=self.weight_dependence.potentiation_update_code)
        self.wum_defs["learn_post_code"] =\
            self.wum_defs["learn_post_code"].substitute(TD_CODE=td_post_learn_code)

        # Use pre and postsynaptic spike code from timing dependence
        if hasattr(self.timing_dependence, "pre_spike_code"):
            self.wum_defs["pre_spike_code"] = self.timing_dependence.pre_spike_code

        if hasattr(self.timing_dependence, "post_spike_code"):
            self.wum_defs["post_spike_code"] = self.timing_dependence.post_spike_code



class WeightDependence(object):

    vars = {
        "Wmin": "scalar",   # td + 1 - Minimum weight
        "Wmax": "scalar"    # td + 2 - Maximum weight
    }

    depression_update_code = None

    potentiation_update_code = None

    wd_translations = (
        ("w_max",     "Wmax"),
        ("w_min",     "Wmin"),
    )


class AdditiveWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    depression_update_code = "$(g) = min($(Wmax), max($(Wmin), $(g) - (($(Wmax) - $(Wmin)) * update)));\n"

    potentiation_update_code = "$(g) = min($(Wmax), max($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"

    translations = build_translations(*WeightDependence.wd_translations)


class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence, WeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    depression_update_code = "$(g) -= ($(g) - $(Wmin)) * update;\n"

    potentiation_update_code = "$(g) += ($(Wmax) - $(g)) * update;\n"

    translations = build_translations(*WeightDependence.wd_translations)


class AdditivePotentiationMultiplicativeDepression(synapses.AdditivePotentiationMultiplicativeDepression, WeightDependence):
    __doc__ = synapses.AdditivePotentiationMultiplicativeDepression.__doc__

    depression_update_code = "$(g) -= ($(g) - $(Wmin)) * update;\n"

    potentiation_update_code = "$(g) = min($(Wmax), max($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"

    translations = build_translations(*WeightDependence.wd_translations)


class GutigWeightDependence(synapses.GutigWeightDependence, WeightDependence):
    __doc__ = synapses.GutigWeightDependence.__doc__

    vars = deepcopy(WeightDependence.vars)
    vars.update({"muPlus": "scalar",
                 "muMinus": "scalar"})

    depression_update_code = "$(g) -=  pow(($(g) - $(Wmin)), $(muMinus)) * update;\n"

    potentiation_update_code = "$(g) += pow(($(Wmax) - $(g)), $(muPlus)) * update;\n"

    translations = build_translations(
        ("mu_plus",  "muPlus"),
        ("mu_minus", "muMinus"),
        *WeightDependence.wd_translations)

class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    vars = {"tauPlus": "scalar",  # 0 - Potentiation time constant (ms)
            "tauMinus": "scalar", # 1 - Depression time constant (ms)
            "Aplus": "scalar",    # 2 - Rate of potentiation
            "Aminus": "scalar"}   # 3 - Rate of depression

    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # using {.brc} for left{ or right} so that .format() does not freak out
    sim_code = DDTemplate("""
        if (dt > 0)
        {
            const scalar update = $(Aminus) * $(postTrace) * exp(-dt / $(tauMinus));
            $${WD_CODE}
        }
        """)

    learn_post_code = DDTemplate("""
        if (dt > 0)
        {
            const scalar update = $(Aplus) * $(preTrace) * exp(-dt / $(tauPlus));
            $${WD_CODE}
        }
        """)

    pre_spike_code = """\
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
        """

    post_spike_code = """\
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
        """

    translations = build_translations(
        ("tau_plus",   "tauPlus"),
        ("tau_minus",  "tauMinus"),
        ("A_plus",     "Aplus"),
        ("A_minus",    "Aminus"))


class DVDTRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    vars = {"tauPlus": "scalar",  # 0 - Potentiation time constant (ms)
            "tauMinus": "scalar", # 1 - Depression time constant (ms)
            "Aplus": "scalar",    # 2 - Rate of potentiation
            "Aminus": "scalar"}   # 3 - Rate of depression

    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # using {.brc} for left{ or right} so that .format() does not freak out
    # const scalar update = $(Aplus) * $(sT_pre) * $(dvdt);
    sim_code = DDTemplate("""
        scalar update = 0.0;
        if ($(sT_pre) == $(t)){
            update = $(Aplus) * $(dvdt_post);
        }
        $${WD_CODE}
        """)

    learn_post_code = DDTemplate("""
        """)

    pre_spike_code = """\
        """

    post_spike_code = """\
        """

    translations = build_translations(
        ("tau_plus",   "tauPlus"),
        ("tau_minus",  "tauMinus"),
        ("A_plus",     "Aplus"),
        ("A_minus",    "Aminus"))


class Vogels2011Rule(synapses.Vogels2011Rule):
    __doc__ = synapses.Vogels2011Rule.__doc__

    vars = {"Tau": "scalar",      # 0 - Plasticity time constant (ms)
            "Rho": "scalar",      # 1 - Target rate
            "Eta": "scalar"}      # 2 - Learning rate

    sim_code = DDTemplate("""
        const scalar scale = ($(Wmax) - $(Wmin)) * $(Eta);
        const scalar update = scale * (exp(-dt / $(Tau)) - $(Rho));
        $${WD_CODE}
    """)

    learn_post_code = DDTemplate("""
        const scalar scale = ($(Wmax) - $(Wmin)) * $(Eta);
        const scalar update = -scale * exp(-dt / $(Tau));
        $${WD_CODE}
    """)

    translations = build_translations(
        ("tau", "Tau"),
        ("eta", "Eta"),
        ("rho", "Rho"))

