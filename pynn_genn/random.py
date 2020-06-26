import numpy as np
from pyNN.random import (NativeRNG,
                         AbstractRNG, NumpyRNG, RandomDistribution)
from pygenn.genn_model import create_custom_init_var_snippet_class
from . import simulator

import logging


### adapted from pynn_spinnaker/random

# ----------------------------------------------------------------------------
# NativeRNG
# ----------------------------------------------------------------------------
# Signals that the random numbers will be supplied by RNG running on GeNN
class NativeRNG(NativeRNG):
    """
    Signals that the simulator's own native RNG should be used.
    Each simulator module should implement a class of the same name which
    inherits from this and which sets the seed appropriately.
    """
    # this is from the NumpyRNG in PyNN
    _distributions = {
        'uniform': {
            'params': ('low', 'high'),
            'check': lambda p: p['high'] > p['low'],
            'code': """
                const scalar scale = $(high) - $(low);
                $(value) = $(low) + ($(gennrand_uniform) * scale);
            """,
        },
        'uniform_int': {
            'params': ('low', 'high'),
            'check': lambda p: p['high'] > p['low'],
            'code': """
                const scalar scale = $(high) - $(low);
                $(value) = rint( $(low) + ($(gennrand_uniform) * scale) );
            """
        },
        'normal': {
            'params': ('mu', 'sigma'),
            'check': lambda p: p['sigma'] > 0,
            'code': """
                $(value) = $(mu) + ($(gennrand_normal) * $(sigma));
            """
        },
        'normal_clipped': {
            'params': ('mu', 'sigma', 'low', 'high'),
            'check': lambda p: p['sigma'] > 0 and p['high'] > p['low'],
            'code': """
                scalar normal;
                do{
                    normal = $(mu) + ($(gennrand_normal) * $(sigma));
                } while (normal > $(high) || normal < $(low));
                $(value) = normal;
            """
        },
        'normal_clipped_int': {
            'params': ('mu', 'sigma', 'low', 'high'),
            'check': lambda p: p['sigma'] > 0 and p['high'] > p['low'],
            'code': """
                scalar normal;
                do{
                    normal = $(mu) + ($(gennrand_normal) * $(sigma));
                } while (normal > $(high) || normal < $(low));
                $(value) = rint(normal);
            """
        },
        'normal_clipped_to_boundary': {
            'params': ('mu', 'sigma', 'low', 'high'),
            'check': lambda p: p['sigma'] > 0 and p['high'] > p['low'],
            'code': """
                scalar normal = $(mu) + ($(gennrand_normal) * $(sigma));
                $(value) = fmax( fmin(normal, $(high)), $(low) );
            """
        },
        'exponential': {
            'params': ('beta'),
            'check': lambda p: p['beta'] > 0,
            'code': """
                $(value) = $(beta) * $(gennrand_exponential);
            """
        },
        'gamma': {
            'params': ('k', 'theta'),
            'check': lambda p: p['theta'] > 0 and p['k'] > 0,
            'code': """
                $(value) = $(k) * $(gennrand_gamma, $(theta));
            """
        },

        # 'binomial':       ('binomial',     {'n': 'n', 'p': 'p'}),
        # 'lognormal':      ('lognormal',    {'mu': 'mean', 'sigma': 'sigma'}),
        # 'poisson':        ('poisson',      {'lambda_': 'lam'}),
        # 'vonmises':       ('vonmises',     {'mu': 'mu', 'kappa': 'kappa'}),
    }

    def __init__(self, host_rng, seed=None):
        # Superclass
        self.seed = seed
        super(NativeRNG, self).__init__(self.seed)

        self._seed_generator = np.random.RandomState(seed=self.seed)

        # Cache RNG to use on the host
        assert host_rng is not None
        self._host_rng = host_rng

    def __new__(cls, host_rng, seed=None):
        # constructor - make sure we have a single NativeRNG object in all
        # the simulation
        if simulator.state.native_rng is None:
            obj = super().__new__(cls)
            simulator.state.native_rng = obj
        elif simulator.state.native_rng.seed != seed:
            logging.warning("NativeRNG seed has changed (from {} to {})".format(
                simulator.state.native_rng.seed, seed
            ))
        return simulator.state.native_rng

    def __deepcopy__(self, memo):
        # fake a deepcopy to keep a single NativeRNG
        return simulator.state.native_rng


    def __str__(self):
        return 'NativeRNG GeNN(seed=%s)' % self.seed

    @property
    def parallel_safe(self):
        return self._host_rng.parallel_safe

    # ------------------------------------------------------------------------
    # AbstractRNG methods
    # ------------------------------------------------------------------------
    def next(self, n=None, distribution=None, parameters=None, mask_local=None):
        # Draw from host RNG
        return self._host_rng.next(n, distribution, parameters, mask_local)

    next.__doc__ = AbstractRNG.next.__doc__

    # ------------------------------------------------------------------------
    # On-Device generation methods
    # ------------------------------------------------------------------------
    def _check_params(self, distribution, parameters):
        _params = NativeRNG._distributions[distribution]['params']
        for p in _params:
            if p not in parameters:
                return False

        return self._distributions[distribution]['check'](parameters)

    def _supports_dist(self, distribution):
        return distribution in self._distributions

    # Return custom variable initalization class
    def init_var_snippet(self, distribution, parameters):
        if not self._supports_dist(distribution):
            raise NotImplementedError(
                    "PyNN GeNN RNG does not support distribution"
                    "'{}'.".format(distribution))

        if not self._check_params(distribution, parameters):
            p = self._distributions[distribution]['params']
            raise ValueError("PyNN GeNN RNG unexpected parameters {} or "
                             "wrong range specified.".format(p))

        d = self._distributions[distribution]
        return create_custom_init_var_snippet_class(
                "pynn_genn_rand_{}".format(distribution),
                param_names=d['params'], var_init_code=d['code'])

    def get_mean(self, distribution, parameters):
        return self._distributions[distribution]['mean'](parameters)