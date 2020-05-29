import numpy as np
from pyNN.random import (NativeRNG,
                         AbstractRNG, NumpyRNG, RandomDistribution)
from pygenn.genn_model import create_custom_init_var_snippet_class


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
            'mean': lambda p: (p['high'] - p['low']) * 0.5,
            'check': lambda p: p['high'] > p['low'],
            'code': """
                const scalar scale = $(high) - $(low);
                $(value) = $(low) + ($(gennrand_uniform) * scale);
            """,
        },
        'uniform_int': {
            'params': ('low', 'high'),
            'mean': lambda p: (p['high'] - p['low']) * 0.5,
            'check': lambda p: p['high'] > p['low'],
            'code': """
                const scalar scale = $(high) - $(low);
                $(value) = rint( $(low) + ($(gennrand_uniform) * scale) );
            """
        },
        'normal': {
            'params': ('mu', 'sigma'),
            'mean': lambda p: p['mu'],
            'check': lambda p: p['sigma'] > 0,
            'code': """
                $(value) = $(mu) + ($(gennrand_normal) * $(sigma));
            """
        },
        'normal_clipped': {
            'params': ('mu', 'sigma', 'low', 'high'),
            'mean': lambda p: p['mu'],
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
            'mean': lambda p: p['mu'],
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
            'mean': lambda p: p['mu'],
            'params': ('mu', 'sigma', 'low', 'high'),
            'check': lambda p: p['sigma'] > 0 and p['high'] > p['low'],
            'code': """
                scalar normal = $(mu) + ($(gennrand_normal) * $(sigma));
                $(value) = fmax( fmin(normal, $(high)), $(low) );
            """
        },
        'exponential': {
            'params': ('beta'),
            'mean': lambda p: 1./p['beta'],
            'check': lambda p: p['beta'] > 0,
            'code': """
                $(value) = $(beta) * $(gennrand_exponential);
            """
        },
        'gamma': {
            'params': ('k', 'theta'),
            'mean': lambda p: p['k'] * p['theta'],
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
        super(NativeRNG, self).__init__(seed)

        self._seed_generator = np.random.RandomState(seed=seed)

        # Cache RNG to use on the host
        assert host_rng is not None
        self._host_rng = host_rng

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