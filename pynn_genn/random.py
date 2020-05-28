import numpy as np
from pyNN.random import (NativeRNG as pynn_rng,
                         AbstractRNG, NumpyRNG, RandomDistribution)
from pygenn.genn_model import create_custom_init_var_snippet_class


### adapted from pynn_spinnaker/random

# ----------------------------------------------------------------------------
# NativeRNG
# ----------------------------------------------------------------------------
# Signals that the random numbers will be supplied by RNG running on GeNN
class NativeRNG(pynn_rng):
    """
    Signals that the simulator's own native RNG should be used.
    Each simulator module should implement a class of the same name which
    inherits from this and which sets the seed appropriately.
    """
    # this is from the NumpyRNG in PyNN
    _distributions = {
        'uniform': {
            'vars': ('low', 'high'),
            'code': """
                const scalar scale = $(high) - $(low);
                $(value) = $(low) + ($(gennrand_uniform) * scale);
            """
        },
        'uniform_int': {
            'vars': ('low', 'high'),
            'code': """
                const scalar scale = $(high) - $(low);
                $(value) = rint( $(low) + ($(gennrand_uniform) * scale) );
            """
        },
        'normal': {
            'vars': ('mu', 'sigma'),
            'code': """
                $(value) = $(mu) + ($(gennrand_normal) * $(sigma));
            """
        },
        'normal_clipped': {
            'vars': ('mu', 'sigma', 'low', 'high'),
            'code': """
                scalar normal;
                do{
                    normal = $(mu) + ($(gennrand_normal) * $(sigma));
                } while (normal > $(high) || normal < $(low));
                $(value) = normal;
            """
        },
        'normal_clipped_int': {
            'vars': ('mu', 'sigma', 'low', 'high'),
            'code': """
                scalar normal;
                do{
                    normal = $(mu) + ($(gennrand_normal) * $(sigma));
                } while (normal > $(high) || normal < $(low));
                $(value) = rint(normal);
            """
        },
        'normal_clipped_to_boundary': {
            'vars': ('mu', 'sigma', 'low', 'high'),
            'code': """
                scalar normal = $(mu) + ($(gennrand_normal) * $(sigma));
                $(value) = max( min(normal, $(high)), $(low) );
            """
        },
        'exponential': {
            'vars': ('beta'),
            'code': """
                $(value) = $(beta) * $(gennrand_exponential);
            """
        },
        'gamma': {
            'vars': ('k', 'theta'),
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
    @staticmethod
    def _check_params(distribution, parameters):
        _params = NativeRNG._distributions[distribution]['vars']
        check = True
        if 'low' in parameters and 'high' in parameters:
            check &= parameters['low'] < parameters['high']

        for p in _params:
            if p not in parameters:
                return False
        return check

    def _supports_dist(self, distribution):
        return distribution in self._distributions

    # Return custom variable initalization class
    def init_var_snippet(self, distribution, parameters):
        if not self._supports_dist(distribution):
            raise NotImplementedError(
                    "PyNN GeNN RNG does not support distribution"
                    f"'{distribution}'.")

        if not self._check_params(distribution, parameters):
            p = self._distributions[distribution]['vars']
            raise ValueError(f"PyNN GeNN RNG unexpected parameters {p} or "
                             "wrong range specified.")

        d = self._distributions[distribution]
        return create_custom_init_var_snippet_class(
                f"pynn_genn_rand_{distribution}",
                param_names=d['vars'], var_init_code=d['code'])

    @staticmethod
    def get_mean(distribution, parameters):
        if 'uniform' in distribution:
            return (parameters['high'] - parameters['low']) * 0.5

        if 'normal' in distribution:
            return parameters['mu']

        if 'exponential' in distribution:
            return ( (1./(1.e-12))
                    if parameters['beta'] == 0. else
                     (1./parameters['beta']) )

        if 'gamma' in distribution:
            return parameters['k'] * parameters['theta']