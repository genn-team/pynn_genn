import numpy as np

from copy import deepcopy
from six import iteritems, string_types
from collections import Sized
from lazyarray import larray
from scipy.stats import norm, expon

from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import Sequence
from pyNN.parameters import simplify as simplify_params
from pyNN.random import NativeRNG as pynn_rng, AbstractRNG
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.errors import InvalidParameterValueError

from . import simulator

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
    _translations = {
        'uniform':  ('_uniform', {'low': 'min', 'high': 'max'}),
        'uniform_int': ('_uniform_int', {'low': 'min', 'high': 'max'}),
        'normal': ('_normal', {'mu': 'mean', 'sigma': 'sd'}),
        'normal_clipped': ('_normal_clip', {'mu': 'mean', 'sigma': 'sd', 'low': 'min', 'high': 'max'}),
        'normal_clipped_int': ('_normal_clip_int', {'mu': 'mean', 'sigma': 'sd', 'low': 'min', 'high': 'max'}),
        'normal_clipped_to_boundary':
                    ('_normal_clip_boundary', {'mu': 'mean', 'sigma': 'sd', 'low': 'min', 'high': 'max'}),
        'exponential': ('_exponential', {'lambda': 'lambda'}),
        'gamma': ('_gamma', {'k': 'shape', 'theta': 'scale'}),

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
        _params = NativeRNG._translations[distribution][1]
        for p in _params:
            if p not in parameters:
                return False
        return True

    def _supports_dist(self, distribution):
        return distribution in self._translations

    def _var_init_code(self, distribution, parameters):
        if not self._supports_dist(distribution):
            raise NotImplementedError("The GeNN on-device RNG does not support "
                                      f"the {distribution} distribution")

        if not self._check_params(distribution, parameters):
            _params = NativeRNG._translations[distribution][1]
            ks = list(_params.keys())
            raise ValueError(f"GeNN On-Device RNG: Distribution '{distribution}' "
                             f"requires parameters {ks}")

        trans = NativeRNG._translations[distribution][1]
        code = getattr(self, NativeRNG._translations[distribution][0])()
        return code, trans, parameters

    def _uniform(self):
        code = """
            const scalar scale = $(max) - $(min);
            $(value) = $(min) + ($(gennrand_uniform) * scale);
        """
        return code

    def _uniform_int(self):
        code = """
            const scalar scale = $(max) - $(min);
            $(value) = rint( $(min) + ($(gennrand_uniform) * scale) );
        """
        return code

    def _normal(self):
        code = """
            $(value) = $(mean) + ($(gennrand_normal) * $(sd));
        """
        return code

    def _normal_clip(self):
        code = """
            scalar normal;
            do{
                normal = $(mean) + ($(gennrand_normal) * $(sd));
            } while (normal > $(max) || normal < $(min));
            $(value) = normal;
        """
        return code

    def _normal_clip_int(self):
        code = """
            scalar normal;
            do{
                normal = $(mean) + ($(gennrand_normal) * $(sd));
            } while (normal > $(max) || normal < $(min));
            $(value) = rint(normal);
        """
        return code

    def _normal_clip_boundary(self):
        code = """
            scalar normal = $(mean) + ($(gennrand_normal) * $(sd));
            $(value) = max( min(normal, $(high)), $(low) );
        """
        return code

    def _exponential(self):
        code = """
            $(value) = $(lambda) * $(gennrand_exponential);
        """
        return code

    def _gamma(self):
        code = """
            $(value) = $(scale) * $(gennrand_gamma, $(shape));
        """
        return code