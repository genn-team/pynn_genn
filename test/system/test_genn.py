from nose.plugins.skip import SkipTest
from .scenarios.registry import registry
from nose.tools import assert_equal, assert_not_equal, assert_greater, assert_less_equal
from pyNN.utility import init_logging, assert_arrays_equal
import numpy
import copy
from scipy import stats

try:
    import pynn_genn
    have_genn = True
except ImportError:
    have_genn = False

try:
    import unittest2 as unittest
except ImportError:
    import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_scenarios():
    for scenario in registry:
        if "genn" not in scenario.exclude:
            scenario.description = "{}(genn)".format(scenario.__name__)

            # **HACK** work around bug in nose where names of tests don't get cached
            test_scenarios.compat_func_name = scenario.description

            if have_genn:
                yield scenario, pynn_genn
            else:
                raise SkipTest


def rng_checks():
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG()
    rng = sim.random.NativeRNG(np_rng)

    dist = "wrong distribution"
    assert_equal(rng._supports_dist(dist), False)

    dist = "uniform"
    assert_equal(rng._supports_dist(dist), True)

    params = {'a': 1, 'b': 2, 'c': 3}
    assert_equal(rng._check_params(dist, params), False)

    params = {'low': 0, 'high': 1}
    assert_equal(rng._check_params(dist, params), True)

    params = {'low': 2, 'high': 1}
    assert_equal(rng._check_params(dist, params), False)

    rng1 = sim.random.NativeRNG(np_rng)

    assert_equal(rng1.seed, rng.seed)

    seed = 9
    rng2 = sim.random.NativeRNG(np_rng, seed=seed)
    assert_equal(seed, rng.seed)
    assert_equal(seed, rng1.seed)
    assert_equal(seed, rng2.seed)


def v_rest():
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG()
    rng = sim.random.NativeRNG(np_rng, seed=1)

    timestep = 1.
    sim.setup(timestep)

    n_neurons = 10000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    dist_params = {'low': -70.0, 'high': -60.0}
    dist = 'uniform'
    var = 'v_rest'
    rand_dist = sim.random.RandomDistribution(dist, rng=rng, **dist_params)
    params[var] = rand_dist
    params['cm'] = 2.

    pop = sim.Population(n_neurons, sim.IF_curr_exp, params,
                         label='rand pop')

    sim.run(10)

    gen_var = numpy.asarray(pop.get(var))

    sim.end()

    scale = dist_params['high'] - dist_params['low']
    s, p = stats.kstest((gen_var - dist_params['low']) / scale, 'uniform')
    min_p = 0.05
    assert_greater(p, min_p)


def v_init():
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG()
    rng = sim.random.NativeRNG(np_rng, seed=1)

    timestep = 1.
    sim.setup(timestep)

    n_neurons = 10000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    dist_params = {'mu': -70.0, 'sigma': 1.0}
    dist = 'normal'
    rand_dist = sim.random.RandomDistribution(dist, rng=rng, **dist_params)
    var = 'v'

    post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                          label='rand pop')
    post.initialize(**{var: rand_dist})
    post.record(var)

    sim.run(10)

    comp_var = post.get_data(var)
    comp_var = comp_var.segments[0].analogsignals[0]
    comp_var = numpy.asarray([float(x) for x in comp_var[0, :]])
    sim.end()

    s, p = stats.kstest((comp_var - dist_params['mu']) / dist_params['sigma'],
                        'norm')
    min_p = 0.05
    assert_greater(p, min_p)


def w_init_o2o():
    # this now tests both weight and connectivity on-device generation
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG(seed=1)
    rng = sim.random.NativeRNG(np_rng, seed=1)

    timestep = 1.
    sim.setup(timestep)

    n_neurons = 10000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    pre = sim.Population(n_neurons, sim.IF_curr_exp, params,
                         label='pre')
    post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                          label='post')

    dist_params = {'low': 0.0, 'high': 10.0}
    dist = 'uniform'
    rand_dist = sim.random.RandomDistribution(dist, rng=rng, **dist_params)
    conn = sim.OneToOneConnector()
    syn = sim.StaticSynapse(weight=rand_dist, delay=1)
    proj = sim.Projection(pre, post, conn, synapse_type=syn)

    sim.run(10)

    comp_var = numpy.asarray(proj.getWeights(format='array'))
    connected = numpy.where(~numpy.isnan(comp_var))
    comp_var = comp_var[connected]
    num_active = comp_var.size
    sim.end()

    assert_equal(num_active, n_neurons)
    assert_equal(numpy.all(connected[0] == connected[1]), True)

    scale = dist_params['high'] - dist_params['low']
    s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
    min_p = 0.05
    assert_greater(p, min_p)


def conn_init_fix_prob():
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG(seed=1)
    rng = sim.random.NativeRNG(np_rng, seed=1)

    timestep = 1.
    sim.setup(timestep)

    n_pre = 1000
    n_post = 1000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    pre = sim.Population(n_pre, sim.IF_curr_exp, params,
                         label='pre')
    post = sim.Population(n_post, sim.IF_curr_exp, params,
                          label='post')

    dist_params = {'low': 0.0, 'high': 10.0}
    dist = 'uniform'
    rand_dist = sim.random.RandomDistribution(dist, rng=rng, **dist_params)
    p_conn = 0.3
    conn = sim.FixedProbabilityConnector(p_connect=p_conn,
                                         rng=rng)
    syn = sim.StaticSynapse(weight=rand_dist, delay=1)
    proj = sim.Projection(pre, post, conn, synapse_type=syn)

    sim.run(10)

    comp_var = numpy.asarray(proj.getWeights(format='array'))
    connected = numpy.where(~numpy.isnan(comp_var))
    comp_var = comp_var[connected]
    num_active = comp_var.size
    sim.end()

    ideal_num_conn = n_pre * n_post * p_conn
    one_percent = ideal_num_conn * 0.01
    abs_diff = numpy.abs(num_active - ideal_num_conn)
    assert_less_equal(abs_diff, one_percent)

    scale = dist_params['high'] - dist_params['low']
    s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
    min_p = 0.05
    assert_greater(p, min_p)


def conn_init_fix_total():
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG(seed=1)
    rng = sim.random.NativeRNG(np_rng, seed=1)

    timestep = 1.
    sim.setup(timestep)

    n_pre = 1000
    n_post = 1000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    pre = sim.Population(n_pre, sim.IF_curr_exp, params,
                         label='pre')
    post = sim.Population(n_post, sim.IF_curr_exp, params,
                          label='post')

    dist_params = {'low': 0.0, 'high': 10.0}
    dist = 'uniform'
    rand_dist = sim.random.RandomDistribution(dist, rng=rng, **dist_params)
    n = n_post
    conn = sim.FixedTotalNumberConnector(n, with_replacement=True, rng=rng)
    syn = sim.StaticSynapse(weight=rand_dist, delay=1)
    proj = sim.Projection(pre, post, conn, synapse_type=syn)

    sim.run(10)

    comp_var = numpy.asarray(proj.getWeights(format='array'))
    connected = numpy.where(~numpy.isnan(comp_var))
    comp_var = comp_var[connected]
    num_active = comp_var.size
    sim.end()

    one_percent = n * 0.01
    abs_diff = numpy.abs(num_active - n)
    assert_less_equal(abs_diff, one_percent)

    scale = dist_params['high'] - dist_params['low']
    s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
    min_p = 0.05
    assert_greater(p, min_p)


def conn_init_fix_post():
    if not have_genn:
        raise SkipTest
    sim = pynn_genn

    np_rng = sim.random.NumpyRNG(seed=1)
    rng = sim.random.NativeRNG(np_rng, seed=1)

    timestep = 1.
    sim.setup(timestep)

    n_pre = 5000
    n_post = 5000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    pre = sim.Population(n_pre, sim.IF_curr_exp, params,
                         label='pre')
    post = sim.Population(n_post, sim.IF_curr_exp, params,
                          label='post')

    dist_params = {'low': 0.0, 'high': 10.0}
    dist = 'uniform'
    rand_dist = sim.random.RandomDistribution(dist, rng=rng, **dist_params)
    n = 121
    conn = sim.FixedNumberPostConnector(n, with_replacement=True, rng=rng)
    syn = sim.StaticSynapse(weight=rand_dist, delay=1)
    proj = sim.Projection(pre, post, conn, synapse_type=syn)

    sim.run(10)

    comp_var = numpy.asarray(proj.getWeights(format='array'))
    n_cols = []
    for r in comp_var:
        n_cols.append(len(numpy.where(~numpy.isnan(r))[0]))
    connected = numpy.where(~numpy.isnan(comp_var))
    comp_var = comp_var[connected]
    num_active = comp_var.size
    sim.end()

    abs_diff = numpy.abs(n - numpy.mean(n_cols))
    fpc = n * 0.05
    assert_less_equal(abs_diff, fpc)

    scale = dist_params['high'] - dist_params['low']
    s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
    min_p = 0.05
    assert_greater(p, min_p)


def test_multiple_runs_reuse_model():
    if not have_genn:
        raise SkipTest
    
    sim = pynn_genn

    sim.setup(timestep=1)
    assert sim.common.Population._nPop == 0, 'zero after setup populations'
    assert sim.common.Projection._nProj == 0, 'zero after setup projections'
    assert sim.state.num_current_sources == 0, 'zero after setup current sources'

    p0 = sim.Population(1, sim.IF_curr_exp, {})
    assert sim.common.Population._nPop == 1, 'one population added'

    dc0 = sim.DCSource(amplitude=0.5, start=20.0, stop=80.0)
    assert sim.state.num_current_sources == 0, 'no current source without injecting'

    dc0.inject_into(p0)
    assert sim.state.num_current_sources == 1, 'one current source after injecting'

    p1 = sim.Population(1, sim.IF_curr_exp, {})
    assert sim.common.Population._nPop == 2, 'two populations added'

    j0 = sim.Projection(p0, p1, sim.OneToOneConnector())
    assert sim.common.Projection._nProj == 1, 'one projection added'

    sim.run(0)

    sim.reset()

    assert sim.common.Population._nPop == 2, 'two populations after reset'
    assert sim.common.Projection._nProj == 1, 'one projection after reset'
    assert sim.state.num_current_sources == 1, 'one current source after reset'

    sim.end()

    # new simulation setup means we reset counting for neuron and synapse pops
    sim.setup(timestep=1, reuse_genn_model=True)

    # restart counting
    assert sim.common.Population._nPop == 0, 'zero after setup populations'
    assert sim.common.Projection._nProj == 0, 'zero after setup projections'
    assert sim.state.num_current_sources == 0, 'zero after setup current sources'

    p0 = sim.Population(1, sim.IF_curr_exp, {})
    assert sim.common.Population._nPop == 1, 'one population added'

    dc0 = sim.DCSource(amplitude=0.5, start=20.0, stop=80.0)
    assert sim.state.num_current_sources == 0, 'no current source without injecting'

    dc0.inject_into(p0)
    assert sim.state.num_current_sources == 1, 'one current source after injecting'

    p1 = sim.Population(1, sim.IF_curr_exp, {})
    assert sim.common.Population._nPop == 2, 'two populations added'

    j0 = sim.Projection(p0, p1, sim.OneToOneConnector())
    assert sim.common.Projection._nProj == 1, 'one projection added'

    sim.run(0)


if __name__ == '__main__':
    test_scenarios()
    rng_checks()
    v_rest()
    w_init_o2o() # this also tests connectivity init
    conn_init_fix_prob()
    conn_init_fix_total()
    conn_init_fix_post()
