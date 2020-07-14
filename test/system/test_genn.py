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
    epsilon = 0.01
    assert_less_equal(abs_diff, epsilon)

    scale = dist_params['high'] - dist_params['low']
    s, p = stats.kstest((comp_var - dist_params['low']) / scale, 'uniform')
    min_p = 0.05
    assert_greater(p, min_p)

def conn_proc_o2o():
    import numpy as np
    import pynn_genn as sim
    import copy
    timestep = 1.
    sim.setup(timestep)

    n_neurons = 100
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    pre = sim.Population(n_neurons, sim.SpikeSourceArray,
                         {'spike_times': [[1 + i] for i in range(n_neurons)]},
                         label='pre')
    params['tau_syn_E'] = 5.
    post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                          label='post')
    post.record('spikes')

    conn = sim.OneToOneConnector()
    syn = sim.StaticSynapse(weight=5, delay=1)
    proj = sim.Projection(pre, post, conn, synapse_type=syn,
                          use_procedural=bool(1), num_threads_per_spike=1)

    sim.run(2 * n_neurons)
    data = post.get_data()
    spikes = np.asarray(data.segments[0].spiketrains)
    sim.end()

    all_at_appr_time = 0
    sum_spikes = 0
    for i, times in enumerate(spikes):
        sum_spikes += len(times)
        if int(times[0]) == (i + 9):
            all_at_appr_time += 1

    assert_equal(sum_spikes, n_neurons)
    assert_equal(all_at_appr_time, n_neurons)

def conn_proc_a2a():
    import numpy as np
    import pynn_genn as sim
    import copy
    timestep = 1.
    sim.setup(timestep)

    n_neurons = 100
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    pre = sim.Population(n_neurons, sim.SpikeSourceArray,
                         {'spike_times': [[1] for _ in range(n_neurons)]},
                         label='pre')
    params['tau_syn_E'] = 5.
    post = sim.Population(n_neurons, sim.IF_curr_exp, params,
                          label='post')
    post.record('spikes')

    conn = sim.AllToAllConnector()
    syn = sim.StaticSynapse(weight=5. / n_neurons, delay=1)  # rand_dist)
    proj = sim.Projection(pre, post, conn, synapse_type=syn,
                          use_procedural=bool(1))

    sim.run(2 * n_neurons)
    data = post.get_data()
    spikes = np.asarray(data.segments[0].spiketrains)

    sim.end()

    all_at_appr_time = 0
    sum_spikes = 0
    for i, times in enumerate(spikes):
        sum_spikes += len(times)
        if int(times[0]) == 9:
            all_at_appr_time += 1

    assert_equal(sum_spikes, n_neurons)
    assert_equal(all_at_appr_time, n_neurons)

def conn_proc_fix_post():
    import numpy as np
    import pynn_genn as sim
    import copy
    from pynn_genn.random import NativeRNG, NumpyRNG, RandomDistribution

    np_rng = NumpyRNG()
    rng = NativeRNG(np_rng)

    timestep = 1.
    sim.setup(timestep)

    n_pre = 100
    n_post = 50000
    params = copy.copy(sim.IF_curr_exp.default_parameters)
    times = [[1] for _ in range(n_pre)]
    pre = sim.Population(n_pre, sim.SpikeSourceArray,
                         {'spike_times': times},
                         label='pre')
    post = sim.Population(n_post, sim.IF_curr_exp, params,
                          label='post')
    post.record('spikes')

    n = 2
    dist_params = {'low': 4.99, 'high': 5.01}
    dist = 'uniform'
    rand_dist = RandomDistribution(dist, rng=rng, **dist_params)
    conn = sim.FixedNumberPostConnector(n, with_replacement=True, rng=rng)
    syn = sim.StaticSynapse(weight=rand_dist, delay=1)  # rand_dist)
    # needed to use 1 thread per spike to get correct results,
    # this is because the number of connections?
    proj = sim.Projection(pre, post, conn, synapse_type=syn,
                          use_procedural=bool(1), num_threads_per_spike=1)

    sim.run(100)
    data = post.get_data()
    spikes = np.asarray(data.segments[0].spiketrains)
    sim.end()

    all_at_appr_time = 0
    sum_spikes = 0
    for i, times in enumerate(spikes):
        sum_spikes += (1 if len(times) else 0)
        if len(times) == 1 and times[0] == 9:
            all_at_appr_time += 1

    assert_less_equal(np.abs(sum_spikes - (n_pre * n)), 2)
    assert_less_equal(np.abs(all_at_appr_time - (n_pre * n)), 2)


if __name__ == '__main__':
    test_scenarios()
    rng_checks()
    v_rest()
    w_init_o2o() # this also tests connectivity init
    conn_init_fix_prob()
    conn_init_fix_total()
    conn_init_fix_post()
    # todo: these tests are not super good, need to think a better way
    conn_proc_o2o()
    conn_proc_a2a()
    conn_proc_fix_post()

