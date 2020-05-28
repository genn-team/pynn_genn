import numpy as np
import copy
from nose.tools import assert_equal, assert_less_equal
from .registry import register


@register(exclude=["neuron", "nest"])  # for NEURON, this only works when run with MPI and more than one process
def rng_checks(sim):
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

@register(exclude=["neuron", "nest"])  # for NEURON, this only works when run with MPI and more than one process
def v_rest(sim):
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

    gen_var = np.asarray(pop.get(var))

    sim.end()

    v_min = gen_var.min()
    v_max = gen_var.max()
    v_avg = gen_var.mean()

    half_range = dist_params['low'] + \
                 (dist_params['high'] - dist_params['low']) / 2.

    epsilon = 0.1
    assert_less_equal(np.abs(v_min - dist_params['low']), epsilon)
    assert_less_equal(np.abs(v_max - dist_params['high']), epsilon)
    assert_less_equal(np.abs(v_avg - half_range), epsilon)

@register(exclude=["neuron", "nest"])  # for NEURON, this only works when run with MPI and more than one process
def v_init(sim):
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
    comp_var = np.asarray([float(x) for x in comp_var[0, :]])
    sim.end()

    v_std = comp_var.std()
    v_mean = comp_var.mean()

    epsilon = 0.1
    assert_less_equal(np.abs(v_mean - dist_params['mu']), epsilon)
    assert_less_equal(np.abs(v_std - dist_params['sigma']), epsilon)

@register(exclude=["neuron", "nest"])  # for NEURON, this only works when run with MPI and more than one process
def w_init_o2o(sim):
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
    var = 'weight'
    on_device_init = bool(1)
    conn = sim.OneToOneConnector(on_device_init=on_device_init)
    syn = sim.StaticSynapse(weight=rand_dist, delay=1)
    proj = sim.Projection(pre, post, conn, synapse_type=syn)

    sim.run(10)

    comp_var = np.asarray(proj.getWeights(format='array'))
    connected = np.where(~np.isnan(comp_var))
    comp_var = comp_var[connected]
    num_active = comp_var.size
    sim.end()

    assert_equal(num_active, n_neurons)
    assert_equal( np.all(connected[0] == connected[1]), True )

    v_min = comp_var.min()
    v_max = comp_var.max()
    v_avg = comp_var.mean()

    half_range = dist_params['low'] + \
                 (dist_params['high'] - dist_params['low']) / 2.

    epsilon = 0.1
    assert_less_equal(np.abs(v_min - dist_params['low']), epsilon)
    assert_less_equal(np.abs(v_max - dist_params['high']), epsilon)
    assert_less_equal(np.abs(v_avg - half_range), epsilon)

if __name__ == '__main__':
    from pyNN.utility import get_simulator
    sim, args = get_simulator()
    rng_checks(sim)
    v_rest(sim)
    w_init_o2o(sim)

