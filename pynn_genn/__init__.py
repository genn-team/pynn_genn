"""
Mock implementation of the PyNN API, for testing and documentation purposes.

This simulator implements the PyNN API, but generates random data rather than
really running simulations.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import inspect
import logging
import os
from pyNN import common
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from pyNN.connectors import *
from pyNN.recording import *
from pyNN.standardmodels import StandardCellType
from . import simulator
from .model import sanitize_label
from .standardmodels.cells import *
from .standardmodels.synapses import *
from .standardmodels.electrodes import *
from .model import sanitize_label
from .populations import Population, PopulationView, Assembly
from .projections import Projection
from neo.io import get_io


logger = logging.getLogger("PyNN")


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, StandardCellType)]


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):

    max_delay = extra_params.get("max_delay", DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.mpi_rank = extra_params.get("rank", 0)
    simulator.state.num_processes = extra_params.get("num_processes", 1)
    simulator.state.model.use_cpu = extra_params.get("use_cpu", None)

    # If a model name is specified, use that
    if "model_name" in extra_params:
        simulator.state.model.model_name = extra_params["model_name"]
    # Otherwise
    else:
        # Get the parent frame from our current frame (whatever called setup)
        calframe = inspect.getouterframes(inspect.currentframe(), 1)

        # Extract model name and path
        model_name = os.path.splitext(os.path.basename(calframe[1][1]))[0]
        model_name = sanitize_label(model_name)
        model_path = os.path.dirname(calframe[1][1])

        # Set model name and path (adding ./ if path is relative)
        simulator.state.model.model_name = model_name
        simulator.state.model_path = (model_path + os.sep
                                      if os.path.isabs(model_path)
                                      else "./" + model_path + os.sep)
    return rank()


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        # Make directories if necessary
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)

        # Get NEO IO for filename
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []

run, run_until = common.build_run(simulator)
run_for = run

reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
                    num_processes, rank = common.build_state_queries(simulator)

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)


record = common.build_record(simulator)

record_v = lambda source, filename: record(["v"], source, filename)

record_gsyn = lambda source, filename: record(["gsyn_exc', 'gsyn_inh"], source, filename)
