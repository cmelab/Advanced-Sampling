#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.

"""

import signac
import logging
from collections import OrderedDict
from itertools import product
from MCMC.utils import lj_energy


def get_parameters():
    parameters = OrderedDict()

    parameters["radius"] = [0.5]
    parameters["N_particles"] = [500]
    parameters["density"] = [0.68]
    parameters["r_cut"] = [2.5]
    parameters["max_trans"] = [0.5]
    parameters["energy_write_freq"] = [100]
    parameters["trajectory_write_freq"] = [10000]
    parameters["energy_function"] = [lj_energy]
    parameters["epsilon" = []
    parameters["hard_sphere"] = [False]
    parameters["temperatures"] = []
    parameters["n_steps"] = []
    parameters["max_trans"] = []
    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project("mcmc") # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()

    project.write_statepoints()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
