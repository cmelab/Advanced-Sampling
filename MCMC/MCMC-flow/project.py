"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""
import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from simulation import Simulation
from utils import inverse_distance_attractive, inverse_distance_repulsive, lj_energy

ENERGY_FUNCS = {"lj": lj_energy,
                "i_dist_att": inverse_distance_attractive,
                "i_dist_rep": inverse_distance_repulsive}


class MyProject(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="short",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortq",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )


# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")


@MyProject.label
def initialized(job):
    return job.isfile("trajectory.gsd")


@directives(executable="python -u")
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the system...")
        print("----------------------")

        # Setting up the system
        restart = job.isfile("restart.gsd")
        sim = Simulation(n_particles=job.sp.n_particles, n_density=job.sp.n_density, r=job.sp.r, r_cut=job.sp.r_cut,
                         energy_write_freq=job.sp.energy_write_freq, trajectory_write_freq=job.sp.trajectory_write_freq,
                         energy_func=ENERGY_FUNCS[job.sp.energy_func], hard_sphere=job.sp.hard_sphere, restart=restart,
                         sigma=job.sp.sigma, epsilon=job.sp.epsilon, n=job.sp.n, m=job.sp.m, seed=job.sp.seed)
        job.doc["L"] = sim.L

        print("----------------------")
        print("System generated...")
        print("----------------------")

        # running mixing step if not done before
        if not job.doc["mixed"]:
            print("----------------------")
            print("Starting mixing simulation...")
            print("----------------------")
            sim.run(n_steps=job.sp.mixing_steps, kT=job.sp.mixing_kT, max_trans=job.sp.mixing_max_trans)
            sim.save_trajectory(fname="trajectory_mixing.gsd")
            job.doc["timestep"].append(sim.timestep)
            job.doc["accepted_moves"].append(sim.accepted_moves)
            job.doc["rejected_moves"].append(sim.rejected_moves)
            job.doc["acceptance_ratio"].append(sim.acceptance_ratio)
            job.doc["tps"].append(sim.tps)
            job.doc["energy"].append(sim.energy)
            job.doc["mixed"] = True
            sim.reset_system()

        print("----------------------")
        print("Starting simulation...")
        print("----------------------")
        # Running the simulation
        for (n_steps, kT, max_trans) in zip(job.sp.n_steps, job.sp.kT, job.sp.max_trans):
            sim.run(n_steps=n_steps, kT=kT, max_trans=max_trans)
            sim.save_trajectory(fname="trajectory_{}.gsd".format(job.doc["current_run"]))
            sim.save_snapshot('restart.gsd')
            job.doc["timestep"].append(sim.timestep)
            job.doc["accepted_moves"].append(sim.accepted_moves)
            job.doc["rejected_moves"].append(sim.rejected_moves)
            job.doc["acceptance_ratio"].append(sim.acceptance_ratio)
            job.doc["tps"].append(sim.tps)
            job.doc["energy"].append(sim.energy)
            job.doc["phase_{}".format(job.doc["current_run"])] = True
            sim.reset_system()
            job.doc["current_run"] += 1


        # TODO: Find a way to check the timestep based on all the run attempts from PT.
        job.doc["done"] = True
        print("-----------------------------")
        print("Simulation finished completed")
        print("-----------------------------")


if __name__ == "__main__":
    MyProject().main()
