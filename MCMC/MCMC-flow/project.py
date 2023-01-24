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
            default="shortgpu",
            help="Specify the partition to submit to."
        )


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpuq",
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
@directives(ngpu=1)
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

        # Set up the system
        sim = Simulation(n_particles=job.sp.n_particles, n_density=job.sp.n_density, r=job.sp.r, r_cut=job.sp.r_cut,
                         energy_write_freq=job.sp.energy_write_freq, trajectory_write_freq=job.sp.trajectory_write_freq,
                         energy_func=ENERGY_FUNCS[job.sp.energy_func], hard_sphere=job.sp.hard_sphere,
                         sigma=job.sp.sigma, epsilon=job.sp.epsilon, n=job.sp.n, m=job.sp.m)

        job.doc["L"] = sim.L

        print("----------------------")
        print("System generated...")
        print("----------------------")
        print("----------------------")
        print("Starting simulation...")
        print("----------------------")
        # Running the simulation
        for n_steps, kT in zip(job.sp.n_steps, job.sp.kT):
            sim.run(n_steps=n_steps, kT=kT, max_trans=job.sp.max_trans)
            sim.save_trajectory(fname="trajectory.gsd")
            sim.save_system()
            sim.clear_history()

        job.doc["timestep"] = sim.timestep
        job.doc["accepted_moves"] = sim.accepted_moves
        job.doc["rejected_moves"] = sim.rejected_moves
        job.doc["acceptance_ratio"] = sim.acceptance_ratio
        job.doc["tps"] = sim.tps
        job.doc["energy"] = sim.energy

        job.doc["done"] = True

        print("-----------------------------")
        print("Simulation finished completed")
        print("-----------------------------")


if __name__ == "__main__":
    MyProject().main()
