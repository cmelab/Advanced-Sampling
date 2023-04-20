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

@MyProject.label
def analyzed(job):
    return job.doc.get("analyzed")

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
        restart = job.isfile("system.txt")
        sim = Simulation(n_particles=job.sp.n_particles, n_density=job.sp.n_density, r=job.sp.r, r_cut=job.sp.r_cut,
                         energy_write_freq=job.sp.energy_write_freq, trajectory_write_freq=job.sp.trajectory_write_freq,
                         energy_func=ENERGY_FUNCS[job.sp.energy_func], hard_sphere=job.sp.hard_sphere, restart=restart,
                         sigma=job.sp.sigma, epsilon=job.sp.epsilon, n=job.sp.n, m=job.sp.m, seed=job.sp.seed)
        job.doc["L"] = sim.L

        print("----------------------")
        print("System generated...")
        print("----------------------")
        print("----------------------")
        print("Starting simulation...")
        print("----------------------")
        # Running the simulation
        for i, (n_steps, kT, max_trans) in enumerate(zip(job.sp.n_steps, job.sp.kT, job.sp.max_trans)):
            current_run = i+1
            if not job.doc["phase_{}".format(current_run)]:
                sim.run(n_steps=n_steps, kT=kT, max_trans=max_trans)
                sim.save_trajectory(fname="trajectory_{}.gsd".format(i))
                sim.save_system()
                job.doc["timestep"].append(sim.timestep)
                job.doc["accepted_moves"].append(sim.accepted_moves)
                job.doc["rejected_moves"].append(sim.rejected_moves)
                job.doc["acceptance_ratio"].append(sim.acceptance_ratio)
                job.doc["tps"].append(sim.tps)
                job.doc["energy"].append(sim.energy)
                job.doc["phase_{}".format(current_run)] = True
                sim.reset_system()

        if sum(job.doc["timestep"]) == sum(job.sp.n_steps):
            job.doc["done"] = True
        print("-----------------------------")
        print("Simulation finished completed")
        print("-----------------------------")

@MyProject.operation
@MyProject.pre(sampled)
@MyProject.post(analyzed)
def analysis(job):
    from cmeutils.structure import all_atom_rdf
    import numpy as np
    os.makedirs(os.path.join(job.ws, "analysis/rdf/"))
    gsdfile = job.fn('trajectory_1.gsd')
    rdf = all_atom_rdf(gsdfile, r_max=1.4, start=-30)
    x = rdf.bin_centers
    y = rdf.rdf
    peakx = max(x)
    peaky = max(y)
    logfile = job.fn('log.txt')
    energy = np.genfromtxt(logfile, delimiter=",")
    mean = np.mean(energy[:,0])
    save_path = os.path.join(job.ws, "analysis/rdf/rdf.txt")
    np.savetxt(save_path, np.transpose([x,y]), delimiter=',', header ="bin_centers, rdf")
    save_peak = os.path.join(job.ws, "analysis/rdf/peak.txt")
    np.savetxt(save_peak, np.transpose([peakx, peaky]), delimiter=',', header="max_x, max_y")
    job.doc['average_PE'] = mean
    job.doc["analyzed"] = True
    

if __name__ == "__main__":
    MyProject().main()
