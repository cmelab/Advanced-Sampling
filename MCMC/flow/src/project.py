"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os


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


@directives(executable="python -u")
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    from ../MCMC.simulation import Simulation

    with job:
        print("JOB ID NUMBER:")
        print(job.id)
        print("-------------------")

        sim = Simulation(
                n_density=job.sp.density,
                n_particles=job.sp.n_particles,
                r=job.sp.radius,
                kT=job.sp.kT,
                max_trans=job.sp.max_trans,
                energy_write_freq=job.sp.energy_write_freq,
                trajectory_write_freq=job.sp.trajectory_write_freq,
                energy_func=job.sp.energy_func,
                hard_sphere=job.sp.hard_sphere
        )


if __name__ == "__main__":
    MyProject().main()
