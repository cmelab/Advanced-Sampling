import numpy as np
import itertools
import math

class Simulation:
    def __init__(self, n_density=0.5, r=0.1, L=5, kT=1.0, r_cut=1, max_trans=0.2, write_freq=5):
        self.n_density = n_density
        self.r = r
        self.L = L
        self.kT = kT
        self.r_cut = r_cut
        self.max_trans = max_trans
        self.system = self._init_system()
        self.timestep = 0
        self.system_history = []
        self.energies = []
        self.accepted_moves = 0
        self.tps = None
        self.write_freq = write_freq


    @property
    def n_particles(self):
        """
        Calculate number of disks from the number density, disk radius and box size.
        :return: number of particles.
        """
        n_particles = math.floor((math.pow(self.L, 2) * self.n_density) / (math.pi * math.pow(self.r, 2)))
        if n_particles == 0:
            raise ValueError("cannot fit any disk with this density! "
                             "Either decrease density/disk radius or increase box size.")

    @property
    def energy(self):
        return self.calculate_energy(self.system)

    def _init_system(self):
        """
        Initialize an array of 2D positions, randomly putting disk in the box.
        x and y coordinates are between -L/2 and L/2.
        :return: A 2D numpy array of shape (self.n_particles, 2).
        """
        return NotImplementedError

    def check_overlap(self, system, index):
        """
        Check if particle with specified index overlpas with the other particles in the system.
        :param system: 2D array of particle positions.
        :param index: index of the particle.
        :return: True if ther is any overlap, else False.
        """
        coord1 = system[index]
        for i, coord2 in enumerate(system):
            if i == index:
                continue
            d = math.sqrt(math.pow((coord1[0] - coord2[0]), 2) + math.pow((coord1[1] - coord2[1]), 2))
            # periodic boundary check
            if d >= (self.L/2):
                d -= self.L
            if d < (2 * self.r):
                return True
        return False

    def calculate_energy(self, system):
        """
        Calculates internal energy of the system based on neighbors distance.
        :param system: The system to calculate energy for.
        :return: Energy value.
        """
        return NotImplementedError

    def trial_move(self):
        """
        Initiate a trial move.
            1) Select a random occupied site
            2) Select a random target site
            3) Calculate energy change (Delta U)
            4) If (Delta U) <= 0:
                Update system and energy
            5) If (Delta U) > 0:
                calculate p=exp(-(Delta U)/KT)
                generate a random number p'
                if p' < p : Accept else reject
        """
        return NotImplementedError

    def run(self, n_steps=100):
        """Run MCMC for n number of steps."""
        for i in range(n_steps):
            # step 1: Do trial move

            if i % self.write_freq == 0:
                # step 2: append to system history
                continue

            self.timestep += 1

    def visualize(self, save_path=""):
        """
        Plot the current grid using matplotlib.
        :param save_path: Path to save figure
        """
        return NotImplementedError

    def trajectory(self, save_path=""):
        """
        Find a way to play the trajectory from the system history.
        Note: matplotlib.animation might be helpful(https://matplotlib.org/stable/api/animation_api.html)
        :param save_path: Path to save the trajectory.
        """
        return NotImplementedError
