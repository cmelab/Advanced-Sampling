import numpy as np
import itertools

class Simulation:
    def __init__(self, n_density=0.5, N=10, kT=1.0, write_freq=5):
        self.n_density = n_density
        self.N = N
        self.kT = kT
        self.n_particles = (N*N) * n_density
        self.grid_indices = list(itertools.product(np.arange(self.N), np.arange(self.N)))
        self.system = self._init_system()
        self.energy = self.calculate_energy(self.system)
        self.timestep = 0
        self.system_history = []
        self.write_freq = write_freq

    def _init_system(self):
        """
        Initialize 2D NxN array, randomly putting particles in the grid.
        occupied sites are 1 and empty sites are 0.
        :return: A 2D numpy array of shape (N, N).
        """
        return NotImplementedError

    def find_neighbors(self, coordinate):
        """
        List of neighbors of a coordinate.
        Note: account for grid edge.
        :param coordinate: (x,y) coordinate
        :return: List of neighbors (x,y)
        """
        return NotImplementedError

    def calculate_energy(self, system):
        """
        Calculates internal energy of the system based on neighbors.
        For each immediate neighbor --> U += -1
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

