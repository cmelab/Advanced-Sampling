import numpy as np
import itertools

class Simulation:
    def __init__(self, n_density=0.5, N=10, kT=1.0, write_freq=5):
        self.n_density = n_density
        self.N = N
        self.kT = kT
        self.n_particles = int((N*N) * n_density)
        self.grid_indices = list(itertools.product(np.arange(self.N), np.arange(self.N)))
        self.system = self._init_system()
        self.timestep = 0
        self.system_history = []
        self.energies = []
        self.write_freq = write_freq
        self.accepted_moves = 0

    def _init_system(self):
        """
        Initialize 2D NxN array, randomly putting particles in the grid.
        occupied sites are 1 and empty sites are 0.
        :return: A 2D numpy array of shape (N, N).
        """
        return NotImplementedError
    
    @property
    def energy(self):
        return self.calculate_energy(self.system)

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
        _system = np.copy(self.system)
        non_zero_indices = list(np.nonzero(self.system))
        random.shuffle(non_zero_indices)
        site1 = non_zero_indices[0]
        site1_val = _system[site1]
        all_indices = list(self.grid_indices) 
        random.shuffle(all_indices)
        site2 = all_indices[0]
        site2_val = _system[site2]

        _system[site1] = site2_val
        _system[site2] = site1_val
        delta_U = self.calculate_energy(_system) - self.energy

        if delta_U <= 0:
            self.system = _system
            self.accepted_moves += 1
        else:
            rand_num = random.uniform(0, 1)
            if np.exp(-delta_U/self.kT) <= rand_num:
                self.system = _system
                self.accepted_moves += 1
            else:
                pass

    def run(self, n_steps=100):
        """Run MCMC for n number of steps."""
        start = time.time()
        for i in range(n_steps):
            self.trial_move()

            if i % self.write_freq == 0:
                self.system_history.append(self.system)
                self.energies.append(self.energy)

            self.timestep += 1
        end = time.time()
        self.total_time_sec = end - start
        self.tps = n_steps / self.total_time_sec

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

