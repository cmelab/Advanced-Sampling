import itertools
import math
import numpy as np
import random


class Simulation:
    def __init__(self, n_density=0.5, r=0.1, r_factor=5, kT=1.0, r_cut=1, max_trans=0.2, write_freq=5,
                 energy_func=None, hard_sphere=True):
        """

        :param n_density: Number density.
        :param r: Disk radius.
        :param r_factor: Box length is r * r_factor.
        :param kT: Kinetic temperature.
        :param r_cut: Neighbor distance cut off.
        :param max_trans: Max move size.
        :param write_freq: Save system history frequency.
        :param energy_func: Function to calculate energy.
        """
        self.n_density = n_density
        self.r = r
        self.r_factor = r_factor
        self.L = r * r_factor
        self.kT = kT
        self.r_cut = r_cut
        self.max_trans = max_trans
        self.n_particles = math.floor((math.pow(self.L, 2) * self.n_density) / (math.pi * math.pow(self.r, 2)))
        if self.n_particles == 0:
            raise ValueError("cannot fit any disk with this density! "
                             "Either decrease density/disk radius or increase box size.")
        self.system = self._init_system()
        self.timestep = 0
        self.tps = None
        self.system_history = []
        self.energies = []
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.tps = None
        self.write_freq = write_freq
        self.energy_func = energy_func
        self.hard_sphere = hard_sphere
    
    @property
    def acceptance_ratio(self):
        return self.accepted_moves / self.rejected_moves

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
            d = self._calculate_distance(coord1, coord2)
            if d < (2 * self.r):
                return True
        return False

    def calculate_energy(self, system, overlap=False):
        """
        Calculates internal energy of the system based on neighbors distance.
        :param system: The system to calculate energy for.
        :return: Energy value.
        """
        if overlap and self.hard_sphere:
            return np.inf

        if not self.energy_func:
            return 0

        distances = []
        for (i, j) in itertools.combinations(np.arange(self.n_particles), 2):
            coord1 = system[i]
            coord2 = system[j]
            d = self._calculate_distance(coord1, coord2)
            if d <= self.r_cut:
                distances.append(d)

        return self.energy_func(np.asarray(distances))

    def _calculate_distance(self, coord1, coord2):
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        dx, dy = self._periodic_boundary(dx, dy)
        d = np.sqrt(np.pow(dx, 2) + np.pow(dy, 2))
        return d

    def _periodic_boundary(self, x, y):
        """
        Check periodic boundary conditions and update x and y accordingly.
        :param x: x coordinate
        :param y: y coordinate
        :return: updated x and y coordinates
        """
        if x > self.L/2:
            x -= self.L
        elif x < -self.L/2:
            x += self.L
        if y > self.L/2:
            y -= self.L
        elif y < -self.L/2:
            y += self.L

        return x, y

    def trial_move(self):
        """"""
        # Pick a random particle:
        move_idx = random.randint(0, self.system.shape[0])
        # Uniformly sample a direction and move distance
        direction = random.uniform(0, math.pi)
        distance = random.uniform(0, self.max_trans) 
        # Update the coordinates of the particle
        trial_system = np.copy(self.system)
        new_x = trial_system[move_idx][0] + distance * np.cos(direction)
        new_y = trial_system[move_idx][1] + distance * np.sin(direction)
        new_x, new_y = self._periodic_boundary(new_x, new_y)
        trial_system[move_idx][0] = new_x 
        trial_system[move_idx][1] = new_y 
        return trial_system, move_idx

    def run(self, n_steps=100):
        """Run MCMC for n number of steps."""
        start = time.time()
        for i in range(n_steps):
            trial_system, move_idx = trial_move()
            # Check for overlapping particles
            overlap = self.check_overlap(trial_system, move_idx)
            trial_energy = self.calculate_energy(trial_system, overlap)
            if np.isfinite(trial_energy):
                delta_U = trial_energy - self.energy
                if delta_U <= 0:
                    # update self.system
                    self.system = trial_system
                    self.accepted_moves += 1
                else:
                    rand_num = random.uniform(0, 1)
                    if np.exp(-delta_U/self.kT) <= rand_num:
                        self.system = trial_system
                        self.accepted_moves += 1
                    else:
                        self.rejected_moves += 1
            else: # Energy is infinite (overlapping hard spheres)
                self.rejected_moves += 1

            if i % self.write_freq == 0:
                self.energies.append(self.energy)
                self.system_history.append(self.system)

            self.timestep += 1
        end = time.time()
        self.tps = np.round(n_steps / (end-start), 3)

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

