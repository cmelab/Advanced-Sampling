import gsd.hoomd
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time

from MCMC.utils import pair_distances, check_overlap


class Simulation:
    def __init__(
            self,
            n_density=0.5,
            n_particles=5,
            r=0.5,
            kT=1.0,
            r_cut=1,
            max_trans=0.5,
            energy_write_freq=100,
            trajectory_write_freq=10000,
            energy_func=None,
            hard_sphere=True,
            **kwargs):
        """
        :param n_density: Number density.
        :param r: Disk radius.
        :param kT: Kinetic temperature.
        :param r_cut: Neighbor distance cut off.
        :param max_trans: Max move size.
        :param write_freq: Save system history frequency.
        :param energy_func: Function to calculate energy.
        """
        self.n_density = n_density
        self.r = r
        self.n_particles = n_particles
        self.L = (math.pow(self.n_particles, 0.5)) / (math.pow(self.n_density, 0.5))
        self.kT = kT
        self.r_cut = r_cut
        self.max_trans = max_trans
        self.system = self._init_system()
        self.timestep = 0
        self.system_history = []
        self.system_history_writes = 0
        self.energies = []
        self.accepted_moves = 0
        self.rejected_moves = 0
        self._tps = []
        self.energy_write_freq = energy_write_freq
        self.trajectory_write_freq = trajectory_write_freq
        self.energy_func = energy_func
        self.hard_sphere = hard_sphere
        self.kwargs = kwargs


    @property
    def tps(self):
        return np.mean(self._tps)

    @property
    def acceptance_ratio(self):
        return self.accepted_moves / self.timestep

    @property
    def energy(self):
        return self.calculate_energy(self.system)

    def _init_system(self):
        """
        Initialize an array of 2D positions, randomly putting disk in the box.
        x and y coordinates are between -L/2 and L/2.
        :return: A 2D numpy array of shape (self.n_particles, 2).
        """
        disks_per_row = math.floor((self.L - self.r) / (2 * self.r))
        n_rows = math.ceil(self.n_particles / disks_per_row)
        init_x_even = (-self.L / 2) + self.r
        init_x_odd = (-self.L / 2) + (2 * self.r)
        init_y = (-self.L / 2) + self.r
        system = []
        for i in np.arange(n_rows):
            row_disk_counter = 0
            if i % 2 == 0:
                row_init_x = init_x_even
            else:
                row_init_x = init_x_odd
            while row_disk_counter < disks_per_row and len(system) <= self.n_particles:
                system.append([row_init_x + (row_disk_counter * 2 * self.r), init_y])
                row_disk_counter += 1
            init_y += 2 * self.r
        system = np.asarray(system)
        return system

    def calculate_energy(self, system, overlap=False):
        """
        Calculates internal energy of the system based on neighbors distance.
        :param system: The system to calculate energy for.
        :param overlap: If disk overlap exists.
        :return: Energy value.
        """
        if overlap and self.hard_sphere:
            return np.inf

        if not self.energy_func:
            return 0
        else:
            distances = pair_distances(system, self.L, self.r_cut)
            return self.energy_func(np.asarray(distances), **self.kwargs)

    def _calculate_distance(self, coord1, coord2):
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        dx, dy = self._periodic_boundary(dx, dy)
        d = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        return d

    def _periodic_boundary(self, x, y):
        """
        Check periodic boundary conditions and update x and y accordingly.
        :param x: x coordinate
        :param y: y coordinate
        :return: updated x and y coordinates
        """
        if x >= self.L / 2:
            x -= self.L
        elif x <= -self.L / 2:
            x += self.L
        if y >= self.L / 2:
            y -= self.L
        elif y <= -self.L / 2:
            y += self.L

        return x, y

    def trial_move(self):
        """"""
        # Pick a random particle; store initial value:
        move_idx = random.randint(0, self.system.shape[0] - 1)
        original_coords = tuple(self.system[move_idx])
        # Uniformly sample a direction and move distance
        direction = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, self.max_trans)
        # Update the coordinates of the particle
        new_x = self.system[move_idx][0] + distance * np.cos(direction)
        new_y = self.system[move_idx][1] + distance * np.sin(direction)
        new_x, new_y = self._periodic_boundary(new_x, new_y)
        return move_idx, original_coords, (new_x, new_y)

    def run(self, n_steps=100):
        """Run MCMC for n number of steps."""
        start = time.time()
        for i in range(int(n_steps)):
            initial_energy = self.energy
            # Make move; get particle, original and new coordinates
            move_idx, original_coords, new_coords = self.trial_move()
            self.system[move_idx] = new_coords
            overlap = check_overlap(self.system, move_idx, self.L, self.r)
            trial_energy = self.calculate_energy(self.system, overlap)
            if np.isfinite(trial_energy):
                delta_U = trial_energy - initial_energy
                if delta_U <= 0:  # Move accepted; keep updated self.system 
                    self.accepted_moves += 1
                else:
                    rand_num = random.uniform(0, 1)
                    if np.exp(-delta_U / self.kT) <= rand_num:
                        # Move accepted; keep updated self.system
                        self.accepted_moves += 1
                    else:  # Move rejected; change self.system to prev state
                        self.system[move_idx] = original_coords
                        self.rejected_moves += 1
            else:  # Energy is infinite (overlapping hard spheres)
                self.system[move_idx] = original_coords
                self.rejected_moves += 1

            if i % self.energy_write_freq == 0:
                self.energies.append(self.energy)
            if i % self.trajectory_write_freq == 0:
                self.system_history.append(np.copy(self.system))
            #if len(self.system_history) == 500:
            #    self._update_trajectory()
            #    self.system_history.clear()

            self.timestep += 1
        end = time.time()
        self._tps.append(np.round(n_steps / (end - start), 3))
        self._update_trajectory()

    def visualize(self, frame_number=0, save_path=None):
        """
        Plot the current grid using matplotlib.
        :param save_path: Path to save figure
        """
        figure, axes = plt.subplots()
        for i in self.system_history[frame_number]:
            colored_circle = plt.Circle((i[0], i[1]), self.r)
            axes.add_artist(colored_circle)
        axes.set_aspect(1)
        plt.title('System')
        plt.ylim(-self.L / 2, self.L / 2)
        plt.xlim(-self.L / 2, self.L / 2)
        if save_path:
            fig_name = "system_frame_" + str(frame_number)
            plt.savefig(os.path.join(save_path, fig_name))
        return plt

    def trajectory(self, save_path=""):
        """
        Find a way to play the trajectory from the system history.
        Note: matplotlib.animation might be helpful(https://matplotlib.org/stable/api/animation_api.html)
        :param save_path: Path to save the trajectory.
        """
        return NotImplementedError
    
    def save_system(self, frame=-1):
        """
        Save a single snapshot of system_history to a .txt file.
        :param frame: Index number of system_history
        """
        system_array = self.system_history[frame]
        np.savetxt(fname="system.txt", X=system_array)

    def _update_trajectory(self):
        with gsd.hoomd.open("trajectory.gsd", "wb") as traj:
            for sys in self.system_history:
                snap = gsd.hoomd.Snapshot()
                snap.particles.N = sys.shape[0] 
                snap.configuration.box = np.array([self.L, self.L, 0, 0, 0, 0])
                coords = np.append(
                        sys, np.zeros((sys.shape[0], 1)), axis=1
                )
                snap.particles.position = coords 
                snap.particles.types = ["A"]
                snap.particles.typeid = np.array([0]*snap.particles.N)
                traj.append(snap)

    def _load_system(self):
        return NotImplementedError
