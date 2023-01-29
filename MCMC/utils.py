import freud
import numpy as np
import gsd.hoomd
from numba import jit

def avg_nn(gsd_file, frame=-1, r_max=3):
    with gsd.hoomd.open(gsd_file) as f:
        snap = f[frame]
        box = snap.configuration.box
        points = snap.particles.position
        frame_avg_nn = []
        aq = freud.locality.AABBQuery(box, points)
        nlist = aq.query(points, {'r_max': 3}).toNeighborList()
        neighbors = []
        for i in range(snap.particles.N):
            neighbors.append(np.where(nlist[:, 0] == i)[0].shape[0]-1)
            frame_avg_nn.append(np.average(neighbors))
    return np.average(frame_avg_nn)

def structure_factor(gsd_file, start=0, stop=-1, num_k_values=100, k_max=10):
    with gsd.hoomd.open(gsd_file) as f:
        snap = f[0]
        sf = freud.diffraction.StaticStructureFactorDebye(
                num_k_values=num_k_values, k_max=k_max
                )
        for snap in f[start:stop]:
            box = snap.configuration.box
            points = snap.particles.position
        sf.compute((box, points))
    return sf


def rdf(gsd_file, frame=-1, bins=50, r_max=None):
    with gsd.hoomd.open(gsd_file) as f:
        snap = f[frame]
        box = snap.configuration.box
        points = snap.particles.position
        if not r_max:
            r_max = np.nextafter(
                    np.max(snap.configuration.box[:3]) * 0.3, 0, dtype=np.float32
                    )
        rdf = freud.density.RDF(bins, r_max)
        rdf.compute((box,points))
    return rdf


def inverse_distance_attractive(distances):
    """
    Calculates internal energy given center-center distance (r).
    Energy = - 1. / r*r
    :param distances: 1D array of distances.
    :return: Total energy.
    """
    return (-1. / np.power(distances, 2)).sum()


def inverse_distance_repulsive(distances):
    """
    Calculates internal energy given center-center distance (r).
    Energy = - 1. / r*r
    :param distances: 1D array of distances.
    :return: Total energy.
    """
    return (1. / np.power(distances, 2)).sum()


def lj_energy(distances, epsilon=1.0, sigma=1.0, n=12, m=6):
    """
    Calculates Lennard-Jones pair potential.
    :param distances: 1D array of distances.
    :param epsilon: Energy parameter epsilon.
    :param sigma: Particle size sigma.
    :param n: Repulsive power factor.
    :param m: Attractive power factor.
    :return: Total energy.
    """
    return 4 * epsilon * (np.power(sigma / distances, n) - np.power(sigma / distances, m)).sum()

@jit(nopython=True)
def get_distance(pos1, pos2, L):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    if dx >= L / 2:
        dx -= L
    elif dx < -L / 2:
        dx += L
    if dy >= L / 2:
        dy -= L
    elif dy < -L / 2:
        dy += L
    d = (dx ** 2 + dy ** 2) ** 0.5
    return d


@jit(nopython=True)
def pair_distances(pos_array, L, r_cut):
    distances = []
    for i, pos in enumerate(pos_array[:-1]):
        for pos2 in pos_array[i + 1:]:
            d = get_distance(pos2, pos, L)
            if d <= r_cut:
                distances.append(d)
    return distances


@jit(nopython=True)
def check_overlap(system, index, L, r):
    """
    Check if particle with specified index overlaps with the other particles in the system.
    :param system: 2D array of particle positions.
    :param index: index of the particle.
    :return: True if there is any overlap, else False.
    """
    pos1 = system[index]
    for i, pos2 in enumerate(system):
        if i == index:
            continue
        d = get_distance(pos1, pos2, L)
        if d < (2 * r):
            return True
    return False
