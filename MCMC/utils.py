import numpy as np
from numba import jit


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


@jit(nopython=True)
def get_distance(pos1, pos2, L):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    if dx > L / 2:
        dx -= L
    elif dx < -L / 2:
        dx += L
    if dy > L / 2:
        dy -= L
    elif dy < -L / 2:
        dy += L
    d = (dx ** 2 + dy ** 2) ** 0.5
    return d


@jit(nopython=True)
def pair_distances(pos_array, L, r_cut):
    distances = []
    for i, pos in enumerate(pos_array):
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
