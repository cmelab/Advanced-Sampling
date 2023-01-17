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
def pair_distances(pos_array, L, r_cut):
    distances = []
    for i, pos in enumerate(pos_array):
        for pos2 in pos_array[i+1:]:
            dx = pos2[0]-pos[0]
            dy = pos2[1]-pos[1]            
            if dx > L/2:
                dx -= L
            elif dx < -L/2:
                dx += L
            if dy > L/2:
                dy -= L
            elif dy < -L/2:
                dy += L
            d = (dx**2 + dy**2)**0.5
            if d <= r_cut:
                distances.append(d)
    return distances
