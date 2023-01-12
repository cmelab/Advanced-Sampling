import numpy as np


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
