"""
Indoor environment generation.
"""

import numpy as np


def create_access_points(width: float, height: float, n_ap: int) -> np.ndarray:
    """
    Randomly place n_ap access points within the indoor area.

    Parameters
    ----------
    width : float
        Width of the indoor area.
    height : float
        Height of the indoor area.
    n_ap : int
        Number of access points.

    Returns
    -------
    np.ndarray
        Array of shape (n_ap, 2), where each row is [x, y].
    """
    access_points = np.random.uniform(low=[0, 0], high=[width, height], size=(n_ap, 2))
    return access_points


def generate_user_positions(num: int, width: float, height: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random user positions uniformly inside the area.

    Returns
    -------
    np.ndarray
        Array of shape (num_positions, 2).
    """
    x = rng.uniform(0, width, num)
    y = rng.uniform(0, height, num)
    return np.column_stack((x, y))