"""
RSSI propagation and measurement simulation.
"""

import numpy as np
import pandas as pd

def calculate_distances(user_positions: np.ndarray, access_points: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between every user position
    and every access point.

    Parameters
    ----------
    user_positions : np.ndarray
        Shape: (M, 2)

    access_points : np.ndarray
        Shape: (N, 2)

    Returns
    -------
    np.ndarray
        Shape: (M, N)
    """
    differences = (user_positions[:, np.newaxis,:] - access_points[np.newaxis, :, :])
    distances = np.linalg.norm(differences, axis=2)
    return distances  # returns a 2D distance grid


def generate_rssi(
        distances: np.ndarray,
        rssi_at_reference: float,
        reference_distance: float,
        path_loss_exponent: float,
        noise_std: float,
        noise_mean: float,
        rng: np.random.Generator) -> np.ndarray:
    """
    Generate RSSI measurements using the log-distance path-loss model.

    RSSI(d) = RSSI(d0) - 10*n*log10(d/d0) + X_sigma

    where X_sigma ~ N(0, noise_std^2).
    """
    distances = np.maximum(distances, 1e-10) # To avoid log10(0)
    mean_rssi = (rssi_at_reference - 10.0 * path_loss_exponent * np.log10(distances / reference_distance))
    noise = rng.normal(loc=noise_mean, scale=noise_std, size=distances.shape)
    rssi = mean_rssi + noise
    return rssi


def create_fingerprint_database(positions: np.ndarray, rssi: np.ndarray, filename: str = "fingerprint.csv") -> pd.DataFrame:
    data = {
        "x": positions[:, 0],
        "y": positions[:, 1],
    }
    for i in range(rssi.shape[1]):
        data[f"AP{i + 1}_RSSI"] = rssi[:, i]
    fingerprint_db = pd.DataFrame(data)
    return fingerprint_db
