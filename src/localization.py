"""
Wi-Fi RSSI localization algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

def nearest_ap_localization(rssi: np.ndarray, access_points: np.ndarray) -> np.ndarray:
    """
    Estimate position using the access point with the strongest RSSI.

    Parameters
    ----------
    rssi : np.ndarray
        RSSI measurements, shape (M, N)

    access_points : np.ndarray
        AP coordinates, shape (N, 2)

    Returns
    -------
    np.ndarray
        Estimated positions, shape (M, 2)
    """
    strongest_ap_indices = np.argmax(rssi, axis=1)
    estimated_positions = access_points[strongest_ap_indices]
    return estimated_positions


def weighted_centroid_localization(rssi: np.ndarray, access_points: np.ndarray) -> np.ndarray:
    """
    Weighted centroid localization.

    The reference uses:

        w_i = 10^(RSSI_i / 10)

    and:

        x_hat = sum(w_i*x_i) / sum(w_i)
        y_hat = sum(w_i*y_i) / sum(w_i)
    """
    weights = 10 ** (rssi / 10.0)
    weighted_x = np.sum(weights * access_points[:, 0], axis=1)
    weighted_y = np.sum(weights * access_points[:, 1], axis=1)
    weight_sum = np.sum(weights, axis=1)
    estimated_x = weighted_x / weight_sum
    estimated_y = weighted_y / weight_sum
    estimated_positions = np.column_stack((estimated_x, estimated_y))
    return estimated_positions


def knn_fingerprint_localization(
        query_rssi: np.ndarray,
        fingerprint_rssi: np.ndarray,
        fingerprint_positions: np.ndarray,
        k: int = 3) -> np.ndarray:
    """
    KNN-based RSSI fingerprint localization.

    A KNN regressor is trained using RSSI vectors as features
    and (x, y) coordinates as targets.
    """
    k = min(k, len(fingerprint_rssi))
    model = KNeighborsRegressor(n_neighbors=k, weights="uniform", metric="euclidean")
    model.fit(fingerprint_rssi, fingerprint_positions)
    estimated_positions = model.predict(query_rssi)
    return estimated_positions