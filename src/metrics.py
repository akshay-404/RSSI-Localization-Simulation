"""
Localization performance metrics.
"""

import numpy as np


def localization_errors(actual_positions: np.ndarray, estimated_positions: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean localization error for every sample.
    """
    differences = (actual_positions - estimated_positions)
    errors = np.linalg.norm(differences, axis=1)
    return errors


def mae(errors: np.ndarray) -> float:
    """Mean Absolute Localization Error."""
    return float(np.mean(errors))


def rmse(errors: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean(errors ** 2)))


def median_error(errors: np.ndarray) -> float:
    """Median localization error."""
    return float(np.median(errors))


def accuracy_within(errors: np.ndarray, threshold: float) -> float:
    """
    Percentage of predictions whose localization
    error is less than or equal to the threshold.
    """
    percentage = (np.mean(errors <= threshold)* 100.0)
    return float(percentage)


def calculate_metrics(errors: np.ndarray) -> dict:
    """
    Calculate all baseline performance metrics.
    """
    return {
        "MAE": mae(errors),
        "RMSE": rmse(errors),
        "Median": median_error(errors),
        "Accuracy_1m": accuracy_within(errors, 1.0),
        "Accuracy_2m": accuracy_within(errors, 2.0),
        "Accuracy_3m": accuracy_within(errors, 3.0),
    }