"""
Baseline Wi-Fi RSSI localization simulation.
"""

import numpy as np
from .environment import create_access_points, generate_user_positions
from .rssi import calculate_distances, generate_rssi, create_fingerprint_database
from .localization import (
    nearest_ap_localization, 
    weighted_centroid_localization, 
    knn_fingerprint_localization,
)
from .metrics import calculate_metrics, localization_errors


def run_baseline_simulation(config) -> dict:
    """
    Run the complete baseline experiment.
    """
    rng = np.random.default_rng(config.random_seed)

    # 1. Create indoor environment
    access_points = create_access_points(
        config.area_width,
        config.area_height,
        config.num_aps
    )

    # 2. Generate fingerprint/reference positions
    fingerprint_positions = generate_user_positions(
        config.num_fingerprint_positions,
        config.area_width,
        config.area_height,
        rng
    )

    # 3. Generate RSSI fingerprints
    fingerprint_distances = calculate_distances(
        fingerprint_positions,
        access_points
    )
    fingerprint_rssi = generate_rssi(
        fingerprint_distances,
        config.rssi_at_reference,
        config.reference_distance,
        config.path_loss_exponent,
        config.rssi_noise_std,
        config.rssi_noise_mean,
        rng
    )
    fingerprint_df = create_fingerprint_database(fingerprint_positions, fingerprint_rssi)

    # 4. Generate independent test positions
    test_positions = generate_user_positions(
        config.num_test_positions,
        config.area_width,
        config.area_height,
        rng
    )

    # 5. Generate test RSSI measurements
    test_distances = calculate_distances(
        test_positions,
        access_points
    )
    test_rssi = generate_rssi(
        test_distances,
        config.rssi_at_reference,
        config.reference_distance,
        config.path_loss_exponent,
        config.rssi_noise_std,
        config.rssi_noise_mean,
        rng
    )

    # 6. Nearest AP
    nearest_ap_positions = nearest_ap_localization(
        test_rssi,
        access_points
    )
    nearest_ap_errors = localization_errors(
        test_positions,
        nearest_ap_positions
    )
    nearest_ap_metrics = calculate_metrics(
        nearest_ap_errors
    )

    # 7. Weighted Centroid
    centroid_positions = weighted_centroid_localization(
        test_rssi,
        access_points
    )
    centroid_errors = localization_errors(
        test_positions,
        centroid_positions
    )
    centroid_metrics = calculate_metrics(
        centroid_errors
    )

    # 8. KNN fingerprinting
    knn_positions = knn_fingerprint_localization(
        test_rssi,
        fingerprint_rssi,
        fingerprint_positions,
        config.k_neighbors
    )
    knn_errors = localization_errors(
        test_positions,
        knn_positions
    )
    knn_metrics = calculate_metrics(
        knn_errors
    )

    # 9. Collect results
    results = {
        "Nearest AP": {
            "positions": nearest_ap_positions,
            "errors": nearest_ap_errors,
            "metrics": nearest_ap_metrics
        },

        "Weighted Centroid": {
            "positions": centroid_positions,
            "errors": centroid_errors,
            "metrics": centroid_metrics
        },

        f"KNN (K={config.k_neighbors})": {
            "positions": knn_positions,
            "errors": knn_errors,
            "metrics": knn_metrics
        }
    }

    return {
        "access_points": access_points,
        "fingerprint_positions": fingerprint_positions,
        "fingerprint_rssi": fingerprint_rssi,
        "fingerprint_df": fingerprint_df,
        "test_positions": test_positions,
        "test_rssi": test_rssi,
        "results": results
    }