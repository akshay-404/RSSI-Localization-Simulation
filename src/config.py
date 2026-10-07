"""
Simulation parameters for the Wi-Fi RSSI localization simulation.
"""

from dataclasses import dataclass
from time import time

@dataclass
class SimulationConfig:
    # Indoor environment
    area_width: float = 20.0
    area_height: float = 20.0

    # Access points
    num_aps: int = 4

    # RSSI propagation model
    rssi_at_reference: float = -40.0   # dBm
    reference_distance: float = 1.0    # m
    path_loss_exponent: float = 3.0
    rssi_noise_mean: float | list = 0.0       # dB
    rssi_noise_std: float | list = 2.0        # dB

    # Simulation
    num_test_positions: int = 1000
    random_seed: int = int(time())

    # KNN
    num_fingerprint_positions: int = 1000
    k_neighbors: int = 3

    # Weighted KNN
    epsilon: float = 1e-6

    # Simulation testing
    testing = True