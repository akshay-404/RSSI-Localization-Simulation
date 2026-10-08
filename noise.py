"""
Determine the effect of different noise level
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from src.simulation import run_baseline_simulation
from src.config import SimulationConfig

ROOT = lambda x: Path(__file__).resolve().parents[x]
DATA_PATH = ROOT(0) / ('data_testing' if SimulationConfig.testing else 'data_official')

noise_std = [0, 1, 2, 3, 4, 5, 6, 7, 8]
MAE = [[], [], []]
results: dict = {}
output = {'noise_std': list(noise_std)}

for std in noise_std:
    config = SimulationConfig(rssi_noise_std=std)
    simulation = run_baseline_simulation(config)
    results = simulation["results"]
    MAE[0].append(results['Nearest AP']['metrics']['MAE'])
    MAE[1].append(results['Weighted Centroid']['metrics']['MAE'])
    MAE[2].append(results[f'KNN (K={config.k_neighbors})']['metrics']['MAE'])

plt.figure(figsize=(8, 6))
for i, key in enumerate(results.keys()):
    plt.plot(noise_std, MAE[i], label=key)
    output[key] = MAE[i]

plt.xlabel(r"Noise Std, $\sigma$ (dB)")
plt.ylabel("Mean Absolute Error (m)")
plt.xlim((noise_std[0], noise_std[-1]))
plt.title("Effect of RSSI Noise on Localisation Error")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(DATA_PATH / 'noise_effect.png', dpi=600)
plt.savefig(DATA_PATH / 'pdf/noise_effect.pdf')

json.dump(output, open(DATA_PATH / 'noise.json', 'w'), indent=2)