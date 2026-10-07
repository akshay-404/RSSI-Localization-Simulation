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

ap_num = [1, 2, 3, 4, 5, 6, 7, 8]
MAE = [[], [], []]
results: dict = {}
output = {'ap_nums': ap_num}

for ap in ap_num:
    config = SimulationConfig(num_aps=ap)
    simulation = run_baseline_simulation(config)
    results = simulation["results"]
    MAE[0].append(results['Nearest AP']['metrics']['MAE'])
    MAE[1].append(results['Weighted Centroid']['metrics']['MAE'])
    MAE[2].append(results[f'KNN (K={config.k_neighbors})']['metrics']['MAE'])

plt.figure(figsize=(8, 6))
for i, key in enumerate(results.keys()):
    plt.plot(ap_num, MAE[i], label=key)
    output[key] = MAE[i]

plt.xlabel(r"Number of Access Points")
plt.ylabel("Mean Absolute Error (m)")
plt.xlim((ap_num[0], ap_num[-1]))
plt.title("Effect of Number of AP on Localisation Error")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig((DATA_PATH) / 'ap_effect.png', dpi=600)

json.dump(output, open(DATA_PATH / 'ap_effect.json', 'w'), indent=2)