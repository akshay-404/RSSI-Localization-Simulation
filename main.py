"""
Main entry point for the baseline Wi-Fi RSSI localization project.
"""

import json
from pathlib import Path
from src.config import SimulationConfig
from src.simulation import run_baseline_simulation
from src.visualization import (
    plot_environment,
    plot_actual_vs_estimated,
    plot_error_cdf
)

config = SimulationConfig()

ROOT = lambda x: Path(__file__).resolve().parents[x]
DATA_PATH = ROOT(0) / ('data_testing' if config.testing else 'data_official')

def dump(simulation, config:SimulationConfig) -> dict:
    output = {}
    output['num_AP'] = config.num_aps
    output['num_fingerprint'] = config.num_fingerprint_positions
    output['num_test_positions'] = config.num_test_positions
    output['metrics'] = {}
    results = simulation['results']
    for key, value in results.items():
        output['metrics'][key] = value['metrics']
    return output

def main():
    simulation = run_baseline_simulation(config)
    access_points = simulation["access_points"]
    test_positions = simulation["test_positions"]
    results = simulation["results"]
    simulation['fingerprint_df'].to_csv(DATA_PATH / 'rssi_fingerprint.csv', index=False)

    json.dump(
        dump(simulation, config), 
        open(DATA_PATH / 'main.json', 'w'), 
        indent=2
    )

    plot_environment(
        access_points,
        test_positions,
        config.area_width,
        config.area_height
    ).savefig(DATA_PATH / 'environment.png', dpi=600)

    for method, data in results.items():
        plot_actual_vs_estimated(
            test_positions,
            data["positions"],
            access_points,
            method,
            config.area_width,
            config.area_height,
        ).savefig(DATA_PATH / f'estimated_{method}.png', dpi=600)

    plot_error_cdf(results).savefig(DATA_PATH / 'cumulative_error.png', dpi=600)


if __name__ == "__main__":
    main()