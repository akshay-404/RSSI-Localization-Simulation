"""
Visualization functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT= lambda x: Path(__file__).resolve().parents[x]
DPI = 300

def plot_environment(access_points, test_positions, width, height):
    """
    Plot the indoor environment and AP locations.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        test_positions[:, 0],
        test_positions[:, 1],
        s=10,
        c='black',
        edgecolors='none',
        alpha=0.5,
        label="Test positions"
    )
    ax.scatter(
        access_points[:, 0],
        access_points[:, 1],
        marker="^",
        s=180,
        c='orange',
        label="Wi-Fi AP"
    )

    for i, (x, y) in enumerate(access_points):
        ax.annotate(
            f"AP{i + 1}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Indoor Environment and Access Point Placement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_actual_vs_estimated(actual_positions, estimated_positions, access_points, method, width, height):
    """
    Compare actual and estimated user positions.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        access_points[:, 0],
        access_points[:, 1],
        marker="^",
        s=180,
        c='orange',
        label="Wi-Fi AP"
    )
    ax.scatter(
        actual_positions[:, 0],
        actual_positions[:, 1],
        s=10,
        c='black',
        edgecolors='none',
        alpha=0.7,
        label="Actual"
    )
    ax.scatter(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        s=10,
        c='green',
        edgecolors='none',
        alpha=0.7,
        label="Estimated"
    )
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title(f"Actual vs Estimated Positions - {method}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_cdf(results):
    """
    Plot cumulative distribution of localization error.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, data in results.items():
        errors = np.sort(data["errors"])
        cumulative_probability = (np.arange(1, len(errors) + 1)/ len(errors))
        ax.plot(
            errors,
            cumulative_probability,
            label=method
        )
    ax.set_xlabel("Localization error (m)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Cumulative Distribution of Localization Error")
    ax.set_xlim(0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig