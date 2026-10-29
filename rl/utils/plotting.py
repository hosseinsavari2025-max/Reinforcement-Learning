"""Plot learning curves from the CSV files written by ``CSVLogger``."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(csv_path: str | Path, out_path: str | Path, window: int = 20, title: str = "") -> None:
    steps, returns = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            returns.append(float(row["episode_return"]))

    returns_arr = np.asarray(returns)
    smoothed = np.convolve(returns_arr, np.ones(window) / window, mode="valid") if len(returns_arr) >= window else returns_arr

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, returns_arr, alpha=0.25, label="episode return")
    ax.plot(steps[len(steps) - len(smoothed):], smoothed, label=f"moving average ({window})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
