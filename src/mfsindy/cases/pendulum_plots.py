# plot_utils.py
"""
Plotting utilities for Burgers and Lorenz experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Consistent colour scheme for Lorenz MF plots
COLORS_MODELS = {
    "HF":   "tab:blue",
    "LF":   "tab:orange",
    "MF":   "tab:green",
    "MF_w": "tab:red",
}


def set_dark_theme(rc=None):
    """Apply a dark theme suitable for Lorenz 3D plots."""
    base = {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "font.size": 12,
    }
    if rc is not None:
        base.update(rc)
    plt.rcParams.update(base)

def plot_multifidelity_trajectories(X_hf, X_lf, X_ref):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    for X in X_lf:
        ax.plot(
            X[::10, 0],
            X[::10, 1],
            ".",
            color="tab:red",
            alpha=0.1,
            linewidth=0.6,
        )

    for X in X_hf:
        ax.plot(
            X[::10, 0],
            X[::10, 1],
            ".",
            color="tab:blue",
            alpha=0.2,
            linewidth=0.8,
        )

    ax.plot(
        X_ref[0][:, 0],
        X_ref[0][:, 1],
        "-",
        color="black",
        alpha=1.0,
        linewidth=1.0,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)

    plt.tight_layout(pad=0.05)
    plt.show()
