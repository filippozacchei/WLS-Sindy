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

def plot_multifidelity_trajectories(grid, U_hf, U_lf, U_ref):

    k_mid = 0  # fixed time index

    u_hf  = U_hf[:, :, k_mid, 2]   # (Nx, Ny)
    u_lf  = U_lf[:, :, k_mid, 2]   # (Nx, Ny)
    u_ref = U_ref[:,     :, k_mid, 2]    # (Nx, Ny)

    x = grid[:, :, k_mid, 0]   # (Nx, Ny)
    y = grid[:, :, k_mid, 1]   # (Nx, Ny)

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        x, y, u_lf,
        color="tab:red",
        alpha=0.25,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot_surface(
        x, y, u_hf,
        color="tab:blue",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot(
        x[:, 0],
        y[:, 0],
        u_ref[:, 0],
        color="black",
        linewidth=1.0,
    )

    # Remove ticks / axes for table-friendly panel
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_axis_off()
    ax.grid(False)
