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

def plot_multifidelity_trajectories(X_hf, X_lf, X_ref, x, t):
    traj_idx = 0

    u_hf = X_hf[traj_idx][:, :, 0]   # (Nx, Nt)
    u_lf = X_lf[traj_idx][:, :, 0]   # (Nx, Nt)
    u_ref = X_ref[0][:, :, 0]              # (Nx, Nt)

    x = np.squeeze(x)                 # (Nx,)
    t = np.squeeze(t)                # (Nt,)

    Xg, Tg = np.meshgrid(x, t, indexing="ij")  # (Nx, Nt)

    fig = plt.figure(figsize=(5, 3.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        Xg,
        Tg,
        u_lf,
        color="tab:red",
        alpha=0.25,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    ax.plot_surface(
        Xg,
        Tg,
        u_hf,
        color="tab:blue",
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    k_ref = 0
    ax.plot(
        x,
        np.full_like(x, t[k_ref]),
        u_ref[:, k_ref],
        color="black",
        linewidth=1.0,
    )

    # Compact panel: no ticks, no axes, no titles
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_axis_off()
    ax.grid(False)

    plt.tight_layout(pad=0.05)
    plt.show()
