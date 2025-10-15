"""
plot_synthetic_doublependulum_results_with_mad.py

Creates synthetic, publication-quality figures illustrating
expected outcomes of multi-fidelity Weak-SINDy learning
on the double pendulum system.

Plots:
    (a) R² heatmaps for LF, HF, MF
    (b) MAD heatmaps for LF, HF, MF
    (c) Representative synthetic coefficient/trajectory plots

All data are synthetic and generated to qualitatively match
expected trends: R² ↑ with n_LF and n_HF, MAD ↓ with n_LF and n_HF.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

out_dir = Path("./Figures_Synthetic_DP")
out_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Synthetic setup
# ---------------------------------------------------------------------
n_lf_vals = np.arange(10, 101, 10)
n_hf_vals = np.arange(1, 11, 1)
N_lf, N_hf = len(n_lf_vals), len(n_hf_vals)

# Normalized grids
lf_grid = (n_lf_vals - n_lf_vals.min()) / (n_lf_vals.max() - n_lf_vals.min())
hf_grid = (n_hf_vals - n_hf_vals.min()) / (n_hf_vals.max() - n_hf_vals.min())
LF, HF = np.meshgrid(lf_grid, hf_grid, indexing="ij")

# ---------------------------------------------------------------------
# R² surfaces — increase with n_LF and n_HF
# ---------------------------------------------------------------------
lf_R2 = 0.55 + 0.25 * LF + 0.05 * HF + 0.01 * np.random.randn(N_lf, N_hf)
hf_R2 = 0.75 + 0.15 * HF + 0.05 * LF + 0.01 * np.random.randn(N_lf, N_hf)
mf_R2 = 0.80 + 0.15 * LF + 0.20 * HF + 0.02 * np.random.randn(N_lf, N_hf)
lf_R2, hf_R2, mf_R2 = map(lambda M: np.clip(M, 0, 1), [lf_R2, hf_R2, mf_R2])

# ---------------------------------------------------------------------
# MAD surfaces — decrease with n_LF and n_HF
# ---------------------------------------------------------------------
mad_lf = 0.25 - 0.1 * LF - 0.03 * HF + 0.02 * np.random.rand(N_lf, N_hf)
mad_hf = 0.15 - 0.05 * HF - 0.02 * LF + 0.01 * np.random.rand(N_lf, N_hf)
mad_mf = 0.10 - 0.07 * HF - 0.04 * LF + 0.01 * np.random.rand(N_lf, N_hf)
mad_lf, mad_hf, mad_mf = map(lambda M: np.clip(M, 0, None), [mad_lf, mad_hf, mad_mf])

# ---------------------------------------------------------------------
# Helper for heatmaps
# ---------------------------------------------------------------------
def plot_heatmap(matrix, title, n_lf_vals, n_hf_vals, cmap="magma", label="", fname=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    im = ax.imshow(matrix, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(label, rotation=270, labelpad=12)
    ax.set_xticks(np.arange(len(n_hf_vals)))
    ax.set_xticklabels(n_hf_vals)
    ax.set_yticks(np.arange(len(n_lf_vals)))
    ax.set_yticklabels(n_lf_vals)
    ax.set_xlabel(r"$n_{\mathrm{HF}}$")
    ax.set_ylabel(r"$n_{\mathrm{LF}}$")
    ax.set_title(title)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=600, transparent=True)
    plt.close(fig)

# ---------------------------------------------------------------------
# 1️⃣ R² heatmaps
# ---------------------------------------------------------------------
plot_heatmap(lf_R2, "LF $R^2$", n_lf_vals, n_hf_vals, label=r"$R^2$", fname=out_dir/"heatmap_lf_r2.png", vmin=0.5, vmax=1)
plot_heatmap(hf_R2, "HF $R^2$", n_lf_vals, n_hf_vals, label=r"$R^2$", fname=out_dir/"heatmap_hf_r2.png", vmin=0.5, vmax=1)
plot_heatmap(mf_R2, "MF $R^2$", n_lf_vals, n_hf_vals, label=r"$R^2$", fname=out_dir/"heatmap_mf_r2.png", vmin=0.5, vmax=1)

# ---------------------------------------------------------------------
# 2️⃣ MAD heatmaps
# ---------------------------------------------------------------------
plot_heatmap(mad_lf, "LF MAD", n_lf_vals, n_hf_vals, label="MAD", fname=out_dir/"heatmap_lf_mad.png", cmap="cividis", vmin=0, vmax=0.25)
plot_heatmap(mad_hf, "HF MAD", n_lf_vals, n_hf_vals, label="MAD", fname=out_dir/"heatmap_hf_mad.png", cmap="cividis", vmin=0, vmax=0.25)
plot_heatmap(mad_mf, "MF MAD", n_lf_vals, n_hf_vals, label="MAD", fname=out_dir/"heatmap_mf_mad.png", cmap="cividis", vmin=0, vmax=0.25)

print(f"✅ Synthetic R² and MAD figures saved in {out_dir}")
