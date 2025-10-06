import sys
sys.path.append("../../../src")

import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps

from utils import (
    DataConfig, ESINDyConfig,
    generate_data, run_models, generate_all_data, lorenz_true_coefficients
)

# ---------------- Data configuration ----------------
data_cfg = DataConfig(
    dt_lf=0.001,
    dt_hf=0.001,
    lf_noise=np.array([50]),   # 50% LF noise
    hf_noise=np.array([10]),   # 10% HF noise
    n_trajectories_lf=np.array([50]),
    n_trajectories_hf=np.array([10]),
    t_end_train_hf=0.1,
    t_end_train_lf=0.1,
    t_end_test=5.0,
    n_trajectories_test=1,
)

es_cfg = ESINDyConfig(
    n_ensembles=100,  # smaller for speed; increase for more robustness
    n_runs=10,
    library_functions=ps.PolynomialLibrary(degree=2, include_bias=False),
)

# ---------------- Test data ----------------
x_test, t_test = generate_data(
    dt=data_cfg.dt_hf,
    t_end=data_cfg.t_end_test,
    noise_level=0.0,
    n_trajectories=data_cfg.n_trajectories_test,
    seed=123
)

# ---------------- Generate training pools ----------------
data_dict, max_n_hf, max_n_lf = generate_all_data(data_cfg, es_cfg, base_seed=1)

# ---------------- Repeat experiment ----------------
n_runs = 200
all_results = []

for run_id in range(n_runs):
    # regenerate data with new seed every run
    data_dict, max_n_hf, max_n_lf = generate_all_data(data_cfg, es_cfg, base_seed=run_id)
    print(run_id)
    res = run_models(
        data_dict=data_dict,
        lf_noise=50,
        hf_noise=10,
        n_lf=50,
        n_hf=10,
        es_config=es_cfg,
        x_test=x_test,
        t_test=t_test,
        library_functions=es_cfg.library_functions,
        max_n_hf=max_n_hf,
        max_n_lf=max_n_lf,
        run_id=run_id
    )
    all_results.append(res)

# ---------------- Aggregate coefficients ----------------
def stack_coefs(all_results, key_list):
    """Stack all coefficient lists across runs, return (all, median)."""
    coef_list = np.concatenate([r[key_list] for r in all_results], axis=0)
    coef_median = np.median(coef_list, axis=0)
    return coef_list, coef_median

models = {
    "LF only": stack_coefs(all_results, "coef_lf_list"),
    "HF only": stack_coefs(all_results, "coef_hf_list"),
    "MF": stack_coefs(all_results, "coef_mf_list"),
}

import seaborn as sns
sns.set_context("paper")
sns.set_style("whitegrid")

features = ["x", "y", "z", "x²", "xy", "xz", "y²", "yz", "z²"]
components = ["dx/dt", "dy/dt", "dz/dt"]
# Muted palette
colors = {"LF only": "#e41a1c", "HF only": "#377eb8", "MF": "#4daf4a"}

C_true = lorenz_true_coefficients()
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_context("paper")
sns.set_style("whitegrid")

features   = ["x", "y", "z", "x²", "xy", "xz", "y²", "yz", "z²"]
components = ["dx/dt", "dy/dt", "dz/dt"]
colors     = {"LF only": "#e41a1c", "HF only": "#377eb8", "MF": "#4daf4a"}

C_true = lorenz_true_coefficients()

for j, comp in enumerate(components):
    fig, ax = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(features))
    all_vals = []

    legend_handles, legend_labels = [], []

    for k, (label, (coef_list, coef_median)) in enumerate(models.items()):
        coef_list_j   = coef_list[:, :, j]   # (n_ens, n_features)
        coef_median_j = coef_median[:, j]
        all_vals.append(coef_list_j)

        # group separation
        pos_shifted = positions + (k - 1) * 0.32

        # boxplot per feature
        bp = ax.boxplot(
            coef_list_j,
            positions=pos_shifted,
            widths=0.25,
            patch_artist=True,
            showfliers=False
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.3)
        for median in bp["medians"]:
            median.set_color(colors[label])
            median.set_linewidth(2)

        # overlay median point
        ax.scatter(pos_shifted, coef_median_j, s=48, color=colors[label],
                   edgecolor="k", linewidths=0.5, zorder=3)

        # legend entry
        legend_handles.append(Line2D([0], [0], color=colors[label], lw=6, alpha=0.5))
        legend_labels.append(label)

    # true coefficients only for active terms
    for i, val in enumerate(C_true[:, j]):
        if abs(val) > 1e-12:
            xmin, xmax = positions[i] - 0.45, positions[i] + 0.45
            ax.hlines(val, xmin, xmax, colors='black', lw=1.4, ls='--', zorder=2)
            legend_handles.append(Line2D([0], [0], color='black', ls='--', lw=1.4))
            legend_labels.append(f"True term: {features[i]}")

    # axis formatting
    all_vals = np.concatenate(all_vals, axis=0)
    ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.1 * max(1e-6, ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(features, rotation=45, fontsize=11)
    ax.set_ylabel("Coefficient value", fontsize=12)
    ax.set_title(f"{comp}", fontsize=13)

    # deduplicate legend
    seen, H, L = set(), [], []
    for h, l in zip(legend_handles, legend_labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    ax.legend(H, L, frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    plt.show()
