import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------
out_dir = Path("./Scores")
scores = np.load(out_dir / "scores_summary.npz", allow_pickle=True)["scores"].item()

# Extract dictionaries
mf_scores = scores["mf"]
lf_scores = scores["lf"]
hf_scores = scores["hf"]

# Parameter grid
n_hf_vals = np.arange(1, 11)
n_lf_vals = np.arange(10, 101, 10)

# Initialize matrices
mf_mat  = np.full((len(n_lf_vals), len(n_hf_vals)), np.nan)
dlf_mat = np.full_like(mf_mat, np.nan, dtype=float)
dhf_mat = np.full_like(mf_mat, np.nan, dtype=float)

# Populate
for i, n_lf in enumerate(n_lf_vals):
    for j, n_hf in enumerate(n_hf_vals):
        # Find matching key (n_hf, n_lf, noise_hf, noise_lf)
        keys = [k for k in mf_scores.keys() if k[0]==n_hf and k[1]==n_lf]
        if not keys:
            continue
        key = keys[0]
        mf = mf_scores[key]
        lf = lf_scores.get(key, np.nan)
        hf = hf_scores.get(key, np.nan)

        mf_mat[i, j]  = mf
        dlf_mat[i, j] = mf - lf
        dhf_mat[i, j] = mf - hf


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def plot_heatmap(matrix, title, cmap="magma", vmin=None, vmax=None, fname=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=r"$R^2$")

    ax.set_xticks(np.arange(len(n_hf_vals)))
    ax.set_xticklabels(n_hf_vals)
    ax.set_yticks(np.arange(len(n_lf_vals)))
    ax.set_yticklabels(n_lf_vals)
    ax.set_xlabel(r"$n_{\mathrm{HF}}$")
    ax.set_ylabel(r"$n_{\mathrm{LF}}$")
    ax.set_title(title)
    plt.tight_layout()

    if fname:
        fig.savefig(out_dir / fname, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------
plot_heatmap(
    mf_mat,
    title=r"$R^2_{\mathrm{MF}}$",
    fname="heatmap_mf_score.png",
)

plot_heatmap(
    dlf_mat,
    title=r"$R^2_{\mathrm{MF}} - R^2_{\mathrm{LF}}$",
    fname="heatmap_mf_minus_lf.png",
)

plot_heatmap(
    dhf_mat,
    title=r"$R^2_{\mathrm{MF}} - R^2_{\mathrm{HF}}$",
    fname="heatmap_mf_minus_hf.png",
)

print("✅ Saved three heatmaps in:", out_dir)
