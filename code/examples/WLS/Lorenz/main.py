"""
main_evaluate_lorenz.py

Evaluate multi-fidelity SINDy performance on the Lorenz system,
using on-the-fly trajectory generation (no dataset files).

For each (n_LF, n_HF) combination:
    → Generate LF, HF, and combined MF trajectories.
    → Train Ensemble Weak-SINDy models.
    → Evaluate R², coefficient MAD, and disagreement.
    → Save 3×3 heatmaps to ./Results.

Outputs:
    ./Results/scores_mf.png
    ./Results/scores_mf_minus_lf.png
    ./Results/scores_mf_minus_hf.png
    ./Results/mad_mf.png
    ./Results/mad_mf_minus_lf.png
    ./Results/mad_mf_minus_hf.png
    ./Results/dis_mf.png
    ./Results/dis_mf_minus_lf.png
    ./Results/dis_mf_minus_hf.png
"""

import numpy as np
import pysindy as ps
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from sklearn.metrics import mean_squared_error


from generator import generate_lorenz_data

import warnings 
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def absolute_deviation(C_hat, C_ref, tau=1e-6):
    """Coefficient-wise relative absolute deviation."""
    return np.abs(C_hat - C_ref) / np.maximum(np.abs(C_ref), tau)


def ensemble_disagreement(optimizer):
    """Coefficient ensemble disagreement."""
    coefs = np.stack(optimizer.coef_list, axis=0)
    median_coef = np.median(coefs, axis=0)
    abs_diff = np.abs(coefs - median_coef)
    return np.median(abs_diff, axis=0)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def run_sindy(X_list, t_list, dt, degree=2, threshold=0.1, library=None, weights=None):
    opt = ps.EnsembleOptimizer(ps.STLSQ(threshold=threshold), bagging=True, n_models=50)
    model = ps.SINDy(feature_library=library, optimizer=opt)
    model.fit(X_list, t=dt, sample_weight=weights)
    return model, opt


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def plot_heatmap(
    matrix,
    title,
    n_lf_vals,
    n_hf_vals,
    cmap="magma",
    fname=None,
    label=r"$R^2$",
    annotate=False,
    fmt=".2f",
):
    """
    Produce a publication-quality heatmap.

    Parameters
    ----------
    matrix : 2D ndarray
        Heatmap values.
    title : str
        Figure title.
    n_lf_vals, n_hf_vals : array-like
        Axis tick values.
    cmap : str, optional
        Colormap name (default: 'viridis').
    fname : str or None, optional
        Save path (PDF/PNG).
    label : str, optional
        Colorbar label.
    annotate : bool, optional
        Overlay numeric annotations.
    fmt : str, optional
        Numeric format for annotations.
    """

    # --- Matplotlib defaults for paper aesthetics ---
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

    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # column width ≈ 90 mm

    # --- Plot heatmap ---
    im = ax.imshow(matrix, origin="lower", cmap=cmap, aspect="auto", interpolation="nearest")

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(label, rotation=270, labelpad=12)
    cbar.ax.tick_params(length=2, width=0.5)

    # --- Axes ticks and labels ---
    ax.set_xticks(np.arange(len(n_hf_vals)))
    ax.set_xticklabels(n_hf_vals)
    ax.set_yticks(np.arange(len(n_lf_vals)))
    ax.set_yticklabels(n_lf_vals)
    ax.set_xlabel(r"$n_{\mathrm{HF}}$")
    ax.set_ylabel(r"$n_{\mathrm{LF}}$")

    # --- Add gridlines for clarity ---
    ax.set_xticks(np.arange(-0.5, len(n_hf_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(n_lf_vals), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", length=0)

    # --- Annotate cells ---
    if annotate:
        for i in range(len(n_lf_vals)):
            for j in range(len(n_hf_vals)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:{fmt}}",
                        ha="center", va="center",
                        color="white" if val > np.nanmean(matrix) else "black",
                        fontsize=9,
                    )

    # --- Layout and save ---
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def evaluate_grid(
    n_lf_vals,
    n_hf_vals,
    runs=3,
    dt=1e-3,
    threshold=0.5,
    degree=2,
    out_dir="./Results",
):
    Path(out_dir).mkdir(exist_ok=True)

    mf_score = np.full((len(n_lf_vals), len(n_hf_vals)), np.nan)
    lf_score = np.full_like(mf_score, np.nan)
    hf_score = np.full_like(mf_score, np.nan)    
    dlf_score = np.full_like(mf_score, np.nan)
    dhf_score = np.full_like(mf_score, np.nan)

    mf_mad = np.full_like(mf_score, np.nan)
    lf_mad = np.full_like(mf_score, np.nan)
    hf_mad = np.full_like(mf_score, np.nan)    
    dlf_mad = np.full_like(mf_score, np.nan)
    dhf_mad = np.full_like(mf_score, np.nan)

    mf_dis = np.full_like(mf_score, np.nan)
    lf_dis = np.full_like(mf_score, np.nan)
    hf_dis = np.full_like(mf_score, np.nan)    
    dlf_dis = np.full_like(mf_score, np.nan)
    dhf_dis = np.full_like(mf_score, np.nan)

    # Ground-truth Lorenz model coefficients (for MAD)
    C_true = np.zeros((9, 3))
    C_true[0, 0] = -10.0     # x
    C_true[1, 0] = 10.0      # y
    C_true[0, 1] = 28.0      # x
    C_true[5, 1] = -1.0      # xz
    C_true[1, 1] = -1.0      # y
    C_true[4, 2] = 1.0       # xy
    C_true[2, 2] = -8.0/3.0  # z

    # Test data (for R²)
    X_test, Xdot_test, t_test = generate_lorenz_data(n_traj=1, noise_level=0.0, seed=999)
    std_per_dim = np.std(X_test[0])
    
    for i, n_lf in enumerate(tqdm(n_lf_vals, desc="LF grid")):
        for j, n_hf in enumerate(n_hf_vals):
            r2_mf_runs, r2_lf_runs, r2_hf_runs = [], [], []
            mad_mf_runs, mad_lf_runs, mad_hf_runs = [], [], []
            dis_mf_runs, dis_lf_runs, dis_hf_runs = [], [], []

            for run in range(runs):
                # Generate LF, HF trajectories
                X_hf, _, t_hf = generate_lorenz_data(n_hf, noise_level=0.025*std_per_dim, T=0.1, seed=run)
                X_lf, _, t_lf = generate_lorenz_data(n_lf, noise_level=0.25*std_per_dim, T=0.1, seed=run + 100)

                X_mf = X_hf + X_lf
                t_mf = t_hf + t_lf
                w_mf = [(1/0.01)**2]*n_hf + [(1/0.1)**2]*n_lf

                # Polynomial library
                library = ps.feature_library.WeightedWeakPDELibrary(
                    ps.PolynomialLibrary(degree=2, include_bias=False),
                    spatiotemporal_weights=np.ones_like(t_hf[0]),
                    spatiotemporal_grid=t_hf[0],
                    p=2, K=100
                )
                # Train
                model_hf, opt_hf = run_sindy(X_hf, t_hf, dt, degree, threshold, library)
                model_lf, opt_lf = run_sindy(X_lf, t_lf, dt, degree, threshold, library)
                model_mf, opt_mf = run_sindy(X_mf, t_mf, dt, degree, threshold, library, w_mf)
                
                lib_eval = ps.PolynomialLibrary(degree=degree, include_bias=False)
                opt_eval_hf = ps.STLSQ(threshold=threshold)
                opt_eval_lf = ps.STLSQ(threshold=threshold)
                opt_eval_mf = ps.STLSQ(threshold=threshold)

                # --- Fit evaluation models only to get same internal structure ---
                #     Then inject coefficients from the weak SINDy results
                eval_model_hf = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval_hf)
                eval_model_lf = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval_lf)
                eval_model_mf = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval_mf)
                
                eval_model_hf.fit([X_hf[0]], t=dt)
                eval_model_lf.fit([X_hf[0]], t=dt)
                eval_model_mf.fit([X_hf[0]], t=dt)
                
                opt_eval_hf.coef_ = opt_hf.coef_
                opt_eval_lf.coef_ = opt_lf.coef_
                opt_eval_mf.coef_ = opt_mf.coef_

                # Evaluate
                r2_hf = eval_model_hf.score(X_test, t=dt)
                r2_lf = eval_model_lf.score(X_test, t=dt)
                r2_mf = eval_model_mf.score(X_test, t=dt)

                # Coefficients
                C_hf, C_lf, C_mf = opt_hf.coef_.T, opt_lf.coef_.T, opt_mf.coef_.T

                mad_hf = np.median(absolute_deviation(C_hf, C_true))
                mad_lf = np.median(absolute_deviation(C_lf, C_true))
                mad_mf = np.median(absolute_deviation(C_mf, C_true))

                dis_hf = np.median(ensemble_disagreement(opt_hf))
                dis_lf = np.median(ensemble_disagreement(opt_lf))
                dis_mf = np.median(ensemble_disagreement(opt_mf))

                # Collect
                r2_hf_runs.append(r2_hf)
                r2_lf_runs.append(r2_lf)
                r2_mf_runs.append(r2_mf)
                mad_hf_runs.append(mad_hf)
                mad_lf_runs.append(mad_lf)
                mad_mf_runs.append(mad_mf)
                dis_hf_runs.append(dis_hf)
                dis_lf_runs.append(dis_lf)
                dis_mf_runs.append(dis_mf)

    
            # Aggregate medians across runs
            mf_score[i, j] = np.median(r2_mf_runs)
            lf_score[i, j] = np.median(np.array(r2_lf_runs))
            hf_score[i, j] = np.median(np.array(r2_hf_runs))            
            dlf_score[i, j] = np.median(np.array(r2_mf_runs) - np.array(r2_lf_runs))
            dhf_score[i, j] = np.median(np.array(r2_mf_runs) - np.array(r2_hf_runs))

            mf_mad[i, j] = np.median(mad_mf_runs)
            lf_mad[i, j] = np.median(np.array(mad_lf_runs))
            hf_mad[i, j] = np.median(np.array(mad_hf_runs))
            dlf_mad[i, j] = np.median(np.array(mad_mf_runs) - np.array(mad_lf_runs))
            dhf_mad[i, j] = np.median(np.array(mad_mf_runs) - np.array(mad_hf_runs))

            mf_dis[i, j] = np.median(dis_mf_runs)
            lf_dis[i, j] = np.median(np.array(dis_lf_runs))
            hf_dis[i, j] = np.median(np.array(dis_hf_runs))            
            dlf_dis[i, j] = np.median(np.array(dis_mf_runs) - np.array(dis_lf_runs))
            dhf_dis[i, j] = np.median(np.array(dis_mf_runs) - np.array(dis_hf_runs))
            
    np.savez_compressed(
        Path(out_dir) / "lorenz_results.npz",
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        mf_score=mf_score,
        lf_score=lf_score,
        hf_score=hf_score,
        dlf_score=dlf_score,
        dhf_score=dhf_score,
        mf_mad=mf_mad,
        lf_mad=lf_mad,
        hf_mad=hf_mad,
        dlf_mad=dlf_mad,
        dhf_mad=dhf_mad,
        mf_dis=mf_dis,
        lf_dis=lf_dis,
        hf_dis=hf_dis,
        dlf_dis=dlf_dis,
        dhf_dis=dhf_dis,
    )
    print("✅ Saved data arrays →", Path(out_dir) / "lorenz_results.npz")

    # -----------------------------------------------------------------
    # Plot results
    # -----------------------------------------------------------------
    plot_heatmap(np.clip(mf_score,a_min=0,a_max=None), "MF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_mf.png")
    plot_heatmap(np.clip(lf_score,a_min=0,a_max=None), "LF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_lf.png")
    plot_heatmap(np.clip(hf_score,a_min=0,a_max=None), "HF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_hf.png")
    plot_heatmap(np.clip(dlf_score,a_min=None,a_max=1), "MF−LF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_mf_minus_lf.png")
    plot_heatmap(np.clip(dhf_score,a_min=None,a_max=1), "MF−HF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_mf_minus_hf.png")

    plot_heatmap(mf_mad, "MF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_mf.png", label="MAD")
    plot_heatmap(lf_mad, "MF−LF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_lf.png", label="MAD")
    plot_heatmap(hf_mad, "MF−HF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_hf.png", label="MAD")
    plot_heatmap(dlf_mad, "MF−LF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_mf_minus_lf.png", label="ΔMAD")
    plot_heatmap(dhf_mad, "MF−HF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_mf_minus_hf.png", label="ΔMAD")

    plot_heatmap(mf_dis, "MF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf.png", label="Disagreement")
    plot_heatmap(lf_dis, "LF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_lf.png", label="Disagreement")
    plot_heatmap(hf_dis, "HF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf_hf.png", label="Disagreement")    
    plot_heatmap(dlf_dis, "MF−LF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf_minus_lf.png", label="ΔDisagreement")
    plot_heatmap(dhf_dis, "MF−HF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf_minus_hf.png", label="ΔDisagreement")

    print(f"\n✅ Results saved to {out_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    n_lf_vals = np.arange(10, 101, 10)
    n_hf_vals = np.arange(1, 11, 1)
    evaluate_grid(n_lf_vals, n_hf_vals, runs=25)
