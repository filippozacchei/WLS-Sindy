"""
main_evaluate_compressible.py

Evaluates multi-fidelity SINDy performance for the 2D isothermal
compressible Navier–Stokes system. Generates LF, HF, and MF
trajectories on the fly, computes R² scores, coefficient absolute
deviations, and ensemble disagreements, and produces 3×3 summary
heatmaps.

Outputs:
    heatmaps_scores_*.png
    heatmaps_mad_*.png
    heatmaps_disagreement_*.png
"""

import numpy as np
import pysindy as ps
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from generator import generate_compressible_flow, plot_snapshot, animate_field, compare_trajectories



# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def absolute_deviation(C_hat, C_ref, tau=1e-6):
    """Compute per-coefficient relative absolute deviation."""
    return np.abs(C_hat - C_ref) / np.maximum(np.abs(C_ref), tau)


def ensemble_disagreement(optimizer):
    """Compute per-coefficient ensemble disagreement."""
    coefs = np.stack(optimizer.coef_list, axis=0)
    median_coef = np.median(coefs, axis=0)
    abs_diff = np.abs(coefs - median_coef)
    return np.median(abs_diff, axis=0)


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def run_sindy(X_list, t_list, dt, degree=2, threshold=0.1, library=None, weights=None):
    opt = ps.EnsembleOptimizer(ps.STLSQ(threshold=threshold), bagging=True, n_models=50)
    model = ps.SINDy(feature_library=library, optimizer=opt)
    model.fit(X_list, t=dt, sample_weight=weights)
    return model, opt


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def plot_heatmap(matrix, title, n_lf_vals, n_hf_vals, cmap="magma", fname=None, label=r"$R^2$"):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, origin="lower", cmap=cmap)
    plt.colorbar(im, ax=ax, label=label)
    ax.set_xticks(np.arange(len(n_hf_vals)))
    ax.set_xticklabels(n_hf_vals)
    ax.set_yticks(np.arange(len(n_lf_vals)))
    ax.set_yticklabels(n_lf_vals)
    ax.set_xlabel(r"$n_{\mathrm{HF}}$")
    ax.set_ylabel(r"$n_{\mathrm{LF}}$")
    ax.set_title(title)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main experimental driver
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

    # Containers
    mf_score = np.full((len(n_lf_vals), len(n_hf_vals)), np.nan)
    dlf_score = np.full_like(mf_score, np.nan)
    dhf_score = np.full_like(mf_score, np.nan)

    mf_mad = np.full_like(mf_score, np.nan)
    dlf_mad = np.full_like(mf_score, np.nan)
    dhf_mad = np.full_like(mf_score, np.nan)

    mf_dis = np.full_like(mf_score, np.nan)
    dlf_dis = np.full_like(mf_score, np.nan)
    dhf_dis = np.full_like(mf_score, np.nan)

    for i, n_lf in enumerate(tqdm(n_lf_vals, desc="LF grid")):
        for j, n_hf in enumerate(n_hf_vals):
            r2_mf_runs, r2_lf_runs, r2_hf_runs = [], [], []
            mad_mf_runs, mad_lf_runs, mad_hf_runs = [], [], []
            dis_mf_runs, dis_lf_runs, dis_hf_runs = [], [], []

            for run in range(runs):
                # Generate LF and HF data
                u_test, grid, t_test, L, T = generate_compressible_flow(
                    1, N=64, Nt=20, T=0.1, noise_level=0.0, seed=run
                )
                print(1)   
                u_hf, _, t_hf, _, _ = generate_compressible_flow(
                    n_hf, N=64, Nt=20,  T=0.1, noise_level=0.0, seed=run+runs
                )
                print(2)   
                u_lf, _, t_lf, _, _ = generate_compressible_flow(
                    n_lf, N=64, Nt=20,  T=0.1, noise_level=0.0, seed=run+2*runs
                )
                print(3)   

                # Match MF data (combine)
                u_mf = u_hf + u_lf
                t_mf = t_hf + t_lf
                w_mf = [(1/0.01)**2]*n_hf + [(1/0.1)**2]*n_lf

                # Simple library for now (polynomial)
                library_functions = [
                    lambda x: x,
                    lambda x: 1 / (1e-6 + np.abs(x))
                ]
                library_function_names = [
                    lambda x: x,
                    lambda x: x + "^-1"
                ]

                # Create a custom feature library
                custom_library = ps.CustomLibrary(
                    library_functions=library_functions,
                    function_names=library_function_names
                )

                # Build weak PDE library using your custom features
                lib = ps.WeakPDELibrary(
                    custom_library,
                    spatiotemporal_grid=grid,
                    derivative_order=2,
                    K=20,                         # number of test functions
                    H_xt=[L/10, L/10, T/10],
                )

                # Train
                model_hf, opt_hf = run_sindy(u_hf, t_hf, dt, degree, threshold, lib)
                model_lf, opt_lf = run_sindy(u_lf, t_lf, dt, degree, threshold, lib)
                model_mf, opt_mf = run_sindy(u_mf, t_mf, dt, degree, threshold, lib, w_mf)

                # Evaluate on HF data
                r2_hf = model_hf.score([u_test], t=dt)
                print(r2_hf)
                r2_lf = model_lf.score([u_test], t=dt)
                print(r2_lf)
                r2_mf = model_mf.score([u_test], t=dt)
                print(r2_mf)

                # Coeffs (just use learned)
                C_hf = opt_hf.coef_
                C_lf = opt_lf.coef_
                C_mf = opt_mf.coef_

                mad_hf = np.median(np.abs(C_hf))
                mad_lf = np.median(np.abs(C_lf))
                mad_mf = np.median(np.abs(C_mf))

                dis_hf = np.median(ensemble_disagreement(opt_hf))
                dis_lf = np.median(ensemble_disagreement(opt_lf))
                dis_mf = np.median(ensemble_disagreement(opt_mf))

                # Append
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
            dlf_score[i, j] = np.median(np.array(r2_mf_runs) - np.array(r2_lf_runs))
            dhf_score[i, j] = np.median(np.array(r2_mf_runs) - np.array(r2_hf_runs))

            mf_mad[i, j] = np.median(mad_mf_runs)
            dlf_mad[i, j] = np.median(np.array(mad_mf_runs) - np.array(mad_lf_runs))
            dhf_mad[i, j] = np.median(np.array(mad_mf_runs) - np.array(mad_hf_runs))

            mf_dis[i, j] = np.median(dis_mf_runs)
            dlf_dis[i, j] = np.median(np.array(dis_mf_runs) - np.array(dis_lf_runs))
            dhf_dis[i, j] = np.median(np.array(dis_mf_runs) - np.array(dis_hf_runs))

    # -----------------------------------------------------------------
    # Plot results
    # -----------------------------------------------------------------
    plot_heatmap(mf_score, "MF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_mf.png")
    plot_heatmap(dlf_score, "MF−LF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_mf_minus_lf.png")
    plot_heatmap(dhf_score, "MF−HF $R^2$", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"scores_mf_minus_hf.png")

    plot_heatmap(mf_mad, "MF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_mf.png", label="MAD")
    plot_heatmap(dlf_mad, "MF−LF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_mf_minus_lf.png", label="ΔMAD")
    plot_heatmap(dhf_mad, "MF−HF MAD", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"mad_mf_minus_hf.png", label="ΔMAD")

    plot_heatmap(mf_dis, "MF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf.png", label="Disagreement")
    plot_heatmap(dlf_dis, "MF−LF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf_minus_lf.png", label="ΔDisagreement")
    plot_heatmap(dhf_dis, "MF−HF Disagreement", n_lf_vals, n_hf_vals, fname=Path(out_dir)/"dis_mf_minus_hf.png", label="ΔDisagreement")

    print(f"\n✅ Results saved to {out_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    
    # Generate multiple trajectories
    print(1)
    trajectories, grid, ts, L, T = generate_compressible_flow(n_traj=3, N=64, Nt=100, T=0.5, L=5, noise_level=0.1)
    print(2)

    # Plot one trajectory snapshot
    plot_snapshot(trajectories[0], ts[0], L, idx=10, title_prefix="Trajectory 1: ")

    # Animate velocity field
    animate_field(trajectories[0], ts[0], L, var_index=0, title="Taylor–Green flow", save_path="flow_u.gif")

    # Compare several trajectories
    compare_trajectories(trajectories, ts[0], L, component=2, idx=20)

    n_lf_vals = np.arange(5, 56, 10)
    n_hf_vals = np.arange(1, 11, 5)
    evaluate_grid(n_lf_vals, n_hf_vals, runs=1)
