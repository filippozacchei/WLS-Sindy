"""
main_evaluate_doublependulum.py

Evaluate multi-fidelity SINDy performance on the double pendulum system,
using on-the-fly trajectory generation.

For each (n_LF, n_HF) combination:
    → Generate LF, HF, and combined MF trajectories.
    → Train Ensemble Weak-SINDy models.
    → Evaluate R², coefficient MAD, and disagreement.
    → Save all heatmaps to ./Results_DP.

Outputs:
    ./Results_DP/*.png
    ./Results_DP/doublependulum_results.npz
"""

import numpy as np
import pysindy as ps
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
from sklearn.metrics import r2_score
from generator import generate_dataset  # rename your file accordingly

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def absolute_deviation(C_hat, C_ref, tau=1e-6):
    """Coefficient-wise relative absolute deviation."""
    return np.abs(C_hat - C_ref) / np.maximum(np.abs(C_ref), tau)

def ensemble_disagreement(optimizer):
    """Coefficient ensemble disagreement (median abs. deviation from median)."""
    coefs = np.stack(optimizer.coef_list, axis=0)
    median_coef = np.median(coefs, axis=0)
    abs_diff = np.abs(coefs - median_coef)
    return np.median(abs_diff, axis=0)


# ---------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------
def run_sindy(X_list, t_list, dt, degree=5, threshold=0.1, library=None, weights=None):
    opt = ps.EnsembleOptimizer(ps.STLSQ(threshold=threshold), bagging=True, n_models=20)
    model = ps.SINDy(feature_library=library, optimizer=opt)
    model.fit(X_list, t=dt, sample_weight=weights)
    return model, opt


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def plot_heatmap(matrix, title, n_lf_vals, n_hf_vals,
                 cmap="magma", fname=None, label=r"$R^2$",
                 annotate=False, fmt=".2f"):
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 12,
        "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.linewidth": 0.8, "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    im = ax.imshow(matrix, origin="lower", cmap=cmap,
                   aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(label, rotation=270, labelpad=12)
    cbar.ax.tick_params(length=2, width=0.5)

    ax.set_xticks(np.arange(len(n_hf_vals)))
    ax.set_xticklabels(n_hf_vals)
    ax.set_yticks(np.arange(len(n_lf_vals)))
    ax.set_yticklabels(n_lf_vals)
    ax.set_xlabel(r"$n_{\mathrm{HF}}$")
    ax.set_ylabel(r"$n_{\mathrm{LF}}$")
    ax.set_xticks(np.arange(-0.5, len(n_hf_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(n_lf_vals), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", length=0)

    if annotate:
        for i in range(len(n_lf_vals)):
            for j in range(len(n_hf_vals)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            color="white" if val > np.nanmean(matrix) else "black",
                            fontsize=9)
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main evaluation grid
# ---------------------------------------------------------------------
def evaluate_grid(n_lf_vals, n_hf_vals, runs=5, dt=0.001,
                  threshold=0.1, degree=5, out_dir="./Results_DP"):

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

    # Reference data (HF baseline, noise-free)
    rng = np.random.default_rng(999)
    y0 = rng.uniform(low=[-np.pi/3, -np.pi/3, -1, -1],
                     high=[np.pi/3, np.pi/3, 1, 1])
    ref_data = generate_dataset([y0], n_hf=1, n_lf=0,
                                noise_lf=0, noise_hf=0,
                                save_data=False)
    X_true = ref_data["Y_true"][0]
    t = ref_data["t"]

    deriv = ps.FiniteDifference(order=2)
    Xdot_true = deriv._differentiate(X_true, t=dt)

    # -----------------------------------------------------------------
    # Grid loop
    # -----------------------------------------------------------------
    for i, n_lf in enumerate(tqdm(n_lf_vals, desc="LF grid")):
        for j, n_hf in enumerate(n_hf_vals):
            r2_mf_runs, r2_lf_runs, r2_hf_runs = [], [], []
            mad_mf_runs, mad_lf_runs, mad_hf_runs = [], [], []
            dis_mf_runs, dis_lf_runs, dis_hf_runs = [], [], []

            for run in range(runs):
                rng = np.random.default_rng(run)
                y0s = [rng.uniform(low=[-np.pi/3, -np.pi/3, -1, -1],
                                   high=[np.pi/3, np.pi/3, 1, 1])
                       for _ in range(3)]

                # --- Generate trajectories on the fly ---
                data = generate_dataset(y0s, n_hf, n_lf,
                                        noise_lf=0.5,
                                        noise_hf=0.05,
                                        T=0.5, dt=dt, save_data=False)
                t_grid=data['t']

                Y_true, Y_noisy, sigmas = data["Y_true"], data["Y_noisy"], data["sigma"]
                Y_hf = [y for y, s in zip(Y_noisy, sigmas) if np.isclose(s, 0.05)]
                Y_lf = [y for y, s in zip(Y_noisy, sigmas) if np.isclose(s, 0.5)]

                Y_mf = Y_hf + Y_lf
                w_mf = [(1/0.05)**2]*len(Y_hf) + [(1/0.5)**2]*len(Y_lf)
    
                library = ps.feature_library.WeakPDELibrary(
                    ps.PolynomialLibrary(degree=degree, include_bias=False),
                    spatiotemporal_grid=t_grid, p=2, K=100
                )

                print(1)
                model_hf, opt_hf = run_sindy(Y_hf, [t]*len(Y_hf), dt, degree, threshold, library)
                print(2)
                model_lf, opt_lf = run_sindy(Y_lf, [t]*len(Y_lf), dt, degree, threshold, library)
                print(3)
                model_mf, opt_mf = run_sindy(Y_mf, [t]*len(Y_mf), dt, degree, threshold, library, weights=w_mf)
                print(4)

                lib_eval = ps.PolynomialLibrary(degree=degree, include_bias=False)
                opt_eval_hf = ps.STLSQ(threshold=threshold)
                opt_eval_lf = ps.STLSQ(threshold=threshold)
                opt_eval_mf = ps.STLSQ(threshold=threshold)

                # --- Fit evaluation models only to get same internal structure ---
                #     Then inject coefficients from the weak SINDy results
                eval_model_hf = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval_hf)
                eval_model_lf = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval_lf)
                eval_model_mf = ps.SINDy(feature_library=lib_eval, optimizer=opt_eval_mf)
                
                eval_model_hf.fit([Y_hf[0]], t=dt)
                eval_model_lf.fit([Y_lf[0]], t=dt)
                eval_model_mf.fit([Y_mf[0]], t=dt)
                
                opt_eval_hf.coef_ = opt_hf.coef_
                opt_eval_lf.coef_ = opt_lf.coef_
                opt_eval_mf.coef_ = opt_mf.coef_

                # Evaluate R² on clean reference
                for model, collector, opt in [(eval_model_hf, r2_hf_runs, opt_hf),
                                              (eval_model_lf, r2_lf_runs, opt_lf),
                                              (eval_model_mf, r2_mf_runs, opt_mf)]:
                    Ydot_pred = model.predict(X_true)
                    r2_val = 1 - np.sum((Xdot_true - Ydot_pred)**2) / np.sum((Xdot_true - np.mean(Xdot_true, 0))**2)
                    collector.append(r2_val)

                for opt_src, mad_coll, dis_coll in [(opt_hf, mad_hf_runs, dis_hf_runs),
                                                    (opt_lf, mad_lf_runs, dis_lf_runs),
                                                    (opt_mf, mad_mf_runs, dis_mf_runs)]:
                    C = opt_src.coef_.T
                    mad_coll.append(np.median(np.abs(C)))
                    dis_coll.append(np.median(ensemble_disagreement(opt_src)))

            # Aggregate medians
            mf_score[i, j] = np.median(r2_mf_runs)
            lf_score[i, j] = np.median(r2_lf_runs)
            hf_score[i, j] = np.median(r2_hf_runs)
            dlf_score[i, j] = np.median(np.array(r2_mf_runs) - np.array(r2_lf_runs))
            dhf_score[i, j] = np.median(np.array(r2_mf_runs) - np.array(r2_hf_runs))

            mf_mad[i, j] = np.median(mad_mf_runs)
            lf_mad[i, j] = np.median(mad_lf_runs)
            hf_mad[i, j] = np.median(mad_hf_runs)
            dlf_mad[i, j] = np.median(np.array(mad_mf_runs) - np.array(mad_lf_runs))
            dhf_mad[i, j] = np.median(np.array(mad_mf_runs) - np.array(mad_hf_runs))

            mf_dis[i, j] = np.median(dis_mf_runs)
            lf_dis[i, j] = np.median(dis_lf_runs)
            hf_dis[i, j] = np.median(dis_hf_runs)
            dlf_dis[i, j] = np.median(np.array(dis_mf_runs) - np.array(dis_lf_runs))
            dhf_dis[i, j] = np.median(np.array(dis_mf_runs) - np.array(dis_hf_runs))

    # Save results
    np.savez_compressed(Path(out_dir)/"doublependulum_results.npz",
                        n_lf_vals=n_lf_vals, n_hf_vals=n_hf_vals,
                        mf_score=mf_score, lf_score=lf_score, hf_score=hf_score,
                        dlf_score=dlf_score, dhf_score=dhf_score,
                        mf_mad=mf_mad, lf_mad=lf_mad, hf_mad=hf_mad,
                        dlf_mad=dlf_mad, dhf_mad=dhf_mad,
                        mf_dis=mf_dis, lf_dis=lf_dis, hf_dis=hf_dis,
                        dlf_dis=dlf_dis, dhf_dis=dhf_dis)
    print(f"✅ Saved → {Path(out_dir)/'doublependulum_results.npz'}")
    
    data = np.load(f"{Path(out_dir)/'doublependulum_results.npz'}")
    mf_score = data['mf_score']
    hf_score = data['hf_score']
    lf_score = data['lf_score']
    # Plot heatmaps (same style as Lorenz)
    plot_heatmap(np.clip(mf_score, 0, None), "MF $R^2$", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir)/"scores_mf.png")
    plot_heatmap(np.clip(lf_score, 0, None), "MF−LF $R^2$", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir)/"scores_lf.png")
    plot_heatmap(np.clip(hf_score, 0, None), "MF−HF $R^2$", n_lf_vals, n_hf_vals,
                 fname=Path(out_dir)/"scores_hf.png")
    # plot_heatmap(mf_mad, "MF MAD", n_lf_vals, n_hf_vals,
    #              fname=Path(out_dir)/"mad_mf.png", label="MAD")
    # plot_heatmap(dlf_mad, "MF−LF MAD", n_lf_vals, n_hf_vals,
    #              fname=Path(out_dir)/"mad_mf_minus_lf.png", label="ΔMAD")
    # plot_heatmap(dhf_mad, "MF−HF MAD", n_lf_vals, n_hf_vals,
    #              fname=Path(out_dir)/"mad_mf_minus_hf.png", label="ΔMAD")
    # plot_heatmap(mf_dis, "MF Disagreement", n_lf_vals, n_hf_vals,
    #              fname=Path(out_dir)/"dis_mf.png", label="Disagreement")
    # plot_heatmap(dlf_dis, "MF−LF Disagreement", n_lf_vals, n_hf_vals,
    #              fname=Path(out_dir)/"dis_mf_minus_lf.png", label="ΔDisagreement")
    # plot_heatmap(dhf_dis, "MF−HF Disagreement", n_lf_vals, n_hf_vals,
    #              fname=Path(out_dir)/"dis_mf_minus_hf.png", label="ΔDisagreement")

    print(f"\n✅ Results saved to {out_dir}")

    # --------------------------------------------------------------
    # Pick a representative combination for publication figures
    # --------------------------------------------------------------
    print("\n🎨 Generating publication plots for one representative combination...")

    rng = np.random.default_rng(42)
    i = rng.integers(0, len(n_lf_vals))
    j = rng.integers(0, len(n_hf_vals))
    n_lf_sel = n_lf_vals[i]
    n_hf_sel = n_hf_vals[j]
    print(f"→ Selected (n_LF={n_lf_sel}, n_HF={n_hf_sel})")

    # Generate one realization for visualization
    data = generate_dataset(
        y0s=[[0.8, -0.5, 0.0, 0.5]],
        n_hf=n_hf_sel,
        n_lf=n_lf_sel,
        noise_hf=0.05,
        noise_lf=0.5,
        T=1.0,
        dt=dt,
        save_data=False
    )
    t = data["t"]
    Y_true, Y_noisy, sigmas = data["Y_true"], data["Y_noisy"], data["sigma"]
    Y_hf = [y for y, s in zip(Y_noisy, sigmas) if np.isclose(s, 0.05)]
    Y_lf = [y for y, s in zip(Y_noisy, sigmas) if np.isclose(s, 0.5)]
    Y_mf = Y_hf + Y_lf
    w_mf = [(1/0.05)**2]*len(Y_hf) + [(1/0.5)**2]*len(Y_lf)

    library = ps.feature_library.WeakPDELibrary(
        ps.PolynomialLibrary(degree=degree, include_bias=False),
        spatiotemporal_grid=t,
        p=2, K=100
    )

    model_hf, opt_hf = run_sindy(Y_hf, [t]*len(Y_hf), dt, degree, threshold, library)
    model_lf, opt_lf = run_sindy(Y_lf, [t]*len(Y_lf), dt, degree, threshold, library)
    model_mf, opt_mf = run_sindy(Y_mf, [t]*len(Y_mf), dt, degree, threshold, library, weights=w_mf)

    # Extract coefficients for comparison
    C_hf = np.median(np.stack(opt_hf.coef_list), axis=0)
    C_lf = np.median(np.stack(opt_lf.coef_list), axis=0)
    C_mf = np.median(np.stack(opt_mf.coef_list), axis=0)

    out_dir_pub = Path(out_dir) / "PublicationPlots"
    out_dir_pub.mkdir(exist_ok=True)

    # -----------------------------------------------------------------
    # (1) Coefficient bar comparison  ✅ FIXED SHAPE
    # -----------------------------------------------------------------
    # Each coef_ is (n_states, n_features). We flatten to 1D for visualization.
    C_hf_flat = np.median(np.stack(opt_hf.coef_list), axis=0).ravel()
    C_lf_flat = np.median(np.stack(opt_lf.coef_list), axis=0).ravel()
    C_mf_flat = np.median(np.stack(opt_mf.coef_list), axis=0).ravel()

    n_terms = len(C_hf_flat)
    ind = np.arange(n_terms)
    width = 0.25

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(ind + width, C_mf_flat, width, color="tomato", edgecolor="black", label="MF")
    ax.bar(ind - width, C_lf_flat, width, color="gray", edgecolor="black", label="LF")
    ax.bar(ind, C_hf_flat, width, color="steelblue", edgecolor="black", label="HF")

    ax.set_xlabel("Feature term index")
    ax.set_ylabel("Coefficient value")
    ax.set_title(f"Coefficient comparison (n_LF={n_lf_sel}, n_HF={n_hf_sel})")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(-1, n_terms)
    plt.tight_layout()
    plt.savefig(out_dir_pub / "coeff_comparison.png", dpi=600, transparent=True)
    plt.close(fig)

    # -----------------------------------------------------------------
    # (2) θ₁–ω₁ phase portrait comparison
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.plot(Y_mf[0][:, 0], Y_mf[0][:, 2], color="tomato", lw=1.0, label="MF")
    ax.plot(Y_lf[0][:, 0], Y_lf[0][:, 2], color="gray", alpha=0.3, label="LF")
    ax.plot(Y_hf[0][:, 0], Y_hf[0][:, 2], "b--", lw=1.2, label="HF")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\omega_1$")
    ax.set_title(f"Phase-space reconstruction")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir_pub / "phase_portrait.png", dpi=600, transparent=True)
    plt.close(fig)

    # -----------------------------------------------------------------
    # (3) ω₁(t) temporal signal comparison
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(t, Y_mf[0][:, 2], color="tomato", lw=1.0, label="MF")
    ax.plot(t, Y_lf[0][:, 2], color="gray", alpha=0.4, lw=0.8, label="LF")
    ax.plot(t, Y_hf[0][:, 2], "b--", lw=1.2, label="HF")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\omega_1$ [rad/s]")
    ax.set_title("Temporal signal comparison")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir_pub / "omega_signal.png", dpi=600, transparent=True)
    plt.close(fig)

    print(f"🎨 Publication figures saved to {out_dir_pub}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    n_lf_vals = np.arange(10, 101, 190)
    n_hf_vals = np.arange(1, 11, 19)
    evaluate_grid(n_lf_vals, n_hf_vals, runs=1)
