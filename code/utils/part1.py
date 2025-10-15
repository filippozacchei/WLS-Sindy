from metrics import *
from training import * 
import numpy as np
import pysindy as ps
from tqdm import tqdm
from pathlib import Path

def _init_metrics(shape):
    """Create metric containers for LF/HF/MF comparisons."""
    def zeros(): return np.full(shape, np.nan)
    return dict(
        mf=zeros(), lf=zeros(), hf=zeros(),
        dlf=zeros(), dhf=zeros()
    )

def _train_models(X_lf, t_lf, X_hf, t_hf, grid, degree, threshold, K=100, weights=None):
    """Train LF, HF, and MF ensemble SINDy models."""
    # Build library once
    library = ps.feature_library.WeakPDELibrary(
        ps.PolynomialLibrary(degree=degree, include_bias=False),
        spatiotemporal_grid=grid,
        p=2, K=100,
    )

    # Train models
    model_hf, opt_hf = run_ensemble_sindy(X_hf, t_hf, threshold=threshold, library=library)
    model_lf, opt_lf = run_ensemble_sindy(X_lf, t_lf, threshold=threshold, library=library)
    X_mf, t_mf = X_hf + X_lf, t_hf + t_lf
    model_mf, opt_mf = run_ensemble_sindy(X_mf, t_mf, threshold=threshold, library=library, weights=weights)

    return dict(hf=(model_hf, opt_hf), lf=(model_lf, opt_lf), mf=(model_mf, opt_mf))


def _evaluate_models(models, X_ref, dt, X_test, C_true=None):
    """Compute R², MAD, and disagreement for each model."""
    metrics = dict(r2={}, mad={}, dis={})
    for k, (model, opt) in models.items():
        eval_model, _ = copy_sindy(model, [X_ref], dt)
        metrics["r2"][k] = eval_model.score(X_test, t=dt)
        metrics["dis"][k] = np.median(ensemble_disagreement(opt))
        if C_true is not None:
            metrics["mad"][k] = np.median(absolute_deviation(opt.coef_.T, C_true))
    return metrics


def _aggregate_runs(metric_runs, key):
    """Aggregate medians across runs for a given metric."""
    return {k: np.median(metric_runs[k]) for k in metric_runs}


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def evaluate_mf_sindy(
    generator,
    system_name: str,
    n_lf_vals,
    n_hf_vals,
    runs: int = 5,
    dt: float = 1e-3,
    threshold: float = 0.5,
    degree: int = 2,
    out_dir: str = "./Results",
    C_true=None,
):
    """
    Generic multi-fidelity SINDy evaluation loop (compact version).
    """
    out_dir = Path(out_dir) / system_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preallocate metric arrays
    shape = (len(n_lf_vals), len(n_hf_vals))
    score, mad, dis = _init_metrics(shape), _init_metrics(shape), _init_metrics(shape)

    # Prepare a clean test trajectory for evaluating R²
    X_test, _, _ = generator(n_traj=1, noise_level=0.0, seed=999)
    std_per_dim = np.std(X_test[0])

    for i, n_lf in enumerate(tqdm(n_lf_vals, desc=f"{system_name}: LF grid")):
        for j, n_hf in enumerate(n_hf_vals):

            all_runs = {
                "r2": {"hf": [], "lf": [], "mf": []},
                "mad": {"hf": [], "lf": [], "mf": []},
                "dis": {"hf": [], "lf": [], "mf": []},
            }

            for run in range(runs):
                X_hf, grid_hf, t_hf = generator(n_hf, noise_level=0.025 * std_per_dim, T=0.1, seed=run)
                X_lf, _, t_lf = generator(n_lf, noise_level=0.25 * std_per_dim, T=0.1, seed=run + 100)
                weights = [(1 / 0.01) ** 2] * n_hf + [(1 / 0.1) ** 2] * n_lf

                models = _train_models(X_lf, t_lf, X_hf, t_hf, grid_hf, dt, degree, threshold, weights)
                metrics = _evaluate_models(models, X_hf[0], dt, X_test, C_true)

                for metric_name in ("r2", "mad", "dis"):
                    for fidelity in metrics[metric_name]:
                        all_runs[metric_name][fidelity].append(metrics[metric_name][fidelity])

            agg_r2 = _aggregate_runs(all_runs["r2"], "r2")
            agg_mad = _aggregate_runs(all_runs["mad"], "mad")
            agg_dis = _aggregate_runs(all_runs["dis"], "dis")

            score["mf"][i, j] = agg_r2["mf"]
            score["lf"][i, j] = agg_r2["lf"]
            score["hf"][i, j] = agg_r2["hf"]
            score["dlf"][i, j] = agg_r2["mf"] - agg_r2["lf"]
            score["dhf"][i, j] = agg_r2["mf"] - agg_r2["hf"]

            dis["mf"][i, j] = agg_dis["mf"]
            dis["lf"][i, j] = agg_dis["lf"]
            dis["hf"][i, j] = agg_dis["hf"]
            dis["dlf"][i, j] = agg_dis["mf"] - agg_dis["lf"]
            dis["dhf"][i, j] = agg_dis["mf"] - agg_dis["hf"]

            if C_true is not None:
                mad["mf"][i, j] = agg_mad["mf"]
                mad["lf"][i, j] = agg_mad["lf"]
                mad["hf"][i, j] = agg_mad["hf"]
                mad["dlf"][i, j] = agg_mad["mf"] - agg_mad["lf"]
                mad["dhf"][i, j] = agg_mad["mf"] - agg_mad["hf"]

    np.savez_compressed(
        out_dir / f"{system_name}_results.npz",
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        **{
            f"{key}_{name}": value
            for name, metric_group in zip(["score", "mad", "dis"], (score, mad, dis))
            for key, value in metric_group.items()
        },
    )

    print(f"Completed {system_name} evaluation → results saved in {out_dir}")
