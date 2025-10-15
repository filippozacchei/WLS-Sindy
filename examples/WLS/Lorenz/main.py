"""
Evaluate SINDy model performance on Lorenz datasets across multiple
noise and fidelity configurations. Computes R² test score and
median absolute deviation (MAD) of coefficients across runs.

Outputs:
    ./Data/scores_summary.npz
    ./Data/mad_summary.npz
"""

# Imports
import numpy as np
import random
import os
import sys
import pysindy as ps
from tqdm import tqdm 

# Warnings
import warnings 
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
import pysindy as ps
from sklearn.metrics import r2_score
from pathlib import Path
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Lorenz dynamics (for test data)
# ---------------------------------------------------------------------
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def generate_test_data(dt=1e-3, t_end=10.0, n_traj=1, seed=999):
    rng = np.random.default_rng(seed)
    t = np.arange(0, t_end, dt)
    data, times = [], []
    for _ in range(n_traj):
        ic = rng.uniform([-10, -10, 20], [10, 10, 30])
        sol = solve_ivp(lorenz, (t[0], t[-1]), ic, t_eval=t, method="LSODA",
                        rtol=1e-10, atol=1e-12)
        data.append(sol.y.T)
        times.append(sol.t)
    return data, times


# ---------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------
def run_sindy(X_list, t_list, dt, degree=2, threshold=0.5, library=None, weights=None):
    """
    Train a Weak SINDy model using the given library and optimizer.
    Returns the trained model and its optimizer.
    """

    optimizer = ps.EnsembleOptimizer(
        ps.STLSQ(threshold=threshold),
        bagging=True,
        n_models=100,
    )
    # optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(feature_library=library, optimizer=optimizer)
    model.fit(X_list, t=dt, sample_weight=weights)

    return model, optimizer

def absolute_deviation(C_hat, C_true, tau=1e-3):
    return np.abs(C_hat - C_true) / np.maximum(np.abs(C_true), tau)

def ensemble_disagreement(optimizer):
    """
    Compute per-coefficient ensemble disagreement.
    Median(|coef_i - median(coef)|) over ensemble members.
    Returns: np.ndarray of shape (n_features, n_targets)
    """
    coefs = np.stack(optimizer.coef_list, axis=0)  # (n_models, n_features, n_targets)
    median_coef = np.median(coefs, axis=0)
    abs_diff = np.abs(coefs - median_coef)
    return np.median(abs_diff, axis=0)

# ---------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------
def evaluate_dataset(dataset_path="./Data/lorenz_dataset_trajectories_short.npz",
                     out_dir="./Scores",
                     dt=1e-3,
                     t_end=15.0,
                     degree=2,
                     threshold=0.1):
    Path(out_dir).mkdir(exist_ok=True)
    data = np.load(dataset_path, allow_pickle=True)
    dataset = data["dataset"].item()
    print(f"Loaded {len(dataset)} parameter combinations.")

    # Generate test data
    X_test_list, t_test_list = generate_test_data(dt=dt, t_end=t_end)
    print(f"Generated {len(X_test_list)} test trajectories.")

    grouped_r2  = {"hf": {}, "lf": {}, "mf": {}}
    grouped_mad = {"hf": {}, "lf": {}, "mf": {}}
    grouped_dis = {"hf": {}, "lf": {}, "mf": {}}

    # Ground-truth Lorenz coefficients (for MAD)
    C_true = np.zeros((9, 3))
    C_true[0, 0] = -10.0     # x
    C_true[1, 0] = 10.0      # y
    C_true[0, 1] = 28.0      # x
    C_true[5, 1] = -1.0      # xz
    C_true[1, 1] = -1.0      # y
    C_true[4, 2] = 1.0       # xy
    C_true[2, 2] = -8.0/3.0  # z
    deg=degree
    t=t_test_list[0]
    print(t.shape)
    K=100

    for key, entry in tqdm(dataset.items()):
        run, n_hf, n_lf, noise_hf, noise_lf = key
        
        config_key = (n_hf, n_lf, noise_hf, noise_lf)
        
        X_hf, t_hf, _ = entry["hf"]
        X_lf, t_lf, _ = entry["lf"]
        library = ps.feature_library.WeightedWeakPDELibrary(
            ps.PolynomialLibrary(degree=deg, include_bias=False),
            spatiotemporal_weights=np.ones_like(t_hf[0]),
            spatiotemporal_grid=t_hf[0],
            p=2, K=K
        )
        
        w_hf = [1.0/noise_hf**2] * len(X_hf)
        w_lf = [1.0/noise_lf**2] * len(X_lf)

        w_mf = w_hf + w_lf
        X_mf = X_hf + X_lf
        t_mf = t_hf + t_lf

        # --- Train weak-form SINDy models ---
        model_hf, opt_hf = run_sindy(X_hf, t_hf, dt, degree, threshold, library)
        model_lf, opt_lf = run_sindy(X_lf, t_lf, dt, degree, threshold, library)
        model_mf, opt_mf = run_sindy(X_mf, t_mf, dt, degree, threshold, library, w_mf)

        # --- Rebuild standard polynomial library for evaluation ---
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
        opt_eval_lf.coef_ = opt_mf.coef_
        opt_eval_mf.coef_ = opt_lf.coef_

        r2_hf = eval_model_hf.score(X_test_list, t=dt)
        print(r2_hf)
        r2_lf = eval_model_lf.score(X_test_list, t=dt)
        print(r2_lf)
        r2_mf = eval_model_mf.score(X_test_list, t=dt)
        print(r2_mf)

        # --- Coefficient median absolute deviation ---
        C_hf = opt_hf.coef_.T
        C_lf = opt_hf.coef_.T
        C_mf = opt_hf.coef_.T
        mad_hf = absolute_deviation(C_hf, C_true)
        mad_lf = absolute_deviation(C_lf, C_true)
        mad_mf = absolute_deviation(C_mf, C_true)
        dis_hf = ensemble_disagreement(opt_hf)
        dis_lf = ensemble_disagreement(opt_lf)
        dis_mf = ensemble_disagreement(opt_mf)

        # Append results
        for mode, r2, mad, dis in zip(
            ["hf", "lf", "mf"], 
            [r2_hf, r2_lf, r2_mf], 
            [mad_hf, mad_lf, mad_mf], 
            [dis_hf, dis_lf, dis_mf]
        ):
            grouped_r2[mode].setdefault(config_key, []).append(r2)
            grouped_mad[mode].setdefault(config_key, []).append(mad)
            grouped_dis[mode].setdefault(config_key, []).append(dis)

    scores_summary = {}
    mad_summary = {}
    disagreement_summary = {}


    for mode in ["hf", "lf", "mf"]:
        scores_summary[mode] = {
            cfg: np.median(vals) for cfg, vals in grouped_r2[mode].items()
        }
        mad_summary[mode] = {
            cfg: np.median(np.stack(vals), axis=0) for cfg, vals in grouped_mad[mode].items()
        }
        disagreement_summary[mode] = {
            cfg: np.median(np.stack(vals), axis=0) for cfg, vals in grouped_dis[mode].items()
        }

    # Save aggregated results
    np.savez(Path(out_dir) / "scores_summary.npz", scores=scores_summary)
    np.savez(Path(out_dir) / "mad_summary.npz", mads=mad_summary)
    np.savez(Path(out_dir) / "disagreement_summary.npz", disagreement=disagreement_summary)

    print(f"\n✅ Saved summary statistics → {out_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_dataset()
