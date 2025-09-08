from scipy.integrate import solve_ivp
from scipy.stats import qmc
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import numpy as np
from pysindy.utils import lorenz
import sys
sys.path.append("../../../src/")
from sindy import eSINDy, eWSINDy
from dataclasses import dataclass, field
from typing import List, Any
import pandas as pd
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

def lorenz_true_coefficients(sigma=10, rho=28, beta=8/3):
    # Feature order for PolynomialLibrary(degree=2): 
    # [1, x, y, z, x^2, xy, xz, y^2, yz, z^2]
    C_true = np.zeros((9, 3))
    
    # dx/dt = -sigma * x + sigma * y
    C_true[0, 0] = -sigma   # coefficient on x
    C_true[1, 0] = sigma    # coefficient on y

    # dy/dt = rho * x - xz - y
    C_true[0, 1] = rho      # coefficient on x
    C_true[5, 1] = -1       # coefficient on xz
    C_true[1, 1] = -1       # coefficient on y

    # dz/dt = xy - beta * z
    C_true[4, 2] = 1        # coefficient on xy
    C_true[2, 2] = -beta    # coefficient on z

    return C_true

def generate_data(dt, t_end, noise_level, n_trajectories, seed=1):
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    integrator_kwargs = {'method': 'LSODA', 'rtol': 1e-12, 'atol': 1e-12}
    t_vec = np.arange(0, t_end, dt)
    t_span = (t_vec[0], t_vec[-1])
    x0_samples = sampler.random(n=n_trajectories)
    x0_samples = qmc.scale(x0_samples, l_bounds=[-10]*3, u_bounds=[10]*3)
    x_train, t_train = [], []
    for i in range(n_trajectories):
        sol = solve_ivp(lorenz, t_span, x0_samples[i], t_eval=t_vec, **integrator_kwargs)
        traj = sol.y.T
        rmse = np.sqrt(mean_squared_error(traj, np.zeros(traj.shape)))
        noise = rng.normal(0.0, noise_level * rmse / 100, size=traj.shape)
        x_train.append(traj + noise)
        t_train.append(t_vec.copy())
    return x_train, t_train

def savgol(x, **kwargs):
    return np.column_stack([
        savgol_filter(x[:, j], **kwargs) for j in range(x.shape[1])
    ])
    
def mrad(C_hat, C_true, axis=None, tau=1e-3):
    rel_err = np.abs(C_hat - C_true) / np.maximum(np.abs(C_true), tau)
    return np.median(rel_err, axis=axis)

def run_esindy(x_train, 
               t_train, 
               weights, 
               n_hf, 
               n_ensemble, 
               library_functions, 
               smooth_columns,
               smoother_kws,
               treshold=0.5,
               alpha=1e-9,
               max_iter=20):
    model = eWSINDy(
        library_functions=library_functions,
        features=["x", "y", "z"],
        pde=False,
        win_length=51,
        stride=5, 
        #smoother=smooth_columns,
        #smoother_kws=smoother_kws
    )
    return model.fit(
        x_train, t_train,
        sample_weight=weights,
        stratified_ensemble=True if weights != 1.0 else False,
        n_hf=n_hf,
        threshold=treshold,
        sample_ensemble=True,
        n_ensembles=n_ensemble,
        alpha=alpha,
        max_iter=max_iter
    )

def run_models(data_dict, lf_noise, hf_noise, n_lf, n_hf, config,
               x_test, t_test, library_functions, smoother_kws,
               max_n_hf, max_n_lf, run_id, smoother=savgol):
    
    x_hf_full, t_hf_full = data_dict["hf"][hf_noise]
    x_lf_full, t_lf_full = data_dict["lf"][lf_noise]
    hf_offset, lf_offset = run_id * max_n_hf, run_id * max_n_lf

    x_train_hf, t_train_hf = x_hf_full[hf_offset:hf_offset+n_hf], t_hf_full[hf_offset:hf_offset+n_hf]
    x_train_lf, t_train_lf = x_lf_full[lf_offset:lf_offset+n_lf], t_lf_full[lf_offset:lf_offset+n_lf]

    model_hf = run_esindy(x_train_hf, t_train_hf, 1.0, n_hf, config["n_ensemble"], library_functions, smoother, smoother_kws)
    model_lf = run_esindy(x_train_lf, t_train_lf, 1.0, n_hf, config["n_ensemble"], library_functions, smoother, smoother_kws)

    x_train_mf, t_train_mf = x_train_hf + x_train_lf, t_train_hf + t_train_lf
    weights = [(1.0/hf_noise)**2]*n_hf + [(1.0/lf_noise)**2]*n_lf
    model_mf = run_esindy(x_train_mf, t_train_mf, weights, n_hf, config["n_ensemble"], library_functions, smoother, smoother_kws)

    C_true = lorenz_true_coefficients()
    return {
        "hf_score": model_hf.score(x_test, t_test),
        "lf_score": model_lf.score(x_test, t_test),
        "mf_score": model_mf.score(x_test, t_test),
        "hf_error": model_hf.mrad(C_true),
        "lf_error": model_lf.mrad(C_true),
        "mf_error": model_mf.mrad(C_true),
        "hf_disagreement": model_hf.mrad_disagreement(),
        "lf_disagreement": model_lf.mrad_disagreement(),
        "mf_disagreement": model_mf.mrad_disagreement(),
    }

def evaluate_score(data_configuration, ensemble_configuration, x_test, t_test, base_seed=None):
    """Evaluate HF, LF, MF models across parameter grid and return results as DataFrame."""
    param_grid = itertools.product(
        data_configuration["lf_noise"],
        data_configuration["hf_noise"],
        data_configuration["n_trajectories_lf"],
        data_configuration["n_trajectories_hf"],
    )

    print("Generating data \n", flush=True) 
    
    rng = np.random.default_rng(base_seed)
    max_n_hf, max_n_lf = max(data_configuration["n_trajectories_hf"]), max(data_configuration["n_trajectories_lf"])
    data_dict = {"hf": {}, "lf": {}}

    # --- Generate full datasets (once) ---
    for hf_noise in data_configuration["hf_noise"]:
        data_dict["hf"][hf_noise] = generate_data(
            data_configuration["dt_hf"], data_configuration["t_end_train_hf"],
            hf_noise, max_n_hf * ensemble_configuration["n_runs"],
            seed=rng.integers(0, 1_000_000),
        )

    for lf_noise in data_configuration["lf_noise"]:
        data_dict["lf"][lf_noise] = generate_data(
            data_configuration["dt_lf"], data_configuration["t_end_train_lf"],
            lf_noise, max_n_lf * ensemble_configuration["n_runs"],
            seed=rng.integers(0, 1_000_000),
        )
        
    print("Training and Evaluating models \n", flush=True)

    # --- Train & evaluate models ---
    results_list = []
    total_combinations = (
        len(data_configuration['lf_noise']) *
        len(data_configuration['hf_noise']) *
        len(data_configuration['n_trajectories_lf']) *
        len(data_configuration['n_trajectories_hf'])
    )
    for lf_noise, hf_noise, n_lf, n_hf in tqdm(param_grid, desc="Parameter combinations"):
        scores_list = []
        for run_id in tqdm(range(ensemble_configuration["n_runs"]), desc="Runs", leave=False):
            score = run_models(
                data_dict, lf_noise, hf_noise, n_lf, n_hf,
                ensemble_configuration, x_test, t_test,
                ensemble_configuration["library_functions"],
                ensemble_configuration["smoother_kws"],
                max_n_hf, max_n_lf, run_id
            )
            scores_list.append(score)
        
        results_list.append({
                "lf_noise": lf_noise,
                "hf_noise": hf_noise,
                "n_trajectories_lf": n_lf,
                "n_trajectories_hf": n_hf,
                "score_hf_mean": np.mean([s["hf_score"] for s in scores_list]),
                "score_lf_mean": np.mean([s["lf_score"] for s in scores_list]),
                "score_mf_mean": np.mean([s["mf_score"] for s in scores_list]),
                "score_hf_std": np.std([s["hf_score"] for s in scores_list]),
                "score_lf_std": np.std([s["lf_score"] for s in scores_list]),
                "score_mf_std": np.std([s["mf_score"] for s in scores_list]),
                "error_hf_mean": np.mean([s["hf_error"] for s in scores_list]),
                "error_lf_mean": np.mean([s["lf_error"] for s in scores_list]),
                "error_mf_mean": np.mean([s["mf_error"] for s in scores_list]),
                "error_hf_std": np.std([s["hf_error"] for s in scores_list]),
                "error_lf_std": np.std([s["lf_error"] for s in scores_list]),
                "error_mf_std": np.std([s["mf_error"] for s in scores_list]),
                "disagree_hf_mean": np.mean([s["hf_disagreement"] for s in scores_list]),
                "disagree_lf_mean": np.mean([s["lf_disagreement"] for s in scores_list]),
                "disagree_mf_mean": np.mean([s["mf_disagreement"] for s in scores_list]),
                "disagree_hf_std": np.std([s["hf_disagreement"] for s in scores_list]),
                "disagree_lf_std": np.std([s["lf_disagreement"] for s in scores_list]),
                "disagree_mf_std": np.std([s["mf_disagreement"] for s in scores_list]),
            })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv("model_scores_traj_extreme.csv", index=False)
    return results_df
