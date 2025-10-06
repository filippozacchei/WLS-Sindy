# mf_esindy_lorenz.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Any, Optional, Callable

import itertools
import logging
import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error

from pysindy.utils import lorenz

# Make sure your path & imports work in your environment.
# If this file sits next to your local SINDy implementations, prefer a relative import.
# Otherwise, keep the path append in your runner script (not in the library).
import sys
sys.path.append('../../../src/')
from sindy import eSINDy, eWSINDy  # noqa: F401  # used in run_esindy

logger = logging.getLogger(__name__)


# ----------------------------- Data classes ----------------------------------

@dataclass(frozen=True)
class DataConfig:
    dt_lf: float
    dt_hf: float
    lf_noise: np.ndarray
    hf_noise: np.ndarray
    n_trajectories_lf: np.ndarray
    n_trajectories_hf: np.ndarray
    t_end_train_hf: float
    t_end_train_lf: float
    t_end_test: float
    n_trajectories_test: int


@dataclass(frozen=True)
class ESINDyConfig:
    n_ensembles: int
    n_runs: int
    library_functions: Any  # e.g., ps.PolynomialLibrary(degree=2, include_bias=False)
    smoother_kws: Dict[str, Any] = None  # kept for compatibility if needed elsewhere


# ----------------------------- Utilities -------------------------------------

def lorenz_true_coefficients(sigma: float = 10.0,
                             rho: float = 28.0,
                             beta: float = 8.0 / 3.0) -> np.ndarray:
    """
    Ground-truth sparse coefficient matrix for the Lorenz system under
    a polynomial library of degree 2 with include_bias=False.

    Feature order (degree=2, include_bias=False) is:
      [x, y, z, x^2, x y, x z, y^2, y z, z^2]  -> 9 features

    Returns
    -------
    C_true : (9, 3) ndarray
        Columns correspond to dx/dt, dy/dt, dz/dt in that order.
    """
    C_true = np.zeros((9, 3))

    # dx/dt = -sigma*x + sigma*y
    C_true[0, 0] = -sigma  # x
    C_true[1, 0] = sigma   # y

    # dy/dt = rho*x - x*z - y
    C_true[0, 1] = rho     # x
    C_true[5, 1] = -1.0    # xz
    C_true[1, 1] = -1.0    # y

    # dz/dt = x*y - beta*z
    C_true[4, 2] = 1.0     # xy
    C_true[2, 2] = -beta   # z

    return C_true

def mrad(C_hat: np.ndarray, C_true: np.ndarray, axis: Optional[int] = None, tau: float = 1e-3) -> np.ndarray:
    """
    Median relative absolute deviation between estimated and true coefficients.

    rel_err = |C_hat - C_true| / max(|C_true|, tau)
    """
    rel_err = np.abs(C_hat - C_true) / np.maximum(np.abs(C_true), tau)
    return np.median(rel_err, axis=axis)


# --------------------------- Data generation ----------------------------------

def generate_data(dt: float,
                  t_end: float,
                  noise_level: float,
                  n_trajectories: int,
                  seed: Optional[int] = 1) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate noisy Lorenz trajectories.

    Parameters
    ----------
    dt : float
    t_end : float
    noise_level : float
        Scalar noise level (interpreted as percentage of RMSE of the clean signal).
    n_trajectories : int
    seed : int, optional

    Returns
    -------
    x_list : list of (T, 3) arrays
    t_list : list of (T,) arrays
    """
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    integrator_kwargs = {'method': 'LSODA', 'rtol': 1e-12, 'atol': 1e-12}

    t_vec = np.arange(0, t_end, dt)
    t_span = (float(t_vec[0]), float(t_vec[-1]))

    # Sample initial conditions in a reasonable Lorenz box
    x0_samples = sampler.random(n=n_trajectories)
    x0_samples = qmc.scale(x0_samples, l_bounds=[-10, -10, 20], u_bounds=[10, 10, 30])

    x_list, t_list = [], []
    for i in range(n_trajectories):
        sol = solve_ivp(lorenz, t_span, x0_samples[i], t_eval=t_vec, **integrator_kwargs)
        traj = sol.y.T
        rmse = np.sqrt(mean_squared_error(traj, np.zeros_like(traj)))
        noise_std = (noise_level * rmse) / 100.0
        noise = rng.normal(0.0, noise_std, size=traj.shape)
        x_list.append(traj + noise)
        t_list.append(t_vec.copy())
    return x_list, t_list


# --------------------------- Model training -----------------------------------

def run_esindy(x_train: Sequence[np.ndarray],
               t_train: Sequence[np.ndarray],
               sample_weight: Sequence[float] | float,
               n_hf: int,
               n_ensembles: int,
               library_functions: Any,
               threshold: float = 0.5,
               alpha: float = 1e-9,
               max_iter: int = 20):
    """
    Train eWSINDy with the given data and options.
    """
    model = eWSINDy(
        library_functions=library_functions,
        features=["x", "y", "z"],
        pde=False,
        win_length=51,
        stride=5
    )
    return model.fit(
        x_train, t_train,
        sample_weight=sample_weight,
        stratified_ensemble=(sample_weight != 1.0),
        n_hf=n_hf,
        threshold=threshold,
        sample_ensemble=True,
        n_ensembles=n_ensembles,
        alpha=alpha,
        max_iter=max_iter
    )


def _subset_for_run(x_all: List[np.ndarray],
                    t_all: List[np.ndarray],
                    start: int,
                    count: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    return x_all[start:start + count], t_all[start:start + count]


def run_models(data_dict: Dict[str, Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]]],
               lf_noise: int,
               hf_noise: int,
               n_lf: int,
               n_hf: int,
               es_config: ESINDyConfig,
               x_test: Sequence[np.ndarray],
               t_test: Sequence[np.ndarray],
               library_functions: Any,
               max_n_hf: int,
               max_n_lf: int,
               run_id: int) -> Dict[str, float]:
    """
    Train HF, LF, and MF models for one parameter setting and one run_id, then score.

    In this setup:
      - Each run regenerates its own HF/LF pools (fresh data each run).
      - We do *not* use offsets into a giant pool.
      - Ensembles are stratified between HF and LF for better uncertainty quantification.
    """
    # Pre-generated HF/LF pools (each run is fresh)
    x_hf_full, t_hf_full = data_dict["hf"][hf_noise]
    x_lf_full, t_lf_full = data_dict["lf"][lf_noise]

    # Just take first n_hf and n_lf (no offsets needed since pools are regenerated each run)
    x_train_hf, t_train_hf = _subset_for_run(x_hf_full, t_hf_full, 0, n_hf)
    x_train_lf, t_train_lf = _subset_for_run(x_lf_full, t_lf_full, 0, n_lf)

    # Train HF model (ensembles with stratified sampling)
    model_hf = run_esindy(
        x_train_hf, t_train_hf,
        sample_weight=1.0,
        n_hf=n_hf,
        n_ensembles=es_config.n_ensembles,
        library_functions=library_functions,
        threshold=0.5,
        alpha=1e-9,
        max_iter=20
    )

    # Train LF model (ensembles with stratified sampling)
    model_lf = run_esindy(
        x_train_lf, t_train_lf,
        sample_weight=1.0,
        n_hf=0,   # no HF trajectories in LF-only case
        n_ensembles=es_config.n_ensembles,
        library_functions=library_functions,
        threshold=0.5,
        alpha=1e-9,
        max_iter=20
    )

    # Train MF model (with weights and stratified sampling)
    x_train_mf = x_train_hf + x_train_lf
    t_train_mf = t_train_hf + t_train_lf
    weights = ([float((1.0 / hf_noise) ** 2)] * n_hf) + \
              ([float((1.0 / lf_noise) ** 2)] * n_lf)

    model_mf = run_esindy(
        x_train_mf, t_train_mf,
        sample_weight=weights,
        n_hf=n_hf,
        n_ensembles=es_config.n_ensembles,
        library_functions=library_functions,
        threshold=0.5,
        alpha=1e-9,
        max_iter=20
    )

    # Ground truth coefficients
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
        "coef_lf_list": np.array(model_lf.coef_list),
        "coef_hf_list": np.array(model_hf.coef_list),
        "coef_mf_list": np.array(model_mf.coef_list),
        "coef_lf_median": model_lf.coef_median,
        "coef_hf_median": model_hf.coef_median,
        "coef_mf_median": model_mf.coef_median,
    }
# -------------------------- Experiment orchestration --------------------------

def generate_all_data(data_cfg: DataConfig,
                      es_cfg: ESINDyConfig,
                      base_seed: int) -> Tuple[Dict[str, Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]]],
                                              int, int]:
    """
    Pre-generate the full pools of HF and LF trajectories for all noise levels and runs.
    """
    rng = np.random.seed(base_seed)
    max_n_hf = int(np.max(data_cfg.n_trajectories_hf))
    max_n_lf = int(np.max(data_cfg.n_trajectories_lf))

    data_dict: Dict[str, Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]]] = {"hf": {}, "lf": {}}

    # HF pools
    for hf_noise in data_cfg.hf_noise:
        data_dict["hf"][int(hf_noise)] = generate_data(
            dt=data_cfg.dt_hf,
            t_end=data_cfg.t_end_train_hf,
            noise_level=float(hf_noise),
            n_trajectories=max_n_hf * es_cfg.n_runs,
            seed=base_seed,
        )

    # LF pools
    for lf_noise in data_cfg.lf_noise:
        data_dict["lf"][int(lf_noise)] = generate_data(
            dt=data_cfg.dt_lf,
            t_end=data_cfg.t_end_train_lf,
            noise_level=float(lf_noise),
            n_trajectories=max_n_lf * es_cfg.n_runs,
            seed=base_seed*10000000,
        )

    return data_dict, max_n_hf, max_n_lf


def evaluate_score(data_cfg: DataConfig,
                   es_cfg: ESINDyConfig,
                   x_test: Sequence[np.ndarray],
                   t_test: Sequence[np.ndarray],
                   base_seed: int = 1,
                   csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Evaluate HF, LF, and MF models across a grid of (lf_noise, hf_noise, n_lf, n_hf).
    Returns a DataFrame of aggregated metrics; optionally writes CSV.
    """
    logger.info("Generating data pools.")
    data_dict, max_n_hf, max_n_lf = generate_all_data(data_cfg, es_cfg, base_seed)

    # Parameter grid
    param_grid = itertools.product(
        map(int, data_cfg.lf_noise),
        map(int, data_cfg.hf_noise),
        map(int, data_cfg.n_trajectories_lf),
        map(int, data_cfg.n_trajectories_hf),
    )

    results_list: List[Dict[str, float]] = []

    for lf_noise, hf_noise, n_lf, n_hf in param_grid:
        scores_one_setting: List[Dict[str, float]] = []
        for run_id in range(es_cfg.n_runs):
            out = run_models(
                data_dict=data_dict,
                lf_noise=lf_noise,
                hf_noise=hf_noise,
                n_lf=n_lf,
                n_hf=n_hf,
                es_config=es_cfg,
                x_test=x_test,
                t_test=t_test,
                library_functions=es_cfg.library_functions,
                max_n_hf=max_n_hf,
                max_n_lf=max_n_lf,
                run_id=run_id
            )
            scores_one_setting.append(out)

        # Aggregate
        def mean_of(key: str) -> float:
            return float(np.mean([s[key] for s in scores_one_setting]))

        def std_of(key: str) -> float:
            return float(np.std([s[key] for s in scores_one_setting], ddof=0))

        results_list.append({
            "lf_noise": lf_noise,
            "hf_noise": hf_noise,
            "n_trajectories_lf": n_lf,
            "n_trajectories_hf": n_hf,
            "score_hf_mean": mean_of("hf_score"),
            "score_lf_mean": mean_of("lf_score"),
            "score_mf_mean": mean_of("mf_score"),
            "score_hf_std": std_of("hf_score"),
            "score_lf_std": std_of("lf_score"),
            "score_mf_std": std_of("mf_score"),
            "error_hf_mean": mean_of("hf_error"),
            "error_lf_mean": mean_of("lf_error"),
            "error_mf_mean": mean_of("mf_error"),
            "error_hf_std": std_of("hf_error"),
            "error_lf_std": std_of("lf_error"),
            "error_mf_std": std_of("mf_error"),
            "disagree_hf_mean": mean_of("hf_disagreement"),
            "disagree_lf_mean": mean_of("lf_disagreement"),
            "disagree_mf_mean": mean_of("mf_disagreement"),
            "disagree_hf_std": std_of("hf_disagreement"),
            "disagree_lf_std": std_of("lf_disagreement"),
            "disagree_mf_std": std_of("mf_disagreement"),
        })

    results_df = pd.DataFrame(results_list)
    if csv_path:
        results_df.to_csv(csv_path, index=False)
    return results_df
