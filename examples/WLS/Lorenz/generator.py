"""
Generate Lorenz system trajectories for sparse system identification.

Each dataset contains multiple trajectories of the Lorenz system 
with different initial conditions and controlled noise levels.

Output files:
    ./data/lorenz_dataset.npz
    ./data/metadata.json
"""

import os
import json
import itertools
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ---------------------------------------------------------------------
# Dynamical system definition
# ---------------------------------------------------------------------
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Standard Lorenz system of equations.

    Parameters
    ----------
    t : float
        Time variable (unused but required by `solve_ivp`).
    state : ndarray, shape (3,)
        Current state vector [x, y, z].
    sigma : float, optional
        Prandtl number.
    rho : float, optional
        Rayleigh number.
    beta : float, optional
        Geometric factor.

    Returns
    -------
    dydt : ndarray, shape (3,)
        Time derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


# ---------------------------------------------------------------------
# Simulation routine
# ---------------------------------------------------------------------
def simulate(y0, T=10.0, dt=1e-3, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Integrate the Lorenz system using a stiff solver (LSODA).

    Parameters
    ----------
    y0 : ndarray, shape (3,)
        Initial condition [x0, y0, z0].
    T : float, optional
        Final simulation time.
    dt : float, optional
        Time step for saved data.
    sigma, rho, beta : float
        Lorenz system parameters.

    Returns
    -------
    t : ndarray, shape (n_steps,)
        Time vector.
    Y : ndarray, shape (n_steps, 3)
        Simulated trajectory.
    """
    t = np.arange(0, T, dt)
    sol = solve_ivp(lorenz, (t[0], t[-1]), y0, t_eval=t,
                    args=(sigma, rho, beta),
                    method="LSODA", rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------
def generate_dataset(
    n_hf_list: List[int],
    n_lf_list: List[int],
    noise_hf_list: List[float],
    noise_lf_list: List[float],
    n_runs: int,
    T: float = 10.0,
    dt: float = 1e-3,
    seed: int = 42,
    save_data: bool = True,
    out_path: str = "./data/lorenz_dataset.npz",
):
    """
    Generate Lorenz datasets for all combinations of
    (n_hf, n_lf, noise_hf, noise_lf) across multiple runs.

    HF and LF trajectories have *different initial conditions*.

    Returns
    -------
    dataset : dict
        Nested dictionary indexed as:
            dataset[(run, n_hf, n_lf, noise_hf, noise_lf)] = {
                "hf": (x_list_hf, t_list_hf, ic_hf),
                "lf": (x_list_lf, t_list_lf, ic_lf),
            }
    """
    rng = np.random.default_rng(seed)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    dataset: Dict[Tuple[int, int, int, float, float], Dict] = {}
    # Create reusable parameter combinations
    param_grid = list(itertools.product(n_hf_list, n_lf_list, noise_hf_list, noise_lf_list))

    t, x1 = simulate([-10,-10,25], 15, 1e-3)
    rmse = np.sqrt(mean_squared_error(x1, np.zeros_like(x1)))
    print(rmse)

    for run in range(n_runs):
        print(f"\n[Run {run+1}/{n_runs}] Generating independent ICs for HF/LF pools.")

        for n_hf, n_lf, noise_hf, noise_lf in tqdm(param_grid):
            key = (run, n_hf, n_lf, noise_hf, noise_lf)
            # print(f" → Config HF={n_hf}, LF={n_lf}, σ_HF={noise_hf}%, σ_LF={noise_lf}%")

            # Independent ICs for HF and LF
            sampler_hf = qmc.LatinHypercube(d=3, seed=seed + run * 1000 + int(noise_hf))
            sampler_lf = qmc.LatinHypercube(d=3, seed=seed * 10 + run * 1000 + int(noise_lf))

            ic_hf = qmc.scale(sampler_hf.random(n=n_hf), [-10, -10, 20], [10, 10, 30])
            ic_lf = qmc.scale(sampler_lf.random(n=n_lf), [-10, -10, 20], [10, 10, 30])

            # Generate HF trajectories
            x_list_hf, t_list_hf = [], []
            for i, y0 in enumerate(ic_hf):
                np.random.seed(seed + run + i)
                t, x_noisy = simulate(y0, T, dt)
                noise_std = (noise_hf * rmse) / 100.0
                x_noisy += rng.normal(0.0, noise_std, size=x_noisy.shape)                

                x_list_hf.append(x_noisy)
                t_list_hf.append(t)

            # Generate LF trajectories
            x_list_lf, t_list_lf = [], []
            for j, y0 in enumerate(ic_lf):
                np.random.seed(seed * 100 + run * 10000 + j)

                t, x_noisy = simulate(y0, T, dt)
                noise_std = (noise_lf * rmse) / 100.0
                x_noisy += rng.normal(0.0, noise_std, size=x_noisy.shape)                

                x_list_lf.append(x_noisy)
                t_list_lf.append(t)

            dataset[key] = dict(
                hf=(x_list_hf, t_list_hf, ic_hf),
                lf=(x_list_lf, t_list_lf, ic_lf),
            )

    if save_data:
        np.savez(out_path, dataset=dataset)
        print(f"\n Saved Lorenz dataset → {out_path}")

        meta = dict(
            n_runs=n_runs,
            n_hf_list=n_hf_list,
            n_lf_list=n_lf_list,
            noise_hf_list=noise_hf_list,
            noise_lf_list=noise_lf_list,
            T=T,
            dt=dt,
            seed=seed,
        )
        with open(Path(out_path).with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=4)
        print(f"Saved metadata → {Path(out_path).with_suffix('.json')}")

    return dataset


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    Main entry point for Lorenz dataset generation.
    Configures the parameter sweep and saves results.
    """
    # ---------------- Configuration ----------------
    output_dir = Path("./Data")
    output_dir.mkdir(exist_ok=True)

    config = dict(
        n_hf_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_lf_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        noise_hf_list=[1.0, 5.0],
        noise_lf_list=[10.0, 25.0, 50.0],
        n_runs=5,
        T=0.1,
        dt=1e-3,
        seed=42,
        out_path=str(output_dir / "lorenz_dataset_trajectories_short.npz"),
    )

    print("\n=== Lorenz Dataset Generation ===")
    print(json.dumps(config, indent=4))

    # ---------------- Generate Dataset ----------------
    dataset = generate_dataset(**config)

    # ---------------- Save Metadata ----------------
    metadata = {
        "description": "Lorenz system trajectories for SINDy experiments",
        "system": "Lorenz",
        "parameters": config,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✅ Dataset and metadata saved successfully.")
    print(f"→ Dataset:   {output_dir / 'lorenz_dataset_all.npz'}")
    print(f"→ Metadata:  {output_dir / 'metadata.json'}")
