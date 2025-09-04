from scipy.integrate import solve_ivp
from scipy.stats import qmc
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import numpy as np
from pysindy.utils import lorenz
import sys
sys.path.append("../../../src/")
from sindy import eSINDy
from dataclasses import dataclass, field
from typing import List, Any
import pandas as pd
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed

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
               max_iter=50):
    model = eSINDy(
        library_functions=library_functions,
        features=["x", "y", "z"],
        smoother=smooth_columns,
        smoother_kws=smoother_kws
    )
    model = model.fit(
        x_train, t_train,
        sample_weight=weights,
        stratified_ensemble=bool(weights),
        n_hf=n_hf,
        threshold=treshold,
        sample_ensemble=True,
        n_ensembles=n_ensemble,
        alpha=alpha,
        max_iter=max_iter
    )
    return model

def run_models(x_train_hf, 
               x_train_lf,
               t_train_hf, 
               t_train_lf, 
               lf_noise, 
               hf_noise, 
               n_lf, 
               n_hf, 
               esindy_configuration, 
               x_test, 
               t_test,
               library_functions,
               smoother_kws):
    model_hf = run_esindy(x_train_hf,t_train_hf,1.0,n_hf,esindy_configuration['n_ensemble'],library_functions,savgol,smoother_kws)
    model_lf = run_esindy(x_train_lf,t_train_lf,1.0,n_hf,esindy_configuration['n_ensemble'],library_functions,savgol,smoother_kws)
    x_train_mf = x_train_hf + x_train_lf
    t_train_mf = t_train_hf + t_train_lf
    weights = [(1.0/hf_noise)**2]*n_hf + [(1.0/lf_noise)**2]*n_lf
    model_mf = run_esindy(x_train_mf,t_train_mf,weights,n_hf,esindy_configuration['n_ensemble'],library_functions,savgol,smoother_kws)
    hf_score = model_hf.score(x_test, t_test)
    lf_score = model_lf.score(x_test, t_test)
    mf_score = model_mf.score(x_test, t_test)
    return {'hf_score': hf_score, 'lf_score': lf_score, 'mf_score': mf_score}

def single_run(lf_noise, hf_noise, n_lf, n_hf, data_configuration, ensemble_configuration, x_test, t_test):
    x_train_hf, t_train_hf = generate_data(data_configuration['dt_hf'], data_configuration['t_end_train_hf'], hf_noise, n_hf)
    x_train_lf, t_train_lf = generate_data(data_configuration['dt_lf'], data_configuration['t_end_train_lf'], lf_noise, n_lf)
    scores = run_models(
        x_train_hf, x_train_lf,
        t_train_hf, t_train_lf,
        lf_noise, hf_noise, n_lf, n_hf,
        ensemble_configuration,
        x_test, t_test,
        ensemble_configuration['library_functions'],
        ensemble_configuration['smoother_kws']
    )
    return scores

def evaluate_score(data_configuration, ensemble_configuration, x_test, t_test):
    param_grid = list(itertools.product(
        data_configuration['lf_noise'],
        data_configuration['hf_noise'],
        data_configuration['n_trajectories_lf'],
        data_configuration['n_trajectories_hf']
    ))

    results_list = []
    for lf_noise, hf_noise, n_lf, n_hf in tqdm(param_grid, desc='Parameter combinations'):
        # Parallelize the runs
        scores_list = Parallel(n_jobs=-1)(
            delayed(single_run)(
                lf_noise, hf_noise, n_lf, n_hf,
                data_configuration, ensemble_configuration, x_test, t_test
            ) for _ in range(ensemble_configuration['n_runs'])
        )
        hf_scores = [s['hf_score'] for s in scores_list]
        lf_scores = [s['lf_score'] for s in scores_list]
        mf_scores = [s['mf_score'] for s in scores_list]
        results_list.append({
            'lf_noise': lf_noise,
            'hf_noise': hf_noise,
            'n_trajectories_lf': n_lf,
            'n_trajectories_hf': n_hf,
            'score_hf_mean': np.mean(hf_scores),
            'score_lf_mean': np.mean(lf_scores),
            'score_mf_mean': np.mean(mf_scores),
            'score_hf_std': np.std(hf_scores),
            'score_lf_std': np.std(lf_scores),
            'score_mf_std': np.std(mf_scores)
        })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('model_scores.csv', index=False)
    return results_df