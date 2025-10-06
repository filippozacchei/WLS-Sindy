# run_experiment.py

import sys
sys.path.append("../../../src")  # if needed for your local sindy package

import numpy as np
import pysindy as ps

from utils import (
    DataConfig, ESINDyConfig, generate_data, evaluate_score
)

# ---------------- Data configuration ----------------

data_configuration = DataConfig(
    dt_lf=0.001,
    dt_hf=0.001,
    lf_noise=np.array([50]),
    hf_noise=np.array([10]),
    n_trajectories_lf=np.array([50]),
    n_trajectories_hf=np.array([10]),
    t_end_train_hf=0.1,
    t_end_train_lf=0.1,
    t_end_test=5.0,
    n_trajectories_test=1,
)

# ---------------- eSINDy configuration --------------

esindy_configuration = ESINDyConfig(
    n_ensembles=100,
    n_runs=100,
    library_functions=ps.PolynomialLibrary(degree=2, include_bias=False),
    smoother_kws={"window_length": 51, "polyorder": 3},
)

# ---------------- Test data -------------------------

x_test, t_test = generate_data(
    dt=data_configuration.dt_hf,
    t_end=data_configuration.t_end_test,
    noise_level=0.0,
    n_trajectories=data_configuration.n_trajectories_test,
    seed=97687689767986
)

# ---------------- Evaluate & save -------------------

results_df = evaluate_score(
    data_cfg=data_configuration,
    es_cfg=esindy_configuration,
    x_test=x_test,
    t_test=t_test,
    base_seed=1,
    csv_path="model_scores_traj_extreme.csv"  # or None to skip writing
)

print(results_df.head())
