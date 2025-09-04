# %% IMPORTS
 
import matplotlib.pyplot as plt 
from utils import *
import sys
import numpy as np
sys.path.append("../../../src")
from sindy import eSINDy
import pysindy as ps

# %% DATA CONFIGURATION PARAMETERS

data_configuration = {
    'dt_lf': 0.001,
    'dt_hf': 0.001, 
    'lf_noise': np.linspace(1.0,50.0,5,dtype=np.float32,endpoint=True),
    'hf_noise': np.linspace(1.0,10.0,2,dtype=np.float32,endpoint=True),
    'n_trajectories_lf': np.arange(5,105,5),
    'n_trajectories_hf': np.arange(1,21,20),
    't_end_train_hf': 0.1,
    't_end_train_lf': 0.1,
    't_end_test': 15,
    'n_trajectories_test': 2,
}
 
# %% ESINDY CONFIGURATION PARAMETERS

esindy_configuration = {
    'n_ensemble': 100,
    'n_runs': 10,
    'library_functions': ps.PolynomialLibrary(degree=2,include_bias=False),
    'smoother_kws': {
        'window_length': 51,
        'polyorder': 3
    }
}

# %% PLOT PARAMETERS

case       = 'Figures/name_plot'
test_title = ''
x_label    = 'x_label'
y_label    = 'y_label'
x_vector   = data_configuration['n_trajectories_lf']

# %% SMOOTHER PARAMETERS



# %% GENERATE TEST DATA

x_test, t_test = generate_data(dt=data_configuration['dt_hf'],
                               t_end=data_configuration['t_end_test'],
                               noise_level=0.0,
                               n_trajectories=data_configuration['n_trajectories_test'],
                               seed = 97687689767986)


# %% RUN AND STORE MODEL SCORES USING PANDAS

results_df = evaluate_score(data_configuration, esindy_configuration, x_test, t_test)
