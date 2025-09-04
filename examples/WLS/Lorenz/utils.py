from scipy.integrate import solve_ivp
from scipy.stats import qmc
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import numpy as np
from pysindy.utils import lorenz


def generate_data(dt, t_end_train, noise_level, n_trajectories, seed=1):
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    integrator_kwargs = {'method': 'LSODA', 'rtol': 1e-12, 'atol': 1e-12}
    t_vec = np.arange(0, t_end_train, dt)
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

def smooth_columns(x, **kwargs):
    return np.column_stack([
        savgol_filter(x[:, j], **kwargs) for j in range(x.shape[1])
    ])
    
    