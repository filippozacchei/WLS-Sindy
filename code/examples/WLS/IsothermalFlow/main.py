import sys
sys.path.append("../../../")
from pathlib import Path
from utils.part1 import evaluate_mf_sindy
from generator import generate_compressible_flow
import numpy as np

# We need this signature
# X_hf, grid_hf, t_hf = generator(n_hf, 
#                                 noise_level=noise_level_hf * std_per_dim, 
#                                 T=T, 
#                                 seed=run*seed)
# def generate_compressible_flow(
#     n_traj=1,
#     N=64,
#     Nt=200,
#     L=0.1,
#     T=2,
#     mu=1.0,
#     RT=1.0,
#     noise_level=0.0,
#     seed=42,
#     initial_condition="taylor-green",
#     noise_0 = 0.01
# )

    
if __name__ == "__main__":
    system_name = "lorenz"
    out_dir = "./Results"

    # Define grid and parameters
    n_lf_vals = np.arange(10, 101, 10)
    n_hf_vals = np.arange(1, 11, 1)
    runs = 1
    dt = 1e-3
    threshold = 0.5
    degree = 2

    # Ground-truth Lorenz model coefficients (for MAD)
    C_true = np.zeros((9, 3))
    C_true[0, 0] = -10.0     # dx/dt = -10x + 10y
    C_true[1, 0] = 10.0
    C_true[0, 1] = 28.0      # dy/dt = 28x - xz - y
    C_true[5, 1] = -1.0
    C_true[1, 1] = -1.0
    C_true[4, 2] = 1.0       # dz/dt = xy - (8/3)z
    C_true[2, 2] = -8.0 / 3.0

    # Run the unified evaluation routine
    evaluate_mf_sindy(
        generator=generate_compressible_flow,
        system_name=system_name,
        n_lf_vals=n_lf_vals,
        n_hf_vals=n_hf_vals,
        noise_level_hf=0.01,
        noise_level_lf=0.25,
        runs=runs,
        dt=dt,
        threshold=threshold,
        degree=degree,
        out_dir=out_dir,
        seed=231,
        T=0.1,
        T_test=1
    )
