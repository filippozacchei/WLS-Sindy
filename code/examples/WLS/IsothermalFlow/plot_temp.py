"""
plot_mf_vs_hf_weak.py

Publication-ready demonstration of Weak SINDy for
multi-fidelity learning on the 2D isothermal compressible
Navier–Stokes system.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pysindy as ps
from pathlib import Path
from generator import generate_compressible_flow

# ---------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------
# Utility metrics
# ---------------------------------------------------------------------
def mad(a, b):
    """Median absolute deviation between coefficient arrays."""
    a, b = np.ravel(a), np.ravel(b)
    return np.median(np.abs(a - b))

def disagreement(optimizer):
    """Median absolute ensemble disagreement."""
    if not hasattr(optimizer, "coef_list"):
        return np.nan
    coefs = np.stack(optimizer.coef_list, axis=0)
    med = np.median(coefs, axis=0)
    return np.median(np.abs(coefs - med))

# ---------------------------------------------------------------------
# Generate HF / LF / MF data
# ---------------------------------------------------------------------
print("Generating compressible flow data...")
N = 64
Nt = 100
L = 5.0
T = 0.5

n_hf = 1
n_lf = 2

u_hf, grid, t_arr, L, T = generate_compressible_flow(
    n_traj=n_hf, N=N, Nt=Nt, T=T, L=L, noise_level=0.0, seed=0
)
u_lf, _, _, _, _ = generate_compressible_flow(
    n_traj=n_lf, N=N, Nt=Nt, T=T, L=L, noise_level=0.1, seed=1
)

u_mf = u_hf + u_lf
w_mf = [100.0] * n_hf + [10.0] * n_lf  # HF > LF weighting

print(f"HF trajectories: {len(u_hf)}, LF trajectories: {len(u_lf)}")

# ---------------------------------------------------------------------
# Weak SINDy setup
# ---------------------------------------------------------------------
library_functions = [
    lambda x: x,
    lambda x: x**2,
    lambda x: 1 / (1e-6 + np.abs(x))
]
library_function_names = [
    lambda x: x,
    lambda x: x + "²",
    lambda x: x + "⁻¹"
]

custom_library = ps.CustomLibrary(
    library_functions=library_functions,
    function_names=library_function_names
)

# Build weak PDE library using your custom features
lib = ps.WeakPDELibrary(
    custom_library,
    spatiotemporal_grid=grid,
    derivative_order=2,
    K=2000,                         # number of test functions
    H_xt=[L/10, L/10, T/10],
)

opt = lambda thr: ps.EnsembleOptimizer(
    ps.STLSQ(threshold=thr),
    bagging=True,
    n_models=20,
)

models = {
    "HF": ps.SINDy(feature_library=lib, optimizer=opt(0.5)),
    "LF": ps.SINDy(feature_library=lib, optimizer=opt(0.5)),
    "MF": ps.SINDy(feature_library=lib, optimizer=opt(0.5)),
}

# ---------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------
print("Training HF...")
models["HF"].fit(u_hf, t=t_arr)
print("Training LF...")
models["LF"].fit(u_lf, t=t_arr)
print("Training MF...")
models["MF"].fit(u_mf, t=t_arr, sample_weight=w_mf)

# ---------------------------------------------------------------------
# Extract coefficients
# ---------------------------------------------------------------------
C_hf = np.ravel(models["HF"].optimizer.coef_)
C_lf = np.ravel(models["LF"].optimizer.coef_)
C_mf = np.ravel(models["MF"].optimizer.coef_)

mad_lf = mad(C_lf, C_hf)
mad_mf = mad(C_mf, C_hf)

dis_lf = disagreement(models["LF"].optimizer)
dis_mf = disagreement(models["MF"].optimizer)
dis_hf = disagreement(models["HF"].optimizer)

# ---------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------
out_dir = Path("./Figures_Weak_MF_vs_HF")
out_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1️⃣ Coefficient comparison
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 2.8))
ind = np.arange(len(C_hf))
width = 0.25
ax.bar(ind - width, C_lf, width, label="LF", color="lightgray", edgecolor="black")
ax.bar(ind, C_hf, width, label="HF", color="steelblue", edgecolor="black")
ax.bar(ind + width, C_mf, width, label="MF", color="tomato", edgecolor="black")
ax.set_xlabel("Term index")
ax.set_ylabel("Coefficient value")
ax.set_title("Weak-SINDy Coefficients")
ax.legend(frameon=False, loc="best")
plt.tight_layout()
plt.savefig(out_dir / "weak_coeffs.png", dpi=600, transparent=True)
plt.close(fig)

# ---------------------------------------------------------------------
# 2️⃣ MAD comparison
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.bar(["LF", "MF"], [mad_lf, mad_mf],
       color=["lightgray", "tomato"], edgecolor="black")
ax.set_ylabel("Median Abs. Deviation (vs HF)")
ax.set_title("Coefficient Deviation")
plt.tight_layout()
plt.savefig(out_dir / "weak_mad.png", dpi=600, transparent=True)
plt.close(fig)

# ---------------------------------------------------------------------
# 3️⃣ Ensemble disagreement
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.6))
ax.bar(["LF", "HF", "MF"], [dis_lf, dis_hf, dis_mf],
       color=["lightgray", "steelblue", "tomato"], edgecolor="black")
ax.set_ylabel("Median Ensemble Disagreement")
ax.set_title("Model Stability (Ensemble Spread)")
plt.tight_layout()
plt.savefig(out_dir / "weak_disagreement.png", dpi=600, transparent=True)
plt.close(fig)

# ---------------------------------------------------------------------
print("✅ Weak-SINDy results saved to", out_dir)
print(f"MAD(LF,HF) = {mad_lf:.3e}, MAD(MF,HF) = {mad_mf:.3e}")
print(f"Disagreement (LF,HF,MF) = {dis_lf:.3e}, {dis_hf:.3e}, {dis_mf:.3e}")
