import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import sys
sys.path.append("../examples/WLS/")  # Adjust path as needed to import your modules
sys.path.append("../src/")  # Adjust path as needed to import your modules
# Import your Lorenz and Vorticity modules
from Lorenz import utils as lorenz_mod
from navier_stokes import utils as vort_mod

# -------------------
# 1. Generate Lorenz Data
# -------------------
x_list, t_list = lorenz_mod.generate_data(
    dt=0.01, t_end=5.0, noise_level=10, n_trajectories=1
)
lorenz_traj = x_list[0]
lorenz_t = t_list[0]

# -------------------
# 2. Generate Vorticity Data
# -------------------
dataset = vort_mod.generate_dataset(N=64, L=2*np.pi, T=2.0, Nt=60, ic_types=["vortex_pair"])
omega_clean = dataset["omega_list"][0]
rng = np.random.default_rng(1)
omega_noisy = vort_mod.add_structured_noise(omega_clean, rng, sigma_add=0.1)

# -------------------
# 3. Composite Figure Layout
# -------------------
fig = plt.figure(figsize=(16, 10))

# --- Panel 1: Noisy data ---
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax1.plot(lorenz_traj[:,0], lorenz_traj[:,1], lw=1.5, color="tab:blue", alpha=0.7)
ax1.set_title("Lorenz trajectories (noisy)")
ax1.set_xlabel("x"); ax1.set_ylabel("y")

ax2 = plt.subplot2grid((2, 3), (1, 0))
im = ax2.imshow(omega_noisy[:,:,0], origin="lower", cmap="plasma")
plt.colorbar(im, ax=ax2, shrink=0.7)
ax2.set_title("Navier–Stokes vorticity (noisy snapshot)")
ax2.axis("off")

# --- Panel 2: Workflow schematic ---
ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2)
ax3.axis("off")
steps = [
    "Library of candidate terms",
    "Sparse regression on bootstraps",
    "Ensemble of SINDy models",
    "Thresholded & aggregated model"
]
y_positions = np.linspace(0.85, 0.25, len(steps))
for i, (step, y) in enumerate(zip(steps, y_positions)):
    ax3.text(0.1, y, step,
             fontsize=12, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.6))
    if i < len(steps)-1:
        ax3.add_patch(FancyArrowPatch((0.15, y-0.07), (0.15, y_positions[i+1]+0.05),
                                      arrowstyle="->", mutation_scale=15, color="k"))

ax3.set_title("Ensemble SINDy Workflow", fontsize=14, weight="bold")

# --- Panel 3: Outputs & uncertainty ---
ax4 = plt.subplot2grid((2, 3), (0, 2))
# Simulated uncertainty bands for Lorenz x(t)
mean_pred = np.sin(lorenz_t)
std_pred = 0.2*np.cos(lorenz_t)
ax4.plot(lorenz_t, mean_pred, color="tab:blue", label="prediction")
ax4.fill_between(lorenz_t, mean_pred-std_pred, mean_pred+std_pred,
                 color="tab:blue", alpha=0.3, label="±uncertainty")
ax4.set_title("Forecast with uncertainty")
ax4.set_xlabel("time"); ax4.set_ylabel("x(t)")
ax4.legend()

ax5 = plt.subplot2grid((2, 3), (1, 2))
coeffs = np.random.randn(100, 3)  # fake ensemble coefficients
ax5.boxplot(coeffs, vert=False, labels=["x", "y", "z"])
ax5.set_title("Coefficient uncertainty (ensemble)")

plt.tight_layout()
plt.show()