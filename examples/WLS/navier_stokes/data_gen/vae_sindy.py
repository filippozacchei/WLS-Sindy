#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pysindy as ps
import sys
sys.path.append("../../../../src")  # if needed for your local sindy package   
from sindy import eWSINDy
from var_encoder_def import ConvVAE

# ============================================================
# Load dataset
# ============================================================
data = np.load("vorticity_dataset_0.05_trb.npz")
omega_list = [data[key].astype(np.float32) for key in data.files]

all_snaps = []
for i, om in enumerate(omega_list):
    if om.ndim == 3 and om.shape[2] > 1:   # (Nx, Ny, Nt)
        om_fixed = om.transpose(2, 0, 1)   # -> (Nt, Nx, Ny)
        all_snaps.append(om_fixed)
    elif om.ndim == 3:
        all_snaps.append(om.copy())
print(f"\nFinal: {len(all_snaps)} trajectories loaded")

# ============================================================
# Normalize globally (same as training)
# ============================================================
X_all = np.concatenate(all_snaps, axis=0)
mean, std = X_all.mean(), X_all.std()

# ============================================================
# Load trained VAE
# ============================================================
latent_dim = 4
device = torch.device("cpu")
model = ConvVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("vae_model_fc.pt", map_location=device))
model.eval()

# ============================================================
# Encode each trajectory separately
# ============================================================
z_list, t_list, var_blocks = [], [], []

for om in all_snaps:  # (Nt, Nx, Ny)
    Nt, Nx, Ny = om.shape

    # normalize input
    X_norm = (om - mean) / std
    X_tensor = torch.tensor(X_norm[:, None, :, :], dtype=torch.float32)

    # encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(X_tensor.to(device))
        z = mu.cpu().numpy()                    # (Nt, latent_dim)
        var = torch.exp(logvar).cpu().numpy()   # (Nt, latent_dim)

    # --- collect trajectory data ---
    z_list.append(z)                            # latent trajectory (Nt, latent_dim)
    t_list.append(np.arange(Nt)*0.05)                # assume Δt = 1
    var_blocks.append(var.mean(axis=1))         # (Nt,) average variance per timestep

print(f"Encoded {len(z_list)} latent trajectories")

# ============================================================
# Plot latent time coefficients for first two trajectories
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for j in range(z_list[0].shape[1]):  # loop over latent dims
    axes[0].plot(t_list[0], z_list[0][:, j], label=f"z{j+1}")
axes[0].set_title("Latent coefficients - Trajectory 1")
axes[0].set_ylabel("Coefficient value")
axes[0].legend(ncol=4, fontsize=8, frameon=False)

for j in range(z_list[1].shape[1]):
    axes[1].plot(t_list[1], z_list[1][:, j], label=f"z{j+1}")
axes[1].set_title("Latent coefficients - Trajectory 2")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Coefficient value")
axes[1].legend(ncol=4, fontsize=8, frameon=False)

plt.tight_layout()
plt.show()

# ============================================================
# Train SINDy with EnsembleOptimizer (new PySINDy API)
# ============================================================
from sklearn.preprocessing import StandardScaler

# --- optimizer ---
threshold = 0.01
sindy_opt = ps.EnsembleOptimizer(
    opt=ps.STLSQ(alpha=1e-12,   # base optimizer
                 threshold=threshold,
                 normalize_columns=False),
    library_ensemble=True,
    n_models=20,           # number of ensemble models
    bagging=True           # subsample trajectories
)

# --- feature library ---
sindy_library = (
    ps.PolynomialLibrary(degree=2)+ps.FourierLibrary(n_frequencies=3)
)

# --- differentiation method ---
diff_method = ps.SmoothedFiniteDifference()

# --- model ---
sindy = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
    # differentiation_method=diff_method,
    discrete_time=False    # continuous-time ODEs
)

# --- fit ---
sindy.fit(z_list, t_list)

print("\nIdentified latent ODEs:")
sindy.print()

# ============================================================
# Simulate one trajectory with SINDy
# ============================================================
z0 = z_list[0][0]  # initial condition from trajectory 0
t_span = t_list[0]

z_sindy = sindy.simulate(z0, t=t_span)

# ============================================================
# Decode latent trajectory back to fields
# ============================================================
with torch.no_grad():
    X_sindy = model.decode(torch.tensor(z_sindy, dtype=torch.float32).to(device))
    X_sindy = X_sindy.cpu().numpy()[:, 0] * std + mean  # (Nt, Nx, Ny)

# For comparison: VAE reconstructions for trajectory 0
om0 = all_snaps[0]
X_norm0 = (om0 - mean) / std
X_tensor0 = torch.tensor(X_norm0[:, None, :, :], dtype=torch.float32)
with torch.no_grad():
    X_vae_recon, _, _ = model(X_tensor0.to(device))
    X_vae_recon = X_vae_recon.cpu().numpy()[:, 0] * std + mean

# ============================================================
# Animation: true vs VAE recon vs SINDy recon (trajectory 0)
# ============================================================
vmin, vmax = om0.min(), om0.max()
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
im0 = axes[0].imshow(om0[0].T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, animated=True)
axes[0].set_title("True")
im1 = axes[1].imshow(X_vae_recon[0].T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, animated=True)
axes[1].set_title("VAE recon")
im2 = axes[2].imshow(X_sindy[0].T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, animated=True)
axes[2].set_title("SINDy (latent ODE)")
for ax in axes: ax.axis("off")

def update(k):
    im0.set_array(om0[k].T)
    im1.set_array(X_vae_recon[k].T)
    if k < len(X_sindy):
        im2.set_array(X_sindy[k].T)
    return im0, im1, im2

anim = FuncAnimation(fig, update, frames=range(0, len(om0), 5),
                     interval=100, blit=True)
anim.save("sindy_latent_vs_true.gif", dpi=120, writer="pillow")
plt.close(fig)

print("Saved GIF: sindy_latent_vs_true.gif")
