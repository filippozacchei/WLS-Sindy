import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

# ============================================================
# POD/SVD energy spectrum
# ============================================================
def compute_pod_energy(omega_list, energy_thresholds=[0.95, 0.98, 0.99]):
    """
    Compute POD (SVD) spectrum for flattened vorticity snapshots.
    
    Parameters
    ----------
    omega_list : list of arrays, each shape (N, N, Nt)
    energy_thresholds : list of floats
        Cumulative energy fractions for which to report required modes.
    
    Returns
    -------
    sing_vals : ndarray
        Singular values
    required_modes : dict
        Threshold → number of modes needed
    """
    # Stack snapshots into matrix X: (N^2, Nt_total)
    snaps = []
    for omega in omega_list:
        N, _, Nt = omega.shape
        for k in range(Nt):
            snaps.append(omega[:, :, k].reshape(-1, order="C"))
    X = np.stack(snaps, axis=1)  # (N^2, Nt_total)

    # Mean subtraction (POD convention)
    X = X - X.mean(axis=1, keepdims=True)

    # Compute SVD
    U, S, Vt = randomized_svd(X, n_components=100, random_state=0)

    # Energy content
    energy = np.cumsum(S**2) / np.sum(S**2)

    # Modes required
    required_modes = {}
    for thr in energy_thresholds:
        k = np.searchsorted(energy, thr) + 1  # +1 because searchsorted is 0-based
        required_modes[thr] = k

    # Plot spectrum
    plt.figure(figsize=(6,4))
    plt.semilogy(S**2 / np.sum(S**2), 'o-')
    plt.xlabel("Mode index")
    plt.ylabel("Normalized singular value energy")
    plt.title("POD spectrum of vorticity snapshots")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(energy, 'o-')
    plt.xlabel("Mode index")
    plt.ylabel("Cumulative energy")
    plt.title("POD cumulative energy")
    plt.axhline(0.95, color="r", ls="--")
    plt.axhline(0.98, color="g", ls="--")
    plt.axhline(0.99, color="b", ls="--")
    plt.grid(True)
    plt.show()

    return U, S, Vt, required_modes

# ============================================================
# Plot a few POD modes
# ============================================================
def plot_pod_modes(U, N, num_modes=6):
    """
    Plot the first few POD modes (reshaped left singular vectors).
    
    Parameters
    ----------
    U : ndarray
        Left singular vectors from SVD, shape (N^2, n_components)
    N : int
        Grid size (assumed square: N x N)
    num_modes : int
        Number of modes to plot
    """
    plt.figure(figsize=(12, 8))
    for i in range(num_modes):
        mode = U[:, i].reshape(N, N, order="C")
        plt.subplot(2, (num_modes + 1)//2, i+1)
        plt.contourf(mode, cmap="RdBu_r", levels=50)
        plt.colorbar()
        plt.title(f"POD mode {i+1}")
        plt.axis("equal")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
    
# ============================================================
# Usage
# ============================================================
omega_list = np.load("vorticity_dataset.npz")["omega_list"]
# shape: (n_traj, N, N, Nt)
print(len(omega_list), "trajectories loaded, ", "each shape", omega_list[0].shape)

U, sing_vals, Vt, required = compute_pod_energy(omega_list)

print("Modes required for energy thresholds:")
for thr, k in required.items():
    print(f"  {int(thr*100)}% energy → {k} modes")

# Plot first few modes
N = omega_list[0].shape[0]  # spatial grid size
plot_pod_modes(U, N, num_modes=50)

print("Modes required for energy thresholds:")
for thr, k in required.items():
    print(f"  {int(thr*100)}% energy → {k} modes")
    
import numpy as np
import matplotlib.pyplot as plt

def pod_reconstruct(omega, U, s, Vt, r=30, mean_center=True):
    Nx, Ny, Nt = omega.shape
    X = omega.reshape(Nx*Ny, Nt)

    if mean_center:
        X_mean = X.mean(axis=1, keepdims=True)
        Xc = X - X_mean
    else:
        X_mean = 0.0
        Xc = X

    r_eff = min(r, s.size)
    Xc_r = (U[:, :r_eff] * s[:r_eff]) @ Vt[:r_eff, :Nt]
    Xr = Xc_r + (X_mean if mean_center else 0.0)
    return Xr.reshape(Nx, Ny, Nt)

# ---- Example usage ----
# pick one trajectory from your loaded omega_list
omega = omega_list[0]   # shape (Nx, Ny, Nt)
omega_r = pod_reconstruct(omega, U, sing_vals, Vt, r=30, mean_center=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# assume omega (true trajectory) and omega_r (reconstruction) already exist
Nt = omega.shape[-1]

# initial fields
true0 = omega[:, :, 0]
recon0 = omega_r[:, :, 0]
err0 = np.abs(true0 - recon0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

vmin, vmax = np.min(omega), np.max(omega)

im0 = axes[0].imshow(true0.T, origin="lower", cmap="RdBu_r",
                     vmin=vmin, vmax=vmax, animated=True)
axes[0].set_title("True vorticity")

im1 = axes[1].imshow(recon0.T, origin="lower", cmap="RdBu_r",
                     vmin=vmin, vmax=vmax, animated=True)
axes[1].set_title("Reconstruction (30 modes)")

im2 = axes[2].imshow(err0.T, origin="lower", cmap="magma",
                     animated=True)
axes[2].set_title("Absolute error")

for ax in axes: ax.axis("off")

fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
plt.tight_layout()

def update(frame):
    f_true = omega[:, :, frame]
    f_recon = omega_r[:, :, frame]
    f_err = np.abs(f_true - f_recon)
    im0.set_array(f_true.T)
    im1.set_array(f_recon.T)
    im2.set_array(f_err.T)
    return im0, im1, im2

anim = FuncAnimation(fig, update, frames=range(0,Nt,10), interval=50, blit=True)

# Save as gif or mp4
anim.save("pod_reconstruction.gif", dpi=120, writer="pillow")
# anim.save("pod_reconstruction.mp4", dpi=120, fps=20,
#           extra_args=['-vcodec', 'libx264'])

plt.close(fig)
