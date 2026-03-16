# plot_utils.py
"""
Plotting utilities for Burgers and Lorenz experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Consistent colour scheme for Lorenz MF plots
COLORS_MODELS = {
    "HF":   "tab:blue",
    "LF":   "tab:orange",
    "MF":   "tab:green",
    "MF_w": "tab:red",
}

def set_dark_theme(rc=None):
    """Apply a dark theme suitable for Lorenz 3D plots."""
    base = {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "font.size": 12,
    }
    if rc is not None:
        base.update(rc)
    plt.rcParams.update(base)

L = 5.0
N = 64          # spatial resolution
Nt = 500        # number of time snapshots
T = 2.5
mu = 1.0
RT = 1.0

trajectories, grid, ts = generate_compressible_flow(
    n_traj=1,
    N=N,
    Nt=Nt,
    L=L,
    T=T,
    mu=mu,
    RT=RT,
    noise_level=0.0,     # clean reference
    seed=1,
    noise_ic=0.0,
)

u_field = trajectories[0]  # (N, N, Nt, 3)
t = ts[0]                  # (Nt,)

print("u_field shape (N, N, Nt, 3):", u_field.shape)
print("t shape:", t.shape)
print("grid shape:", grid.shape)

# Keep layout as (Nx, Ny, Nt, n_states)
U_clean = u_field

# --- Heteroscedastic noise field based on temporal derivatives --------
# Extract components: (N, N, Nt)
u   = U_clean[..., 0]
v   = U_clean[..., 1]
rho = U_clean[..., 2]

dt = t[1] - t[0]

def ddt(f, dt):
    """Centered finite difference in time with periodic wrap (time axis = 2)."""
    return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0 * dt)

u_t   = ddt(u, dt)
v_t   = ddt(v, dt)
rho_t = ddt(rho, dt)

# Magnitude of temporal derivative (N, N, Nt)
time_deriv_mag = np.sqrt(u_t**2 + v_t**2)

sigma0 = 1e-3
alpha  = 0.025   # controls noise level

variance = (sigma0 + alpha * time_deriv_mag)**2    # (N, N, Nt)
variance = np.maximum(variance, 1e-16)
std = np.sqrt(variance)

rng = np.random.default_rng(123)
noise = std[..., None] * rng.standard_normal(size=U_clean.shape)
U_noisy = U_clean + noise

print("variance shape:", variance.shape)
print("std range:", std.min(), "to", std.max())
print("U_noisy shape:", U_noisy.shape)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# --- spatial grid (same as your static code) ---
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
Xg, Yg = np.meshgrid(x, y, indexing="ij")

# --- helper to compute |u| from U[..., 0:2] ---
def velocity_magnitude(U):
    return np.sqrt(U[..., 0]**2 + U[..., 1]**2)

# Precompute velocity magnitudes over time: (N, N, Nt)
vel_clean_all = velocity_magnitude(U_clean)        # (N, N, Nt)
vel_noisy_all = velocity_magnitude(U_noisy)        # (N, N, Nt)
err_all       = np.abs(vel_noisy_all - vel_clean_all)

# Global z-scales (fixed for all frames)
zmin_vel = 0.0
zmax_vel = np.max(vel_clean_all)
zmin_err = 0.0
zmax_err = np.max(err_all)

# --- figure and axes -------------------------------------------------
fig = plt.figure(figsize=(10, 4), dpi=150)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")

for ax in (ax1, ax2):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

# fix z-scale so you see shape changes, not rescaling
ax1.set_zlim(zmin_vel, zmax_vel)
ax2.set_zlim(zmin_err, zmax_err)

# Initial frame (frame 0)
k0 = 0
vel_mag0       = vel_clean_all[:, :, k0]
vel_mag_noisy0 = vel_noisy_all[:, :, k0]
err0           = err_all[:, :, k0]

white_surface0 = np.ones_like(vel_mag0)

# Left: clean (gray) + noisy (steelblue)
ax1.plot_surface(
    Xg, Yg, vel_mag_noisy0,
    color="steelblue",
    edgecolor=None,
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.2,
)
ax1.plot_surface(
    Xg, Yg, vel_mag0,
    facecolors=plt.cm.gray(white_surface0),
    edgecolor="black",
    linewidth=0.15,
    antialiased=True,
    shade=False,
    alpha=1.0,
)

# Right: error surface (red)
ax2.plot_surface(
    Xg, Yg, err0,
    color="#b32425",
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.35,
)

plt.tight_layout(pad=0)

# --- update function for animation -----------------------------------
def update(frame):
    # Remove old surfaces cleanly
    for coll in list(ax1.collections):
        coll.remove()
    for coll in list(ax2.collections):
        coll.remove()

    vel_mag       = vel_clean_all[:, :, frame]
    vel_mag_noisy = vel_noisy_all[:, :, frame]
    err           = err_all[:, :, frame]

    white_surface = np.ones_like(vel_mag)

    # Left panel: clean + noisy
    ax1.plot_surface(
        Xg, Yg, vel_mag_noisy,
        color="steelblue",
        edgecolor=None,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.2,
    )
    ax1.plot_surface(
        Xg, Yg, vel_mag,
        facecolors=plt.cm.gray(white_surface),
        edgecolor="black",
        linewidth=0.15,
        antialiased=True,
        shade=False,
        alpha=1.0,
    )

    # Right panel: error
    ax2.plot_surface(
        Xg, Yg, err,
        color="#b32425",
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.35,
    )

    return []

# --- create animation -------------------------------------------------
# e.g. use every 10th frame so it runs faster and frame index is explicit
frame_indices = list(range(0, Nt, 10))

anim = animation.FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=50,   # ms between frames
    blit=False
)

plt.show()
